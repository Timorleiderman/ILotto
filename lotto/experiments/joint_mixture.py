#!/usr/bin/env python3
"""Reproduce the fixed exploratory protocol: joint mixtures and synthetic controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from lotto.core import data, metrics
from lotto.forecasting.joint_mixture import FAMILY_PRIOR, LOG_FULL_OUTCOMES, build_experts, mix_experts
from lotto.tickets.coverage import best_match_counts, portfolio_odds, refine_coverage_tickets

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('static_joint_mixture', 'fixed_share_joint_mixture')
FOLD_COUNT = 4
FOLD_DRAWS = 300


# The dated run is pinned to the archive as it stood on 2026-11-03. Reading that saved
# copy, not the live input/ file, keeps the experiment runnable after new draws arrive.
ARCHIVE = ROOT/'experiments/archive_2026-11-03.csv'


def _int_in_range(value, low, high):
    return type(value) is int and low <= value <= high


def _real(value, low, high):
    return type(value) in (int, float) and np.isfinite(value) and low <= value <= high


def validate_protocol(config, n_draws):
    """Reject ambiguous or invalid experiments before producing any artifacts."""
    if config.get('schema') != 1 or config.get('family_prior_mass') != FAMILY_PRIOR:
        raise ValueError('Protocol must match schema 1 and the implemented family prior')
    if (config.get('primary_method'), config.get('secondary_method')) != METHODS:
        raise ValueError(f'Protocol method names must be {METHODS}, in that order')
    history = config['min_history']
    if (not _int_in_range(history, 1, n_draws - FOLD_COUNT * FOLD_DRAWS)
            or not _int_in_range(config['primary_test_draws'], 1, n_draws-history)):
        raise ValueError('Require a valid primary test window and at least four disjoint 300-draw folds')
    if (not _real(config['fixed_share_rate'], 0, 1)
            or not _real(config['sparse_prior_draws'], np.finfo(float).tiny, np.finfo(float).max)):
        raise ValueError('Require share in [0,1] and a positive finite sparse prior')
    windows, strengths = config['dense_windows'], config['dense_prior_draws']
    if (not windows or any(w is not None and not _int_in_range(w, 1, sys.maxsize) for w in windows)
            or not strengths
            or any(not _real(s, np.finfo(float).tiny, np.finfo(float).max) for s in strengths)
            or len(set(windows)) != len(windows) or len(set(strengths)) != len(strengths)):
        raise ValueError('Require nonempty, unique, positive dense model windows and strengths')
    synthetic = config['synthetic']
    if (not _int_in_range(synthetic['draws'], history+1, sys.maxsize)
            or not _int_in_range(synthetic['replicates_per_scenario'], 1, sys.maxsize)
            or not _int_in_range(synthetic['seed'], 0, sys.maxsize)
            or not _real(synthetic['diagnostic_threshold'], np.nextafter(1.0, np.inf), np.finfo(float).max)):
        raise ValueError('Require nonempty synthetic trials, sufficient history, and an evidence threshold > 1')
    scenarios = synthetic['scenarios']
    if (not scenarios or any(not isinstance(s.get('name'), str) or not s['name'].strip() for s in scenarios)
            or len({s['name'] for s in scenarios}) != len(scenarios)
            or any(not _real(s['multiplier'], 0, data.N_NUMBERS/data.N_DRAWN)
                   or not _int_in_range(s['bias_starts'], 0, synthetic['draws']) for s in scenarios)):
        raise ValueError('Require unique scenario names, valid inclusion probabilities, and in-range bias starts')
    probe = config['coverage_probe']
    if (not _int_in_range(probe['seeds'], 1, sys.maxsize)
            or not _int_in_range(probe['additional_exact_passes'], 0, 20)):
        raise ValueError('Require a positive coverage-probe count and 0..20 exact passes')


def build(draws, config, *, marginals=True):
    return build_experts(draws, min_history=config['min_history'], sparse_prior=config['sparse_prior_draws'],
                         windows=tuple(config['dense_windows']), strengths=tuple(config['dense_prior_draws']),
                         include_marginals=marginals)


def mixtures(experts, config):
    """The protocol's sequential mixtures of the experts, keyed by method name."""
    if (config['primary_method'], config['secondary_method']) != METHODS:
        raise ValueError(f'Protocol method names must be {METHODS}, in that order')
    return {config['primary_method']: mix_experts(experts.log_prob, experts.prior),
            config['secondary_method']: mix_experts(experts.log_prob, experts.prior,
                                                   share=config['fixed_share_rate'])}


def interval(values, *, seed=61106):
    """Exploratory circular block bootstrap, matching the preceding research."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(4000, (len(values) + 19) // 20))
    indices = (starts[:, :, None] + np.arange(20)) % len(values)
    means = np.asarray(values)[indices.reshape(4000, -1)[:, :len(values)]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def wilson(successes, n):
    """Wilson 95% interval for a binomial proportion."""
    p, z = successes/n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(max(0, center-half)), float(min(1, center+half))]


def real_comparison(draws, config, npz_path):
    """Score every expert and mixture on the real draws; write per-draw arrays to ``npz_path``."""
    validate_protocol(config, len(draws))
    tick = time.perf_counter()
    e = build(draws, config)
    mixes = mixtures(e, config)
    names, families = e.names.copy(), e.families.copy()
    lp, bp, sp = list(e.log_prob), list(e.ball_p), list(e.strong_p)
    for name, m in mixes.items():
        names.append(name)
        families.append('mixture')
        lp.append(m['log_prob'])
        bp.append(np.einsum('te,etb->tb', m['weights_before_draw'], e.ball_p))
        sp.append(np.einsum('te,ets->ts', m['weights_before_draw'], e.strong_p))
    lp, bp, sp = np.asarray(lp), np.asarray(bp), np.asarray(sp)
    actual, strong = draws.balls[e.targets], draws.strong[e.targets]
    count = np.asarray([[metrics.matches(p, a) for p, a in zip(row, actual)] for row in bp])
    ll = np.asarray([[metrics.log_loss(p, a) for p, a in zip(row, actual)] for row in bp])
    strong_hit = (sp.argmax(axis=2) + 1 == strong)
    rows = []
    n_test = config['primary_test_draws']
    for j, (name, family) in enumerate(zip(names, families)):
        row = {'name': name, 'family': family,
               **metrics.summarise(count[j, -n_test:], ll[j, -n_test:], strong_hit[j, -n_test:])}
        delta = -lp[j] - LOG_FULL_OUTCOMES
        row.update(mean_joint_log_loss=float(-lp[j, -n_test:].mean()),
                   joint_loss_difference_vs_uniform=float(delta[-n_test:].mean()),
                   all_evaluation_joint_log_loss=float(-lp[j].mean()),
                   all_evaluation_mean_matches=float(count[j].mean()),
                   joint_loss_difference_95_block_interval=interval(delta[-n_test:]),
                   final_log_e=float(-delta.sum()),
                   max_log_e_including_start=float(max(0, (-delta).cumsum().max())))
        row['folds'] = []
        for block in range(FOLD_COUNT, 0, -1):
            start, stop = len(e.targets) - block*FOLD_DRAWS, len(e.targets) - (block-1)*FOLD_DRAWS
            ix = slice(start, stop)
            row['folds'].append({'start': str(draws.dates[e.targets[start]].astype('datetime64[D]')),
                                 'end': str(draws.dates[e.targets[stop-1]].astype('datetime64[D]')),
                                 'n_draws': FOLD_DRAWS, 'mean_matches': float(count[j, ix].mean()),
                                 'mean_joint_log_loss': float(-lp[j, ix].mean()),
                                 'joint_loss_difference_vs_uniform': float(delta[ix].mean())})
        rows.append(row)
    corrected = metrics.holm_bonferroni({row['name']: row['p_value'] for row in rows})
    for row in rows:
        row['accuracy_significant_after_holm'] = corrected[row['name']]
    weights = {name: {expert: float(w) for expert, w in zip(e.names, m['next_weights'])}
               for name, m in mixes.items()}
    np.savez_compressed(npz_path, names=np.asarray(names), families=np.asarray(families),
                        dates=draws.dates[e.targets].astype('datetime64[D]').astype(str),
                        joint_log_prob=lp, ball_marginals=bp, strong_probabilities=sp,
                        main_matches=count, strong_hits=strong_hit, ball_log_loss=ll,
                        expert_prior=e.prior,
                        **{f'{name}_weights_before_draw': m['weights_before_draw'] for name, m in mixes.items()})
    return {'n_draws': len(draws), 'evaluation_draws': len(e.targets), 'n_experts': len(e.names),
            'test_start': str(draws.dates[-n_test].astype('datetime64[D]')),
            'test_end': str(draws.dates[-1].astype('datetime64[D]')), 'rows': rows,
            'next_expert_weights': weights, 'seconds': time.perf_counter()-tick}


def synthetic_draws(n, rng, multiplier, bias_starts):
    """A fair history in which ball 1 is drawn ``multiplier`` times as often from ``bias_starts`` on."""
    probability = np.full(n, 6/37)
    probability[bias_starts:] *= multiplier
    present = rng.random(n) < probability
    other = np.argsort(rng.random((n, 36)), axis=1)[:, :6] + 2
    other[present, 5] = 1
    return data.Draws(np.datetime64('2000-01-01') + np.arange(n), np.sort(other, axis=1),
                      rng.integers(1, 8, n))


def controls(config):
    """Run the planted-bias simulations: can the mixture detect a bias it was not told about?"""
    settings = config['synthetic']
    rows = []
    for scenario_index, scenario in enumerate(settings['scenarios']):
        trials = []
        tick = time.perf_counter()
        for rep in range(settings['replicates_per_scenario']):
            rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], scenario_index, rep]))
            draws = synthetic_draws(settings['draws'], rng, scenario['multiplier'], scenario['bias_starts'])
            e = build(draws, config, marginals=False)
            trial = {'replicate': rep}
            for name, m in mixtures(e, config).items():
                path = m['log_e_path']
                crossed = np.flatnonzero(path >= np.log(settings['diagnostic_threshold']))
                trial[name] = {'final_log_e': float(path[-1]), 'max_log_e': float(max(0, path.max())),
                               'crossed': bool(len(crossed)),
                               'first_crossing_draw_index': int(e.targets[crossed[0]]) if len(crossed) else None}
            trials.append(trial)
            if (rep+1) % 16 == 0:
                print(f"{scenario['name']}: {rep+1}/{settings['replicates_per_scenario']}", flush=True)
        summary = {}
        for name in [config['primary_method'], config['secondary_method']]:
            hits = sum(t[name]['crossed'] for t in trials)
            summary[name] = {'crossings': hits, 'trials': len(trials), 'crossing_rate': hits/len(trials),
                             'wilson_95_interval': wilson(hits, len(trials)),
                             'median_final_log_e': float(np.median([t[name]['final_log_e'] for t in trials]))}
        rows.append({'scenario': scenario, 'summary': summary, 'trials': trials,
                     'seconds': time.perf_counter()-tick})
    return rows


def coverage_probe(config):
    """Check whether exact refinement can still improve the saved coverage portfolios."""
    previous = json.loads((ROOT/'experiments/coverage_refinement_2026_11_06.json').read_text())
    if config['coverage_probe']['seeds'] > len(previous['rows']):
        raise ValueError('Coverage-probe count exceeds the available initial portfolios')
    rows = []
    for old in previous['rows'][:config['coverage_probe']['seeds']]:
        tick = time.perf_counter()
        new, audit = refine_coverage_tickets(old['refined_portfolio'],
                                              max_passes=config['coverage_probe']['additional_exact_passes'])
        odds = portfolio_odds(new)
        reference = best_match_counts([ticket[:6] for ticket in new], engine='reference')
        np.testing.assert_array_equal(reference, odds['best_match_counts'])
        if odds['p_any_prize'] < old['after']['p_any_prize']:
            raise RuntimeError('Exact refinement reduced coverage')
        rows.append({'seed': old['seed'], 'before': old['after'], 'after': odds, 'portfolio': new,
                     'audit': audit, 'reference_verified': True, 'seconds': time.perf_counter()-tick})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'experiments/algorithm_research_protocol_2026_11_06.json')
    parser.add_argument('--out', type=Path, default=ROOT/'results/joint-mixture/joint_mixture.json')
    args = parser.parse_args()
    inputs = {args.protocol.resolve(), (ARCHIVE).resolve(),
              (ROOT/'experiments/coverage_refinement_2026_11_06.json').resolve()}
    npz = args.out.with_suffix('.npz')
    if args.out.suffix.lower() != '.json' or inputs.intersection({args.out.resolve(), npz.resolve()}):
        parser.error('out must be a .json path with both output files distinct from the experiment inputs')
    config = json.loads(args.protocol.read_text())
    archive_hash = hashlib.sha256((ARCHIVE).read_bytes()).hexdigest()
    if archive_hash != config['archive_sha256']:
        raise ValueError('Protocol must match the archive')
    draws = data.load(raw_csv=str(ARCHIVE), clean_csv=None, refresh=False)
    validate_protocol(config, len(draws))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Failed or interrupted computation must not replace completed evidence.
    # Publish each file by same-filesystem rename only after all checks finish;
    # JSON is last and contains the array checksum for reader verification.
    with tempfile.TemporaryDirectory(prefix='joint-research-', dir=args.out.parent) as staging:
        staged_npz = Path(staging)/npz.name
        staged_json = Path(staging)/args.out.name
        result = {'schema': 1, 'status': 'complete', 'protocol': config,
                  'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                  'platform': platform.platform()},
                  'uniform_full_draw_log_loss': LOG_FULL_OUTCOMES,
                  'bootstrap': {'replicates': 4000, 'block_length': 20, 'seed': 61106,
                                'scope': 'exploratory, individual intervals, not family corrected'},
                  'real': real_comparison(draws, config, staged_npz)}
        print('Real-data comparison complete; retaining previous files until all checks finish', flush=True)
        result['synthetic'] = controls(config)
        result['coverage_probe'] = coverage_probe(config)
        result['per_draw_artifact'] = {'path': npz.name, 'sha256': hashlib.sha256(staged_npz.read_bytes()).hexdigest()}
        staged_json.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        staged_npz.replace(npz)
        staged_json.replace(args.out)
    print(f'Wrote {args.out} and {npz}', flush=True)


if __name__ == '__main__':
    main()
