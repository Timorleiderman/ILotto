#!/usr/bin/env python3
"""Train on 1M simulated draws, validate on 200k, then score all 1661 real draws."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from scipy.stats import binom


from lotto.core import data

from lotto.core import metrics
from lotto.tickets.forecast_portfolios import ranked_portfolios, score_portfolios
from lotto.forecasting.joint_mixture import LOG_FULL_OUTCOMES
from lotto.tickets.coverage import portfolio_odds
from lotto.forecasting.synthetic_learning import SCENARIOS, components, fit_mixture, frozen_forecast, simulate_episode
from lotto.experiments.thousand_draws import summarize

ROOT = Path(__file__).resolve().parents[2]
# The dated run is pinned to the archive as it stood on 2026-11-03. Reading that saved
# copy, not the live input/ file, keeps the experiment runnable after new draws arrive.
ARCHIVE = ROOT / 'experiments/archive_2026-11-03.csv'
PROTOCOL = ROOT / 'experiments/synthetic_learning_protocol_2026_11_07.json'
# experiments/ holds the frozen evidence and is only read. A new training run and
# every evaluation write to results/ (not tracked).
SAVED_FROZEN = ROOT / 'experiments/synthetic_learning_frozen_2026_11_07.json'
SAVED_DATASET = ROOT / 'experiments/synthetic_learning_dataset_2026_11_07.npz'
OUT = ROOT / 'results/synthetic-training'
FROZEN = OUT / 'frozen.json'
DATASET = OUT / 'dataset.npz'
OUTPUT = OUT / 'real.json'
SOURCES = ['lotto/forecasting/synthetic_learning.py', 'lotto/forecasting/calibration.py', 'lotto/approaches/date_conditioned.py',
           'lotto/forecasting/joint_mixture.py', 'lotto/core/data.py', 'lotto/core/metrics.py', 'lotto/tickets/forecast_portfolios.py',
           'lotto/experiments/thousand_draws.py', 'lotto/experiments/synthetic_training.py']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        staging = Path(handle.name)
        try:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
    staging.replace(path)


def validate_protocol(config):
    if (config.get('schema') != 1 or config['draws_per_episode'] != 2000
            or config['training_episodes'] != 500 or config['validation_episodes'] != 100
            or config['scenario_cycle'] != ['fair', 'fair', 'persistent_bias', 'changing_bias']
            or config['methods'] != ['uniform', 'equal_weight_mixture', 'synthetic_fair_trained',
                                     'synthetic_mixed_trained']
            or config['weight_cleanup_threshold'] != 1e-8 or config['real_blocks'] != 4):
        raise ValueError('Require the fixed million-draw protocol and four declared methods')


def train(config):
    """This stage never loads or fits to the real archive or coverage template."""
    validate_protocol(config)
    tick = time.perf_counter()
    source_hashes = {p: sha256(ROOT / p) for p in SOURCES}
    episode_rows, episode_lp, metadata, oracle_rows = [], [], [], []
    names = None
    for split, count in enumerate([config['training_episodes'], config['validation_episodes']]):
        for episode in range(count):
            scenario = config['scenario_cycle'][episode % len(config['scenario_cycle'])]
            seed = [config['seed'], split, episode]
            d, oracle, specifications = simulate_episode(
                config['draws_per_episode'], np.random.default_rng(np.random.SeedSequence(seed)), scenario,
                change_every=config['change_every'], multipliers=config['bias_multipliers'])
            component = components(d, windows=tuple(config['windows']), strengths=tuple(config['prior_draws']))
            if names is not None and names != component['names']:
                raise RuntimeError('Component order changed between episodes')
            names = component['names']
            episode_rows.append(np.column_stack([d.balls, d.strong]).astype(np.uint8))
            episode_lp.append(component['log_prob'])
            oracle_rows.append(oracle)
            metadata.append({'split': 'train' if split == 0 else 'validation', 'episode': episode,
                             'scenario': scenario, 'seed_sequence': seed, 'blocks': specifications})
            if (episode + 1) % 25 == 0:
                print(f"{'Training' if split == 0 else 'Validation'} simulation: "
                      f'{episode + 1}/{count} episodes; {time.perf_counter() - tick:.1f}s', flush=True)
    lp = np.asarray(episode_lp)
    del episode_lp
    train_lp = lp[:config['training_episodes']].transpose(1, 0, 2).reshape(len(names), -1)
    fair_indices = [i for i, row in enumerate(metadata[:config['training_episodes']]) if row['scenario'] == 'fair']
    fair_lp = lp[fair_indices].transpose(1, 0, 2).reshape(len(names), -1)
    fitted = {}
    for name, values in [('synthetic_fair_trained', fair_lp), ('synthetic_mixed_trained', train_lp)]:
        print(f'Fitting {name} on {values.shape[1]:,} synthetic examples', flush=True)
        fitted[name] = fit_mixture(values, max_iterations=config['max_iterations'],
                                   tolerance=config['optimizer_tolerance'])
        fitted[name]['training_draws'] = values.shape[1]
    uniform = np.zeros(len(names))
    uniform[-1] = 1
    weights = {'uniform': uniform.tolist(), 'equal_weight_mixture': np.full(len(names), 1 / len(names)).tolist(),
               **{name: row['weights'] for name, row in fitted.items()}}
    validation = []
    for i in range(config['training_episodes'], len(metadata)):
        component = {'names': names, 'log_prob': lp[i], 'ball_p': None, 'strong_p': None}
        validation.append({'episode': metadata[i]['episode'], 'scenario': metadata[i]['scenario'],
                           'oracle_mean_log_loss': float(-np.mean(oracle_rows[i])),
                           'models': {name: float(-frozen_forecast(component, w)['log_prob'].mean())
                                      for name, w in weights.items()}})
    summaries = {}
    for scenario in SCENARIOS:
        rows = [r for r in validation if r['scenario'] == scenario]
        summaries[scenario] = {'episodes': len(rows), 'draws': len(rows) * config['draws_per_episode'],
                               'oracle_log_loss_difference_vs_uniform':
                                   float(np.mean([r['oracle_mean_log_loss'] for r in rows]) - LOG_FULL_OUTCOMES),
                               'models': {name: {'log_loss_difference_vs_uniform':
                                                float(np.mean([r['models'][name] for r in rows]) - LOG_FULL_OUTCOMES)}
                                          for name in weights}}
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-learning-', dir=DATASET.parent) as directory:
        staged = Path(directory) / DATASET.name
        np.savez_compressed(staged, draws=np.asarray(episode_rows),
                            split=np.asarray([r['split'] for r in metadata]),
                            scenario=np.asarray([r['scenario'] for r in metadata]),
                            oracle_log_prob=np.asarray(oracle_rows))
        staged.replace(DATASET)
    result = {'schema': 1, 'status': 'frozen_before_real_evaluation', 'protocol': config,
              'protocol_sha256': sha256(PROTOCOL), 'source_sha256': source_hashes,
              'component_names': names, 'weights': weights, 'fitted': fitted,
              'training_draws': train_lp.shape[1], 'validation_draws': config['validation_episodes'] *
              config['draws_per_episode'], 'dataset': {'path': DATASET.name, 'sha256': sha256(DATASET)},
              'generation': metadata, 'validation': summaries, 'validation_episodes': validation,
              'seconds': time.perf_counter() - tick,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__}}
    # Persist the learned settings before a separate stage can open real outcomes.
    write_json(FROZEN, result)
    print(f'Frozen synthetic-trained settings saved: {FROZEN}', flush=True)
    return result


def evaluate(config):
    """Score the frozen procedures once, in chronological order, including cold start."""
    validate_protocol(config)
    # Use the weights from a local training run if there is one, else the saved ones.
    frozen_path, dataset_path = (FROZEN, DATASET) if FROZEN.exists() else (SAVED_FROZEN, SAVED_DATASET)
    frozen = json.loads(frozen_path.read_text())
    if frozen['protocol_sha256'] != sha256(PROTOCOL) or frozen['dataset']['sha256'] != sha256(dataset_path):
        raise ValueError('Frozen training inputs do not match')
    # Inputs are still verified by hash. Source files are not: the freeze recorded
    # them by their pre-restructure paths (commit e6a792c reproduces that state
    # exactly); since then, unchanged numbers are guarded by tests/test_golden.py.
    for filename, expected in config['input_sha256'].items():
        if sha256(ARCHIVE if filename == data.RAW_CSV else ROOT / filename) != expected:
            raise ValueError(f'Input changed since freeze: {filename}')
    draws = data.load(raw_csv=str(ARCHIVE), clean_csv=None, refresh=False)
    if len(draws) != 1661:
        raise ValueError('Require the fixed 1661-draw modern archive')
    component = components(draws, windows=tuple(config['windows']), strengths=tuple(config['prior_draws']),
                           include_marginals=True)
    if component['names'] != frozen['component_names']:
        raise ValueError('Real component names differ from frozen training names')
    template = json.loads((ROOT / 'experiments/ticket_chances_2026_11_07.json').read_text())['selected_portfolio']
    p1, p10 = portfolio_odds([template[0]])['p_any_prize'], portfolio_odds(template)['p_any_prize']
    dates = draws.dates.astype('datetime64[D]').astype(str)
    arrays, summaries = {}, []
    for name, weights in frozen['weights'].items():
        forecast = frozen_forecast(component, weights)
        one = np.asarray([[*(metrics.top_k(p) + 1), int(s.argmax()) + 1]
                          for p, s in zip(forecast['ball_p'], forecast['strong_p'])])[:, None, :]
        ten = ranked_portfolios(template, forecast['ball_p'], forecast['strong_p'])
        c1, s1 = score_portfolios(one, draws.balls, draws.strong)
        c10, s10 = score_portfolios(ten, draws.balls, draws.strong)
        row = {'name': name, 'one_ticket': summarize(c1, s1, p1), 'ten_tickets': summarize(c10, s10, p10),
               'joint_log_loss': float(-forecast['log_prob'].mean()),
               'joint_loss_difference_vs_uniform': float(-forecast['log_prob'].mean() - LOG_FULL_OUTCOMES),
               'blocks': []}
        for ix in np.array_split(np.arange(len(draws)), config['real_blocks']):
            row['blocks'].append({'start': dates[ix[0]], 'end': dates[ix[-1]],
                                  'one_ticket': summarize(c1[ix], s1[ix], p1),
                                  'ten_tickets': summarize(c10[ix], s10[ix], p10)})
        summaries.append(row)
        for suffix, values in [('single_tickets', one.astype(np.uint8)), ('portfolios', ten.astype(np.uint8)),
                               ('single_matches', c1.astype(np.uint8)), ('portfolio_matches', c10.astype(np.uint8)),
                               ('single_strong_hits', s1), ('portfolio_strong_hits', s10),
                               ('joint_log_prob', forecast['log_prob']), ('ball_p', forecast['ball_p']),
                               ('strong_p', forecast['strong_p'])]:
            arrays[f'{name}_{suffix}'] = values
    verdict = metrics.holm_bonferroni({f"{r['name']}|{budget}": r[budget]['nominal_p_greater']
                                     for r in summaries for budget in ('one_ticket', 'ten_tickets')})
    for row in summaries:
        for budget in ('one_ticket', 'ten_tickets'):
            row[budget]['significant_after_holm_8'] = verdict[f"{row['name']}|{budget}"]
    result = {'schema': 1, 'status': 'complete', 'frozen_training_sha256': sha256(frozen_path),
              'protocol_sha256': sha256(PROTOCOL), 'source_sha256': frozen['source_sha256'],
              'draws': len(draws), 'start': dates[0], 'end': dates[-1], 'template': template,
              'models': summaries, 'significant_comparisons': [k for k, v in verdict.items() if v],
              'fair_null': {budget: {'p_any_prize': p, 'expected_winning_draws': len(draws) * p,
                                     'central_95_percent_count_range':
                                         binom.ppf([.025, .975], len(draws), p).astype(int).tolist()}
                            for budget, p in [('one_ticket', p1), ('ten_tickets', p10)]}}
    out_arrays = OUTPUT.with_suffix('.npz')
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-real-', dir=OUTPUT.parent) as directory:
        staged = Path(directory) / out_arrays.name
        np.savez_compressed(staged, dates=dates, actual_balls=draws.balls, actual_strong=draws.strong, **arrays)
        result['per_draw_artifact'] = {'path': out_arrays.name, 'sha256': sha256(staged)}
        staged.replace(out_arrays)
    write_json(OUTPUT, result)
    for row in summaries:
        print(f"{row['name']}: one={row['one_ticket']['winning_draws_3plus']}; "
              f"ten={row['ten_tickets']['winning_draws_3plus']}; "
              f"loss minus uniform={row['joint_loss_difference_vs_uniform']:+.8f}", flush=True)
    print(f'Corrected significant comparisons: {result["significant_comparisons"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'evaluate', 'all'], default='evaluate',
                        help='evaluate: score the saved trained weights on the real draws (default, fast); '
                             'train: regenerate the million simulated draws and refit; all: both')
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    if args.stage in ('train', 'all'):
        train(config)
    if args.stage in ('evaluate', 'all'):
        evaluate(config)


if __name__ == '__main__':
    main()
