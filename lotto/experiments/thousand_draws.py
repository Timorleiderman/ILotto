#!/usr/bin/env python3
"""Evaluate 59 past-only forecasts over 1000 draws at one- and ten-ticket budgets."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import binom

from lotto.core import data, metrics, paths
from lotto.core.files import sha256_file
from lotto.experiments.joint_mixture import build, mixtures
from lotto.forecasting.joint_mixture import LOG_FULL_OUTCOMES
from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.forecast_portfolios import longest_run, ranked_portfolios, score_portfolios
from lotto.tickets.strong_assignment import strong_prize_odds

ROOT = paths.REPO_ROOT
# The dated run is pinned to the archive as it stood on 2026-11-03. Reading that saved
# copy, not the live input/ file, keeps the experiment runnable after new draws arrive.
ARCHIVE = ROOT / 'experiments/archive_2026-11-03.csv'
PROTOCOL = ROOT / 'experiments/historical_strikes_protocol_2026_11_07.json'
# New runs go to results/ (not tracked); experiments/ holds the frozen evidence and is only read.
OUTPUT = ROOT / 'results/thousand-draws/historical_strikes.json'


def summarize(matches, strong_hit, p_win):
    """Winning-draw counts, the fair expectation and a binomial p-value for one budget."""
    best = matches.max(axis=1)
    wins = best >= 3
    count, n = int(wins.sum()), len(wins)
    return {'draws': n, 'winning_draws_3plus': count, 'win_rate': count / n,
            'expected_winning_draws_under_fair_null': n * p_win,
            'nominal_p_greater': float(binom.sf(count - 1, n, p_win)),
            'winning_draws_3plus_and_strong': int(((matches >= 3) & strong_hit).any(axis=1).sum()),
            'winning_draws_4plus': int((best >= 4).sum()),
            'winning_draws_5plus': int((best >= 5).sum()),
            'winning_draws_six_main': int((best == 6).sum()),
            'jackpot_draws': int(((matches == 6) & strong_hit).any(axis=1).sum()),
            'mean_best_main_matches': float(best.mean()), 'longest_winning_run': longest_run(wins)}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if (config.get('schema') != 1 or config['n_test'] != 1000 or config['block_draws'] != 250
            or config['selection_draws'] != 500):
        raise ValueError('Require the fixed 1000-draw protocol, four 250-draw blocks and a 500-draw selector')
    for filename, expected in config['input_sha256'].items():
        if sha256_file(ARCHIVE if filename == data.RAW_CSV_IN_REPO else ROOT / filename) != expected:
            raise ValueError(f'Input changed since protocol freeze: {filename}')
    draws = data.load(raw_csv=str(ARCHIVE), clean_csv=None, refresh=False)
    n = config['n_test']
    if len(draws) <= n:
        raise ValueError('Require training history before all 1000 test draws')
    saved = json.loads((ROOT / 'experiments/joint_mixture_research_2026_11_06.json').read_text())
    template = json.loads((ROOT / 'experiments/ticket_chances_2026_11_07.json').read_text())['selected_portfolio']
    if len(template) != 10:
        raise ValueError('Require the ten-ticket template')
    experts = build(draws, saved['protocol'])
    mixed = mixtures(experts, saved['protocol'])
    names = experts.names + list(mixed)
    balls = np.asarray([*experts.ball_p, *[np.einsum('te,etb->tb', m['weights_before_draw'], experts.ball_p)
                                         for m in mixed.values()]])
    strong = np.asarray([*experts.strong_p, *[np.einsum('te,ets->ts', m['weights_before_draw'], experts.strong_p)
                                           for m in mixed.values()]])
    log_prob = np.asarray([*experts.log_prob, *[m['log_prob'] for m in mixed.values()]])
    with np.load(ROOT / 'experiments/joint_mixture_research_2026_11_06.npz', allow_pickle=False) as previous:
        np.testing.assert_array_equal(names, previous['names'])
        # Probabilities are recomputed, so they agree to rounding (about 1e-15 between
        # CPU families), not bit for bit. 1e-12 is far below anything that could move a
        # ticket or a reported number.
        np.testing.assert_allclose(balls, previous['ball_marginals'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(strong, previous['strong_probabilities'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(log_prob, previous['joint_log_prob'], rtol=1e-12, atol=0)
        np.testing.assert_array_equal(previous['dates'][-n:], draws.dates[-n:].astype('datetime64[D]').astype(str))
    print('Reproduced all 59 original past-only forecast paths (to 1e-12)', flush=True)
    balls, strong, log_prob = balls[:, -n:], strong[:, -n:], log_prob[:, -n:]
    actual, actual_strong = draws.balls[-n:], draws.strong[-n:]
    dates = draws.dates[-n:].astype('datetime64[D]').astype(str)
    one_p = portfolio_odds([template[0]])['p_any_prize']
    template_odds = portfolio_odds(template)
    ten_p = template_odds['p_any_prize']
    models, single_tickets, portfolios, one_counts, ten_counts, one_strong, ten_strong = [], [], [], [], [], [], []
    for model, name in enumerate(names):
        top6 = np.sort(np.argsort(-balls[model], axis=1, kind='stable')[:, :6], axis=1) + 1
        top_strong = np.argmax(strong[model], axis=1) + 1
        one = np.concatenate([top6, top_strong[:, None]], axis=1)[:, None, :]
        ten = ranked_portfolios(template, balls[model], strong[model])
        c1, h1 = score_portfolios(one, actual, actual_strong)
        c10, h10 = score_portfolios(ten, actual, actual_strong)
        # Not compared with the saved run: its single tickets were picked before ties
        # between equal scores were given a fixed rule (lowest number first), so tied
        # forecasts, the uniform one above all, now choose different numbers.
        row = {'name': name, 'one_ticket': summarize(c1, h1, one_p), 'ten_tickets': summarize(c10, h10, ten_p),
               'joint_log_loss': float(-log_prob[model].mean()),
               'joint_loss_difference_vs_uniform': float((-log_prob[model] - LOG_FULL_OUTCOMES).mean()),
               'blocks': []}
        for start in range(0, n, config['block_draws']):
            stop = start + config['block_draws']
            row['blocks'].append({'start': dates[start], 'end': dates[stop - 1],
                                  'one_ticket': summarize(c1[start:stop], h1[start:stop], one_p),
                                  'ten_tickets': summarize(c10[start:stop], h10[start:stop], ten_p)})
        models.append(row)
        single_tickets.append(one)
        portfolios.append(ten)
        one_counts.append(c1)
        ten_counts.append(c10)
        one_strong.append(h1)
        ten_strong.append(h10)
    p_values = {f'{row["name"]}|{budget}': row[budget]['nominal_p_greater']
                for row in models for budget in ('one_ticket', 'ten_tickets')}
    verdicts = metrics.holm_bonferroni(p_values)
    for row in models:
        for budget in ('one_ticket', 'ten_tickets'):
            row[budget]['significant_after_holm_118'] = verdicts[f'{row["name"]}|{budget}']
    half = config['selection_draws']
    winner = max(range(len(names)), key=lambda i: int((ten_counts[i][:half].max(axis=1) >= 3).sum()))
    selection = {'name': names[winner], 'model_index': winner,
                 'selected_on_first': {'start': dates[0], 'end': dates[half - 1],
                                      **summarize(ten_counts[winner][:half], ten_strong[winner][:half], ten_p)},
                 'evaluated_on_second': {'start': dates[half], 'end': dates[-1],
                                         **summarize(ten_counts[winner][half:], ten_strong[winner][half:], ten_p)},
                 'interpretation': 'Model choice fixed after first 500; online parameter updates continue. '
                                   'This is retrospective and not a fresh holdout of the research process.'}
    null = {budget: {'p_any_prize': p, 'expected_wins': n * p,
                     'central_95_percent_win_count_range': binom.ppf([0.025, 0.975], n, p).astype(int).tolist()}
            for budget, p in [('one_ticket', one_p), ('ten_tickets', ten_p)]}
    source_files = ['lotto/tickets/forecast_portfolios.py', 'lotto/forecasting/joint_mixture.py', 'lotto/forecasting/calibration.py',
                    'lotto/approaches/date_conditioned.py', 'lotto/core/metrics.py', 'lotto/experiments/joint_mixture.py',
                    'lotto/experiments/thousand_draws.py']
    result = {'schema': 1, 'status': 'complete', 'configuration': config, 'protocol_sha256': sha256_file(PROTOCOL),
              'source_sha256': {p: sha256_file(ROOT / p) for p in source_files},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'platform': platform.platform()},
              'training_draws_before_first_target': len(draws) - n, 'test_start': dates[0], 'test_end': dates[-1],
              'original_forecasts_reproduced_exactly': True, 'fair_null': null,
              'template': template, 'template_odds': template_odds,
              'template_strong_odds': strong_prize_odds(template),
              'models': models, 'first_half_selection': selection,
              'significant_comparisons': [key for key, passed in verdicts.items() if passed]}
    out_arrays = OUTPUT.with_suffix('.npz')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='historical-strikes-', dir=OUTPUT.parent) as staging:
        staged_arrays = Path(staging) / out_arrays.name
        staged_json = Path(staging) / OUTPUT.name
        np.savez_compressed(staged_arrays, names=np.asarray(names), dates=dates,
                            actual_balls=actual, actual_strong=actual_strong,
                            single_tickets=np.asarray(single_tickets, dtype=np.uint8),
                            portfolios=np.asarray(portfolios, dtype=np.uint8),
                            single_main_matches=np.asarray(one_counts, dtype=np.uint8),
                            portfolio_main_matches=np.asarray(ten_counts, dtype=np.uint8),
                            single_strong_hits=np.asarray(one_strong), portfolio_strong_hits=np.asarray(ten_strong),
                            joint_log_prob=log_prob)
        result['per_draw_artifact'] = {'path': out_arrays.name, 'sha256': sha256_file(staged_arrays)}
        staged_json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staged_arrays.replace(out_arrays)
        staged_json.replace(OUTPUT)
    for budget in ('one_ticket', 'ten_tickets'):
        best = max(models, key=lambda row: row[budget]['winning_draws_3plus'])
        print(f"Best {budget}: {best['name']} {best[budget]['winning_draws_3plus']}/{n}, "
              f"nominal p={best[budget]['nominal_p_greater']:.6g}, "
              f"Holm={best[budget]['significant_after_holm_118']}", flush=True)
    print(f"First-half selection: {selection['name']}; "
          f"{selection['selected_on_first']['winning_draws_3plus']}/{half} then "
          f"{selection['evaluated_on_second']['winning_draws_3plus']}/{n-half}", flush=True)
    print(f'Wrote {OUTPUT} and {out_arrays}', flush=True)


if __name__ == '__main__':
    main()
