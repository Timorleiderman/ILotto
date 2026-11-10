#!/usr/bin/env python3
"""Fixed ten-ticket experiment: broader coverage search and exact strong assignment."""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.portfolio import best_match_counts, portfolio_odds, search_coverage_tickets  # noqa: E402
from bench.strong_assignment import optimize_strong_numbers, strong_prize_odds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / 'experiments/ticket_chances_protocol_2026_11_07.json'
OUTPUT = ROOT / 'experiments/ticket_chances_2026_11_07.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_mains(tickets, odds):
    reference = best_match_counts([row[:6] for row in tickets], engine='reference')
    np.testing.assert_array_equal(reference, odds['best_match_counts'])


def verify_strong(tickets, expected):
    reference = strong_prize_odds(tickets, engine='reference')
    if reference['covered_full_outcomes'] != expected:
        raise RuntimeError('Independent strong-prize coverage verification failed')


def main():
    config = json.loads(PROTOCOL.read_text())
    source = ROOT / 'experiments' / config['initial_results']
    if config.get('schema') != 1 or sha256(source) != config['initial_results_sha256']:
        raise ValueError('Protocol must match schema 1 and the saved input checksum')
    previous = json.loads(source.read_text())
    seeds = config['seeds']
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset({row['seed'] for row in previous['rows']})):
        raise ValueError('Require distinct nonnegative seeds present in the input')
    widths = config['perturbation_counts']
    if (not widths or any(type(width) is not int or not 1 <= width <= 100 for width in widths)
            or len(set(widths)) != len(widths)):
        raise ValueError('Require distinct perturbation counts in 1..100')
    baseline = []
    for seed in seeds:
        old = next(row for row in previous['rows'] if row['seed'] == seed)
        tickets = old['portfolio']
        odds = portfolio_odds(tickets)
        if len(tickets) != 10 or odds['distinct_main_tickets'] != 10 or odds != old['after']:
            raise ValueError('Require ten distinct main tickets and reproducible saved baseline odds')
        verify_mains(tickets, odds)
        reassigned, audit = optimize_strong_numbers(tickets)
        verify_strong(tickets, audit['covered_full_outcomes_before'])
        verify_strong(reassigned, audit['covered_full_outcomes_after'])
        baseline.append({'seed': seed, 'portfolio': tickets, 'main_odds': odds,
                         'strong_optimized_portfolio': reassigned, 'strong_assignment': audit,
                         'reference_verified': True})
        print(f"Baseline {seed}: any prize {odds['p_any_prize']:.8%}; "
              f"3+ and strong {audit['probability_before']:.8%} -> {audit['probability_after']:.8%}",
              flush=True)
    results = []
    for width in widths:
        for old in baseline:
            tick = time.perf_counter()
            tickets, search = search_coverage_tickets(
                old['portfolio'], steps=config['proposals_per_run'],
                seed=config['search_seed_offset'] + old['seed'], perturbations=width,
                local_passes=config['local_refine_passes'],
                start_temperature=config['temperature_start_outcomes'],
                end_temperature=config['temperature_end_outcomes'])
            seconds = time.perf_counter() - tick
            odds = portfolio_odds(tickets)
            verify_mains(tickets, odds)
            reassigned, assignment = optimize_strong_numbers(tickets)
            if (portfolio_odds(reassigned) != odds or odds['p_any_prize'] < old['main_odds']['p_any_prize']
                    or odds['p_jackpot'] != old['main_odds']['p_jackpot']):
                raise RuntimeError('Ticket-budget, main-coverage or jackpot invariant failed')
            if strong_prize_odds(reassigned)['covered_full_outcomes'] != assignment['covered_full_outcomes_after']:
                raise RuntimeError('Strong-assignment count disagrees with grouped evaluation')
            results.append({'seed': old['seed'], 'perturbations': width, 'search_seconds': seconds,
                            'initial_portfolio': old['portfolio'], 'portfolio_after_main_search': tickets,
                            'portfolio': reassigned, 'before': old['main_odds'], 'after': odds,
                            'main_search': search, 'strong_assignment': assignment,
                            'main_reference_verified': True, 'strong_group_evaluator_verified': True})
            print(f"Moves {width}, seed {old['seed']}: any prize "
                  f"{old['main_odds']['p_any_prize']:.8%} -> {odds['p_any_prize']:.8%}; "
                  f"3+ and strong {assignment['probability_after']:.8%}; {seconds:.2f}s", flush=True)
    summaries = []
    for width in widths:
        rows = [row for row in results if row['perturbations'] == width]
        summaries.append({'perturbations': width, 'runs': len(rows),
                          'mean_any_prize_before': float(np.mean([r['before']['p_any_prize'] for r in rows])),
                          'mean_any_prize_after': float(np.mean([r['after']['p_any_prize'] for r in rows])),
                          'best_any_prize_after': max(r['after']['p_any_prize'] for r in rows),
                          'improved_runs': sum(r['after']['p_any_prize'] > r['before']['p_any_prize'] for r in rows),
                          'mean_search_seconds': float(np.mean([r['search_seconds'] for r in rows])),
                          'mean_strong_probability_after': float(np.mean(
                              [r['strong_assignment']['probability_after'] for r in rows]))})
    selected_index = max(range(len(results)), key=lambda i: (
        results[i]['after']['p_any_prize'], results[i]['strong_assignment']['covered_full_outcomes_after']))
    selected = results[selected_index]
    verify_mains(selected['portfolio'], selected['after'])
    verify_strong(selected['portfolio'], selected['strong_assignment']['covered_full_outcomes_after'])
    total = comb(37, 6)
    bounds = {str(k): {'single_ticket_main_outcomes': sum(comb(6, j) * comb(31, 6-j) for j in range(k, 7)),
                       'ten_ticket_union_bound': min(1.0, 10 * sum(comb(6, j) * comb(31, 6-j)
                                                                  for j in range(k, 7)) / total)}
              for k in (3, 4, 5, 6)}
    paths = ['bench/_portfolio_exact.py', 'bench/portfolio.py', 'bench/strong_assignment.py',
             'scripts/research_ticket_chances.py']
    output = {'schema': 1, 'status': 'complete', 'configuration': config,
              'protocol_sha256': sha256(PROTOCOL), 'source_sha256': {p: sha256(ROOT / p) for p in paths},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'platform': platform.platform()},
              'timing_scope': 'Search only, after baseline warmup; verification and strong assignment excluded',
              'baseline': baseline, 'rows': results, 'summary': summaries, 'upper_bounds': bounds,
              'selected_result_index': selected_index, 'selected_reference_verified': True,
              'selected_portfolio': selected['portfolio']}
    with tempfile.TemporaryDirectory(prefix='ticket-chances-', dir=OUTPUT.parent) as staging:
        staged = Path(staging) / OUTPUT.name
        staged.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        staged.replace(OUTPUT)
    print(f'Wrote {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
