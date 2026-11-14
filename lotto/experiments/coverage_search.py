#!/usr/bin/env python3
"""Run the bounded discrete basin-hopping follow-up on five saved portfolios."""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from lotto.tickets.coverage import best_match_counts, portfolio_odds, search_coverage_tickets

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT/'experiments/coverage_search_protocol_2026_11_06.json'
    config = json.loads(path.read_text())
    previous_path = ROOT/'experiments'/config['initial_results']
    previous = json.loads(previous_path.read_text())
    if (not config['seeds'] or any(type(seed) is not int or seed < 0 for seed in config['seeds'])
            or len(set(config['seeds'])) != len(config['seeds'])
            or not set(config['seeds']).issubset({row['seed'] for row in previous['rows']})):
        raise ValueError('Require distinct nonnegative seeds present in the initial results')
    rows = []
    for seed in config['seeds']:
        old = next(r for r in previous['rows'] if r['seed'] == seed)
        tick = time.perf_counter()
        tickets, audit = search_coverage_tickets(
            old['refined_portfolio'], steps=config['proposals_per_portfolio'],
            seed=config['search_seed_offset']+seed,
            start_temperature=config['temperature_start_outcomes'],
            end_temperature=config['temperature_end_outcomes'],
            perturbations=config['perturbations_per_proposal'], local_passes=config['local_refine_passes'])
        odds = portfolio_odds(tickets)
        reference = best_match_counts([r[:6] for r in tickets], engine='reference')
        np.testing.assert_array_equal(reference, odds['best_match_counts'])
        if (odds['p_any_prize'] < old['after']['p_any_prize']
                or odds['p_jackpot'] != old['after']['p_jackpot']):
            raise RuntimeError('Coverage or jackpot invariant failed')
        rows.append({'seed': seed, 'initial_portfolio': old['refined_portfolio'],
                     'portfolio': tickets, 'before': old['after'], 'after': odds, 'audit': audit,
                     'reference_verified': True, 'seconds': time.perf_counter()-tick})
        print(f"Seed {seed}: {old['after']['p_any_prize']:.8%} -> {odds['p_any_prize']:.8%}", flush=True)
    result = {'schema': 1, 'configuration': config,
              'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'initial_results_sha256': hashlib.sha256(previous_path.read_bytes()).hexdigest(), 'rows': rows}
    # New runs go to results/ (not tracked); experiments/ holds the frozen evidence.
    out = ROOT/'results/coverage-search/coverage_search.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='coverage-research-', dir=out.parent) as staging:
        staged = Path(staging)/out.name
        staged.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        staged.replace(out)
    print(f'Wrote {out}', flush=True)


if __name__ == '__main__':
    main()
