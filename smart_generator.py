"""
Smart Ticket Generator for Israeli Lotto.

Generates tickets that avoid popular combinations to maximize payout if you win.

Strategy:
1. Avoid birthday numbers bias (favor 32-37)
2. Avoid arithmetic sequences (1,2,3,4,5,6)
3. Avoid common patterns (diagonal lines on ticket)
4. Ensure good number spread
5. Score tickets by "unpopularity" and pick the best
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
from itertools import combinations

from logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

# Israeli Lotto configuration
MAIN_NUMBERS = 37  # 1-37 for main balls
BONUS_NUMBERS = 7  # 1-7 for bonus ball
BALLS_PER_DRAW = 6  # 6 main balls

# Birthday numbers that people commonly pick
BIRTHDAY_NUMBERS = set(range(1, 32))  # 1-31 (days of month)
HIGH_NUMBERS = set(range(32, 38))  # 32-37 (less popular)

# Common sequences to avoid
COMMON_SEQUENCES = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [1, 7, 14, 21, 28, 35],  # Multiples of 7
    [5, 10, 15, 20, 25, 30],  # Multiples of 5
    [6, 12, 18, 24, 30, 36],  # Multiples of 6
    [1, 11, 21, 31, 5, 15],  # Numbers ending in 1
]


class TicketScorer:
    """
    Scores lottery tickets based on how "unpopular" they are.
    Higher score = less likely to be picked by others = higher expected payout if win.
    """

    def __init__(self, historical_data: Optional[np.ndarray] = None):
        """
        Args:
            historical_data: Optional historical lottery data for analyzing patterns
        """
        self.historical_data = historical_data
        self.pair_frequencies = None
        
        if historical_data is not None:
            self._analyze_historical_patterns()

    def _analyze_historical_patterns(self):
        """Analyze historical data for common patterns."""
        self.pair_frequencies = Counter()
        
        if self.historical_data is None:
            return
            
        for draw in self.historical_data:
            main_balls = sorted(draw[:6])
            for pair in combinations(main_balls, 2):
                self.pair_frequencies[pair] += 1

    def score_high_numbers(self, ticket: List[int]) -> float:
        """
        Score based on inclusion of high numbers (32-37).
        People tend to pick birthday numbers (1-31), so high numbers are less popular.
        
        Returns score 0-1, higher = more high numbers = better.
        """
        high_count = sum(1 for n in ticket[:6] if n > 31)
        # Ideal: at least 2 high numbers
        return min(high_count / 2, 1.0)

    def score_spread(self, ticket: List[int]) -> float:
        """
        Score based on number spread across the range.
        People tend to cluster numbers; well-spread tickets are less common.
        
        Returns score 0-1, higher = better spread.
        """
        main = sorted(ticket[:6])
        
        # Calculate gaps between consecutive numbers
        gaps = [main[i+1] - main[i] for i in range(5)]
        
        # Ideal spread: gaps should be roughly equal (~5.3 each for 37 numbers, 6 balls)
        # Score based on how even the gaps are
        gap_variance = np.var(gaps)
        max_variance = 30**2  # Worst case: all numbers clustered
        
        spread_score = 1 - min(gap_variance / max_variance, 1.0)
        
        # Also penalize if all numbers are in bottom half or top half
        range_coverage = (max(main) - min(main)) / 36
        
        return float((spread_score + range_coverage) / 2)

    def score_sequence_avoidance(self, ticket: List[int]) -> float:
        """
        Score based on avoiding arithmetic sequences.
        Sequences are very popular picks.
        
        Returns score 0-1, higher = less sequential.
        """
        main = sorted(ticket[:6])
        
        # Check for arithmetic progressions
        for i in range(4):  # Check subsequences of length 3+
            for length in range(3, 7 - i):
                subseq = main[i:i+length]
                diffs = [subseq[j+1] - subseq[j] for j in range(len(subseq)-1)]
                if len(set(diffs)) == 1:  # All same difference = arithmetic sequence
                    # Penalize based on length
                    return 1 - (length - 2) * 0.25  # 3-seq: 0.75, 4-seq: 0.5, etc.
        
        return 1.0

    def score_pattern_avoidance(self, ticket: List[int]) -> float:
        """
        Score based on avoiding common visual patterns on ticket slip.
        (Diagonal lines, horizontal lines, etc.)
        
        Israeli Lotto ticket layout is typically 6x7 or similar grid.
        """
        main = sorted(ticket[:6])
        
        # Check against known common patterns
        for pattern in COMMON_SEQUENCES:
            matches = len(set(main) & set(pattern))
            if matches >= 4:
                return 0.5
            if matches >= 5:
                return 0.25
            if matches == 6:
                return 0.0
        
        return 1.0

    def score_pair_rarity(self, ticket: List[int]) -> float:
        """
        Score based on rarity of number pairs (from historical data).
        Less common pairs = higher score.
        """
        if self.pair_frequencies is None:
            return 0.5  # Neutral if no historical data
        
        main = sorted(ticket[:6])
        total_freq = 0
        
        for pair in combinations(main, 2):
            total_freq += self.pair_frequencies.get(tuple(pair), 0)
        
        # Normalize: 15 pairs per ticket
        avg_freq = total_freq / 15
        max_freq = max(self.pair_frequencies.values()) if self.pair_frequencies else 1
        
        # Lower frequency = higher score
        return 1 - min(avg_freq / max_freq, 1.0)

    def score_ticket(self, ticket: List[int]) -> Dict:
        """
        Calculate comprehensive unpopularity score for a ticket.
        
        Args:
            ticket: List of 7 numbers (6 main + 1 bonus)
            
        Returns:
            Dictionary with individual scores and total
        """
        scores = {
            "high_numbers": self.score_high_numbers(ticket),
            "spread": self.score_spread(ticket),
            "sequence_avoidance": self.score_sequence_avoidance(ticket),
            "pattern_avoidance": self.score_pattern_avoidance(ticket),
            "pair_rarity": self.score_pair_rarity(ticket),
        }
        
        # Weighted total (can adjust weights based on importance)
        weights = {
            "high_numbers": 0.25,
            "spread": 0.15,
            "sequence_avoidance": 0.25,
            "pattern_avoidance": 0.15,
            "pair_rarity": 0.20,
        }
        
        scores["total"] = sum(scores[k] * weights[k] for k in weights)
        
        return scores


class SmartTicketGenerator:
    """
    Generates lottery tickets optimized to avoid popular combinations.
    """

    def __init__(
        self, 
        historical_data: Optional[np.ndarray] = None,
        n_main: int = MAIN_NUMBERS,
        n_bonus: int = BONUS_NUMBERS
    ):
        self.n_main = n_main
        self.n_bonus = n_bonus
        self.scorer = TicketScorer(historical_data)

    def generate_random_ticket(self) -> List[int]:
        """Generate a single random ticket."""
        main = list(np.random.choice(self.n_main, size=6, replace=False) + 1)  # 1-indexed
        bonus = np.random.randint(1, self.n_bonus + 1)
        return sorted(main) + [bonus]

    def generate_smart_ticket(self, min_high_numbers: int = 2) -> List[int]:
        """
        Generate a ticket with smart constraints.
        
        Args:
            min_high_numbers: Minimum numbers > 31 to include
        """
        # Ensure at least min_high_numbers from 32-37
        high = list(np.random.choice(list(range(32, 38)), 
                                     size=min(min_high_numbers, 6), 
                                     replace=False))
        
        # Fill remaining from all numbers (excluding already picked)
        remaining_count = 6 - len(high)
        available = [n for n in range(1, 38) if n not in high]
        low = list(np.random.choice(available, size=remaining_count, replace=False))
        
        main = sorted(high + low)
        bonus = np.random.randint(1, self.n_bonus + 1)
        
        return main + [bonus]

    def generate_optimized_tickets(
        self, 
        n_tickets: int = 5, 
        n_candidates: int = 1000,
        min_high_numbers: int = 2
    ) -> List[Tuple[List[int], Dict]]:
        """
        Generate the best tickets by scoring many candidates.
        
        Args:
            n_tickets: Number of tickets to return
            n_candidates: Number of random tickets to generate and score
            min_high_numbers: Minimum high numbers per ticket
            
        Returns:
            List of (ticket, scores) tuples, sorted by score descending
        """
        candidates = []
        
        for _ in range(n_candidates):
            ticket = self.generate_smart_ticket(min_high_numbers)
            scores = self.scorer.score_ticket(ticket)
            candidates.append((ticket, scores))
        
        # Sort by total score descending
        candidates.sort(key=lambda x: x[1]["total"], reverse=True)
        
        # Return top n_tickets, ensuring diversity
        selected = []
        seen_combinations = set()
        
        for ticket, scores in candidates:
            ticket_tuple = tuple(ticket[:6])  # Just main numbers for uniqueness
            if ticket_tuple not in seen_combinations:
                selected.append((ticket, scores))
                seen_combinations.add(ticket_tuple)
                if len(selected) >= n_tickets:
                    break
        
        return selected

    def generate_coverage_tickets(
        self,
        n_tickets: int = 10,
        overlap_threshold: int = 3
    ) -> List[Tuple[List[int], Dict]]:
        """
        Generate tickets that maximize coverage (minimize overlap between tickets).
        
        This is useful if you're buying multiple tickets - you want diversity.
        """
        tickets = []
        all_used_numbers = Counter()
        
        for _ in range(n_tickets * 10):  # Generate many candidates
            ticket = self.generate_smart_ticket(min_high_numbers=2)
            
            # Check overlap with existing tickets
            max_overlap = 0
            for existing, _ in tickets:
                overlap = len(set(ticket[:6]) & set(existing[:6]))
                max_overlap = max(max_overlap, overlap)
            
            if max_overlap <= overlap_threshold or len(tickets) == 0:
                scores = self.scorer.score_ticket(ticket)
                
                # Bonus: prefer numbers not yet used much
                diversity_bonus = 1 - sum(all_used_numbers[n] for n in ticket[:6]) / max(1, sum(all_used_numbers.values()))
                scores["diversity_bonus"] = diversity_bonus
                scores["total"] = scores["total"] * 0.7 + diversity_bonus * 0.3
                
                tickets.append((ticket, scores))
                for n in ticket[:6]:
                    all_used_numbers[n] += 1
                
                if len(tickets) >= n_tickets:
                    break
        
        return tickets


def print_tickets(tickets: List[Tuple[List[int], Dict]], title: str = "Smart Tickets"):
    """Print generated tickets in a nice format."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    
    for i, (ticket, scores) in enumerate(tickets, 1):
        main = ticket[:6]
        bonus = ticket[6]
        
        # Format main numbers with padding
        main_str = " ".join(f"{n:2d}" for n in main)
        
        print(f"\n  Ticket #{i}:")
        print(f"    Numbers:  [{main_str}]  Bonus: {bonus}")
        print(f"    Score:    {scores['total']:.3f}")
        print(f"    Details:  high={scores['high_numbers']:.2f} "
              f"spread={scores['spread']:.2f} "
              f"seq={scores['sequence_avoidance']:.2f} "
              f"pattern={scores['pattern_avoidance']:.2f}")
    
    print(f"\n{'=' * 60}")


def analyze_ticket(ticket: List[int], scorer: TicketScorer) -> None:
    """Analyze and print detailed information about a ticket."""
    scores = scorer.score_ticket(ticket)
    main = ticket[:6]
    bonus = ticket[6]
    
    print("\n--- Ticket Analysis ---")
    print(f"Numbers: {main} + Bonus: {bonus}")
    print("\nScoring breakdown:")
    print(f"  High numbers (32-37):   {sum(1 for n in main if n > 31)} included -> Score: {scores['high_numbers']:.2f}")
    print(f"  Number spread:          Score: {scores['spread']:.2f}")
    print(f"  Sequence avoidance:     Score: {scores['sequence_avoidance']:.2f}")
    print(f"  Pattern avoidance:      Score: {scores['pattern_avoidance']:.2f}")
    print(f"  Pair rarity:            Score: {scores['pair_rarity']:.2f}")
    print(f"\n  TOTAL UNPOPULARITY:     {scores['total']:.3f}")
    
    if scores['total'] > 0.7:
        print("  Assessment: Excellent - very unpopular combination!")
    elif scores['total'] > 0.5:
        print("  Assessment: Good - reasonably uncommon")
    else:
        print("  Assessment: Average - many people might pick similar numbers")


if __name__ == "__main__":
    from helpers import fetch_dataset
    
    # Load historical data
    print("Loading historical lottery data...")
    lotto_ds = fetch_dataset()
    historical_data = lotto_ds.values  # Keep 1-indexed for analysis
    
    # Create generator with historical data
    generator = SmartTicketGenerator(historical_data - 1)  # Convert to 0-indexed for internal use
    
    # Generate optimized tickets
    print("\nGenerating optimized tickets...")
    smart_tickets = generator.generate_optimized_tickets(n_tickets=5, n_candidates=5000)
    print_tickets(smart_tickets, "Top 5 Smart Tickets (Unpopular Combinations)")
    
    # Generate coverage tickets (good if buying multiple)
    print("\nGenerating diverse coverage tickets...")
    coverage_tickets = generator.generate_coverage_tickets(n_tickets=5)
    print_tickets(coverage_tickets, "5 Diverse Coverage Tickets")
    
    # Analyze a sample popular ticket for comparison
    print("\n" + "=" * 60)
    print("  Comparison: Popular vs Unpopular Tickets")
    print("=" * 60)
    
    scorer = TicketScorer(historical_data - 1)
    
    print("\nA 'popular' ticket (birthdays + sequence):")
    popular_ticket = [1, 2, 3, 7, 14, 21, 3]  # Sequential + all low numbers
    analyze_ticket(popular_ticket, scorer)
    
    print("\nOur best smart ticket:")
    best_ticket = smart_tickets[0][0]
    analyze_ticket(best_ticket, scorer)
