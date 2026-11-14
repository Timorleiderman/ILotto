"""
Metrics module for ILotto evaluation.

Provides proper evaluation metrics including:
- Partial match counting (0-6 matches)
- Random baseline comparison
- Statistical analysis
- Win rate at different tiers
"""

import logging
import numpy as np
from collections import Counter
from typing import Tuple, Dict
from scipy import stats

from logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


# Israeli Lotto configuration
MAIN_NUMBERS = 37  # 1-37 for main balls
BONUS_NUMBERS = 7  # 1-7 for bonus ball
BALLS_PER_DRAW = 6  # 6 main balls
TOTAL_BALLS = 7  # 6 main + 1 bonus


class LotteryMetrics:
    """
    Comprehensive metrics for lottery prediction evaluation.
    """

    def __init__(self, n_main: int = MAIN_NUMBERS, n_bonus: int = BONUS_NUMBERS):
        self.n_main = n_main
        self.n_bonus = n_bonus

    def count_matches(
        self, prediction: np.ndarray, ground_truth: np.ndarray, include_bonus: bool = True
    ) -> Tuple[int, bool]:
        """
        Count how many numbers match between prediction and ground truth.

        Args:
            prediction: Predicted numbers (7 values: 6 main + 1 bonus)
            ground_truth: Actual drawn numbers (7 values: 6 main + 1 bonus)
            include_bonus: Whether to separately track bonus ball match

        Returns:
            Tuple of (main_matches, bonus_matched)
        """
        pred_main = set(prediction[:6])
        true_main = set(ground_truth[:6])
        main_matches = len(pred_main & true_main)

        bonus_matched = prediction[6] == ground_truth[6] if include_bonus else False

        return main_matches, bonus_matched

    def evaluate_predictions(
        self, predictions: np.ndarray, ground_truths: np.ndarray
    ) -> Dict:
        """
        Evaluate a batch of predictions against ground truths.

        Args:
            predictions: Array of shape (n_samples, 7)
            ground_truths: Array of shape (n_samples, 7)

        Returns:
            Dictionary with comprehensive metrics
        """
        n_samples = len(predictions)
        match_counts = Counter()
        bonus_matches = 0
        total_main_matches = 0

        for pred, truth in zip(predictions, ground_truths):
            main_match, bonus_match = self.count_matches(pred, truth)
            match_counts[main_match] += 1
            total_main_matches += main_match
            if bonus_match:
                bonus_matches += 1

        # Calculate statistics
        avg_matches = total_main_matches / n_samples
        
        # Expected matches for random prediction
        # P(match) = 6/37 for each of 6 predicted numbers ≈ 0.973 expected matches
        expected_random = 6 * (6 / self.n_main)

        # Match distribution
        match_distribution = {i: match_counts.get(i, 0) for i in range(7)}
        match_percentages = {i: match_counts.get(i, 0) / n_samples * 100 for i in range(7)}

        # Win rates at different tiers
        win_rates = {
            "3+": sum(match_counts.get(i, 0) for i in range(3, 7)) / n_samples * 100,
            "4+": sum(match_counts.get(i, 0) for i in range(4, 7)) / n_samples * 100,
            "5+": sum(match_counts.get(i, 0) for i in range(5, 7)) / n_samples * 100,
            "6": match_counts.get(6, 0) / n_samples * 100,
        }

        return {
            "n_samples": n_samples,
            "avg_matches": avg_matches,
            "expected_random": expected_random,
            "improvement_over_random": avg_matches - expected_random,
            "match_distribution": match_distribution,
            "match_percentages": match_percentages,
            "bonus_match_rate": bonus_matches / n_samples * 100,
            "win_rates": win_rates,
            "total_main_matches": total_main_matches,
            "bonus_matches": bonus_matches,
        }

    def generate_random_predictions(
        self, n_samples: int, seed: int = None
    ) -> np.ndarray:
        """
        Generate random lottery predictions as a baseline.

        Args:
            n_samples: Number of predictions to generate
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, 7) with random predictions
        """
        if seed is not None:
            np.random.seed(seed)

        predictions = np.zeros((n_samples, 7), dtype=np.int32)

        for i in range(n_samples):
            # Main numbers: 6 unique from 0-36 (0-indexed)
            predictions[i, :6] = np.random.choice(self.n_main, size=6, replace=False)
            # Bonus number: 1 from 0-6 (0-indexed)
            predictions[i, 6] = np.random.randint(0, self.n_bonus)

        return predictions

    def compare_with_baseline(
        self,
        model_predictions: np.ndarray,
        ground_truths: np.ndarray,
        n_baseline_runs: int = 100,
        seed: int = 42,
    ) -> Dict:
        """
        Compare model predictions with random baseline over multiple runs.

        Args:
            model_predictions: Model's predictions
            ground_truths: Actual drawn numbers
            n_baseline_runs: Number of random baseline runs to average
            seed: Random seed

        Returns:
            Comparison dictionary
        """
        np.random.seed(seed)
        n_samples = len(ground_truths)

        # Evaluate model
        model_metrics = self.evaluate_predictions(model_predictions, ground_truths)

        # Run multiple random baselines
        baseline_matches = []
        baseline_3plus = []
        
        for run in range(n_baseline_runs):
            random_preds = self.generate_random_predictions(n_samples, seed=seed + run)
            baseline_metrics = self.evaluate_predictions(random_preds, ground_truths)
            baseline_matches.append(baseline_metrics["avg_matches"])
            baseline_3plus.append(baseline_metrics["win_rates"]["3+"])

        baseline_avg = np.mean(baseline_matches)
        baseline_std = np.std(baseline_matches)
        baseline_3plus_avg = np.mean(baseline_3plus)

        # Statistical test: is model significantly better than random?
        # Using z-test
        z_score = (model_metrics["avg_matches"] - baseline_avg) / (baseline_std / np.sqrt(n_baseline_runs))
        p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test

        return {
            "model": model_metrics,
            "baseline_avg_matches": baseline_avg,
            "baseline_std": baseline_std,
            "baseline_3plus_rate": baseline_3plus_avg,
            "z_score": z_score,
            "p_value": p_value,
            "significantly_better": p_value < 0.05,
            "improvement_percent": (model_metrics["avg_matches"] - baseline_avg) / baseline_avg * 100,
        }


class RandomnessTests:
    """
    Statistical tests to verify lottery randomness.
    """

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Historical lottery data, shape (n_draws, 7)
        """
        self.data = data
        self.n_draws = len(data)

    def chi_square_test_main_balls(self) -> Dict:
        """
        Chi-square test for uniformity of main ball distribution.

        Tests if each number (1-37) appears with equal probability.
        """
        # Flatten all main balls
        main_balls = self.data[:, :6].flatten()
        
        # Count frequencies
        observed = np.zeros(MAIN_NUMBERS)
        for ball in main_balls:
            observed[int(ball)] += 1

        # Expected frequency (uniform distribution)
        total_balls = len(main_balls)
        expected = np.full(MAIN_NUMBERS, total_balls / MAIN_NUMBERS)

        # Chi-square test
        chi2, p_value = stats.chisquare(observed, expected)

        return {
            "test": "Chi-square uniformity (main balls)",
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": MAIN_NUMBERS - 1,
            "is_random": p_value > 0.05,  # Fail to reject null = random
            "interpretation": "Random" if p_value > 0.05 else "Non-random bias detected",
            "observed_frequencies": observed.tolist(),
            "expected_frequency": expected[0],
        }

    def chi_square_test_bonus_ball(self) -> Dict:
        """
        Chi-square test for uniformity of bonus ball distribution.
        """
        bonus_balls = self.data[:, 6]
        
        observed = np.zeros(BONUS_NUMBERS)
        for ball in bonus_balls:
            observed[int(ball)] += 1

        expected = np.full(BONUS_NUMBERS, len(bonus_balls) / BONUS_NUMBERS)

        chi2, p_value = stats.chisquare(observed, expected)

        return {
            "test": "Chi-square uniformity (bonus ball)",
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": BONUS_NUMBERS - 1,
            "is_random": p_value > 0.05,
            "interpretation": "Random" if p_value > 0.05 else "Non-random bias detected",
            "observed_frequencies": observed.tolist(),
            "expected_frequency": expected[0],
        }

    def runs_test(self) -> Dict:
        """
        Runs test for independence of consecutive draws.
        
        Tests if there's any pattern in the sequence of draws.
        """
        # Use first ball of each draw for simplicity
        first_balls = self.data[:, 0]
        median = np.median(first_balls)
        
        # Convert to binary: above/below median
        binary = (first_balls > median).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1

        # Expected runs and variance under null hypothesis
        n1 = np.sum(binary)
        n0 = len(binary) - n1
        n = n0 + n1
        
        expected_runs = (2 * n0 * n1) / n + 1
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
        
        # Z-score
        z = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed

        return {
            "test": "Runs test (independence)",
            "n_runs": runs,
            "expected_runs": expected_runs,
            "z_score": z,
            "p_value": p_value,
            "is_random": p_value > 0.05,
            "interpretation": "Independent draws" if p_value > 0.05 else "Pattern detected",
        }

    def serial_correlation_test(self, lag: int = 1) -> Dict:
        """
        Test for correlation between consecutive draws.
        """
        first_balls = self.data[:, 0].astype(float)
        
        # Compute autocorrelation at given lag
        n = len(first_balls)
        mean = np.mean(first_balls)
        var = np.var(first_balls)
        
        autocorr = np.sum((first_balls[:-lag] - mean) * (first_balls[lag:] - mean)) / ((n - lag) * var)
        
        # Standard error under null hypothesis
        se = 1 / np.sqrt(n)
        z = autocorr / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "test": f"Serial correlation (lag={lag})",
            "autocorrelation": autocorr,
            "z_score": z,
            "p_value": p_value,
            "is_random": p_value > 0.05,
            "interpretation": "No correlation" if p_value > 0.05 else "Serial correlation detected",
        }

    def run_all_tests(self) -> Dict:
        """Run all randomness tests and return results."""
        return {
            "chi_square_main": self.chi_square_test_main_balls(),
            "chi_square_bonus": self.chi_square_test_bonus_ball(),
            "runs_test": self.runs_test(),
            "serial_correlation": self.serial_correlation_test(),
        }


def print_metrics_report(comparison: Dict, randomness_tests: Dict = None):
    """
    Print a formatted report of metrics and comparisons.
    """
    model = comparison["model"]
    
    print("\n" + "=" * 60)
    print("               LOTTERY PREDICTION METRICS REPORT")
    print("=" * 60)
    
    print("\n--- Model Performance ---")
    print(f"  Samples evaluated:     {model['n_samples']}")
    print(f"  Average matches:       {model['avg_matches']:.3f}")
    print(f"  Expected (random):     {model['expected_random']:.3f}")
    print(f"  Bonus match rate:      {model['bonus_match_rate']:.1f}%")
    
    print("\n--- Match Distribution ---")
    for i in range(7):
        count = model['match_distribution'][i]
        pct = model['match_percentages'][i]
        bar = "█" * int(pct / 2)
        print(f"  {i} matches: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n--- Win Rates ---")
    for tier, rate in model['win_rates'].items():
        print(f"  {tier} matches: {rate:.2f}%")
    
    print("\n--- Comparison with Random Baseline ---")
    print(f"  Baseline avg matches:  {comparison['baseline_avg_matches']:.3f} ± {comparison['baseline_std']:.3f}")
    print(f"  Baseline 3+ rate:      {comparison['baseline_3plus_rate']:.2f}%")
    print(f"  Z-score:               {comparison['z_score']:.3f}")
    print(f"  P-value:               {comparison['p_value']:.4f}")
    print(f"  Significantly better:  {'Yes' if comparison['significantly_better'] else 'No'}")
    print(f"  Improvement:           {comparison['improvement_percent']:+.2f}%")
    
    if randomness_tests:
        print("\n--- Randomness Tests ---")
        for test_name, result in randomness_tests.items():
            print(f"\n  {result['test']}:")
            print(f"    P-value: {result['p_value']:.4f}")
            print(f"    Result:  {result['interpretation']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Quick test
    from helpers import fetch_dataset, train_test_split
    
    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    
    metrics = LotteryMetrics()
    
    # Generate random predictions as mock model output
    random_preds = metrics.generate_random_predictions(len(y_test), seed=42)
    
    # Compare
    comparison = metrics.compare_with_baseline(random_preds, y_test)
    
    # Randomness tests
    all_data = lotto_ds.values - 1  # Convert to 0-indexed
    randomness = RandomnessTests(all_data)
    randomness_results = randomness.run_all_tests()
    
    print_metrics_report(comparison, randomness_results)
