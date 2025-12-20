"""
Comprehensive evaluation script for ILotto.

Evaluates model predictions with proper metrics, compares against baseline,
runs statistical tests, and generates smart ticket recommendations.
"""

import logging
import numpy as np
from mdutils.mdutils import MdUtils

from logger import setup_logger
from train import get_compiled_model
from helpers import beam_search_decoder, fetch_dataset, train_test_split
from metrics import LotteryMetrics, RandomnessTests, print_metrics_report
from smart_generator import SmartTicketGenerator, print_tickets

setup_logger()
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    beam_width: int = 10,
) -> dict:
    """
    Evaluate model and return predictions with metrics.
    """
    logger.info("Loading model from %s", model_path)

    model = get_compiled_model()
    model.load_weights(model_path)
    logger.info("Model loaded successfully")

    # Get predictions
    pred_probs = model.predict(X_test)
    logger.info("Prediction completed")
    pred = np.argmax(pred_probs, axis=2)

    # Get latest prediction (for next draw)
    X_latest = X_test[0][1:]
    X_latest = np.concatenate([X_latest, y_test[0].reshape(1, 7)], axis=0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
    pred_latest_probs = model.predict(X_latest)
    pred_latest_probs = np.squeeze(pred_latest_probs)
    pred_latest_greedy = np.argmax(pred_latest_probs, axis=1)

    # Beam search results
    beam_replace = beam_search_decoder(pred_latest_probs, beam_width, replace=True)
    beam_no_replace = beam_search_decoder(pred_latest_probs, beam_width, replace=False)

    return {
        "predictions": pred,
        "pred_probs": pred_probs,
        "latest_prediction": pred_latest_greedy,
        "latest_probs": pred_latest_probs,
        "beam_replace": beam_replace,
        "beam_no_replace": beam_no_replace,
    }


def generate_readme(
    model_results: dict,
    y_test: np.ndarray,
    metrics_comparison: dict,
    randomness_results: dict,
    smart_tickets: list,
    beam_width: int = 10,
):
    """Generate comprehensive README with all results."""
    
    mdFile = MdUtils(file_name="README", title="ILotto - Israeli Lottery Analysis")
    
    # Introduction
    mdFile.new_paragraph(
        "A machine learning project for analyzing Israeli Lotto patterns. "
        "Includes proper statistical analysis, model evaluation metrics, and "
        "smart ticket generation to avoid popular combinations."
    )
    
    # Model Prediction Section
    mdFile.new_header(level=1, title="Model Prediction")
    mdFile.new_paragraph("Latest prediction from the neural network model:")
    pred_str = str(model_results["latest_prediction"] + 1)
    mdFile.new_line(pred_str, bold_italics_code="bic")
    
    # Beam Search Results
    mdFile.new_header(level=2, title="Beam Search Results")
    
    mdFile.new_line(f"Beam Width: {beam_width}, Allow Repeats: True")
    mdFile.new_list(
        items=[
            f"Prediction: {np.array(seq[0]) + 1}\tLog Likelihood: {seq[1]:.4f}"
            for seq in model_results["beam_replace"]
        ]
    )
    
    mdFile.new_line(f"Beam Width: {beam_width}, Allow Repeats: False")
    mdFile.new_list(
        items=[
            f"Prediction: {np.array(seq[0]) + 1}\tLog Likelihood: {seq[1]:.4f}"
            for seq in model_results["beam_no_replace"]
        ]
    )
    
    # Metrics Section
    mdFile.new_header(level=1, title="Model Evaluation Metrics")
    
    model_metrics = metrics_comparison["model"]
    mdFile.new_header(level=2, title="Performance Summary")
    
    summary_table = [
        "Metric", "Value",
        "Samples Evaluated", str(model_metrics["n_samples"]),
        "Average Matches", f"{model_metrics['avg_matches']:.3f}",
        "Expected (Random)", f"{model_metrics['expected_random']:.3f}",
        "Improvement Over Random", f"{metrics_comparison['improvement_percent']:+.2f}%",
        "Statistically Significant", "Yes" if metrics_comparison["significantly_better"] else "No",
        "P-Value", f"{metrics_comparison['p_value']:.4f}",
    ]
    mdFile.new_table(columns=2, rows=7, text=summary_table, text_align="center")
    
    # Match Distribution
    mdFile.new_header(level=2, title="Match Distribution")
    dist_table = ["Matches", "Count", "Percentage"]
    for i in range(7):
        count = model_metrics["match_distribution"][i]
        pct = model_metrics["match_percentages"][i]
        dist_table.extend([str(i), str(count), f"{pct:.1f}%"])
    mdFile.new_table(columns=3, rows=8, text=dist_table, text_align="center")
    
    # Win Rates
    mdFile.new_header(level=2, title="Win Rates by Tier")
    win_table = ["Tier", "Rate"]
    for tier, rate in model_metrics["win_rates"].items():
        win_table.extend([f"{tier} matches", f"{rate:.2f}%"])
    mdFile.new_table(columns=2, rows=5, text=win_table, text_align="center")
    
    # Randomness Analysis
    mdFile.new_header(level=1, title="Lottery Randomness Analysis")
    mdFile.new_paragraph(
        "Statistical tests to verify if the Israeli Lotto is truly random. "
        "A p-value > 0.05 indicates we cannot reject the null hypothesis (randomness)."
    )
    
    randomness_table = ["Test", "P-Value", "Result"]
    for test_name, result in randomness_results.items():
        randomness_table.extend([
            result["test"],
            f"{result['p_value']:.4f}",
            result["interpretation"]
        ])
    mdFile.new_table(columns=3, rows=5, text=randomness_table, text_align="center")
    
    # Smart Tickets Section
    mdFile.new_header(level=1, title="Smart Ticket Recommendations")
    mdFile.new_paragraph(
        "These tickets are optimized to avoid popular combinations. "
        "If you win, you're less likely to share the jackpot with others. "
        "Strategy includes: favoring high numbers (32-37), avoiding sequences, "
        "and ensuring good number spread."
    )
    
    tickets_table = ["#", "Numbers", "Bonus", "Score"]
    for i, (ticket, scores) in enumerate(smart_tickets, 1):
        main = " ".join(f"{n:2d}" for n in ticket[:6])
        tickets_table.extend([
            str(i),
            f"[{main}]",
            str(ticket[6]),
            f"{scores['total']:.3f}"
        ])
    mdFile.new_table(columns=4, rows=len(smart_tickets) + 1, text=tickets_table, text_align="center")
    
    # Test Set Validation
    mdFile.new_header(level=1, title="Test Set Validation")
    mdFile.new_paragraph("Model predictions vs actual lottery results:")
    
    pred = model_results["predictions"]
    list_of_strings = ["Prediction", "Ground Truth"]
    for p, y in zip(pred, y_test):
        list_of_strings.extend([str(p), str(y)])
    
    mdFile.new_table(
        columns=2, rows=len(pred) + 1, text=list_of_strings, text_align="center"
    )
    
    # Disclaimer
    mdFile.new_header(level=1, title="Disclaimer")
    mdFile.new_paragraph(
        "**Important:** Lottery numbers are designed to be random. "
        "No prediction method can reliably predict future draws. "
        "This project is for educational purposes and to demonstrate ML concepts. "
        "The 'smart tickets' don't increase your probability of winning - "
        "they only optimize for higher expected payout IF you win by avoiding popular combinations. "
        "Please gamble responsibly."
    )
    
    mdFile.create_md_file()
    logger.info("README.md created successfully")


def run_full_evaluation(
    model_path: str = "model/Ilotto.keras",
    beam_width: int = 10,
    n_smart_tickets: int = 5,
):
    """Run complete evaluation pipeline."""
    
    print("\n" + "=" * 70)
    print("                    ILOTTO FULL EVALUATION")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading lottery data...")
    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    all_data = lotto_ds.values - 1  # 0-indexed
    print(f"      Loaded {len(lotto_ds)} historical draws")
    print(f"      Test set: {len(y_test)} samples")
    
    # Evaluate model
    print("\n[2/6] Evaluating model...")
    model_results = evaluate_model(model_path, X_test, y_test, beam_width)
    print(f"      Latest prediction: {model_results['latest_prediction'] + 1}")
    
    # Calculate metrics
    print("\n[3/6] Computing metrics...")
    metrics = LotteryMetrics()
    comparison = metrics.compare_with_baseline(
        model_results["predictions"], y_test, n_baseline_runs=100
    )
    print(f"      Model avg matches: {comparison['model']['avg_matches']:.3f}")
    print(f"      Baseline avg:      {comparison['baseline_avg_matches']:.3f}")
    print(f"      Improvement:       {comparison['improvement_percent']:+.2f}%")
    
    # Randomness tests
    print("\n[4/6] Running randomness tests...")
    randomness = RandomnessTests(all_data)
    randomness_results = randomness.run_all_tests()
    for test_name, result in randomness_results.items():
        status = "PASS" if result["is_random"] else "BIAS"
        print(f"      {result['test']}: [{status}] p={result['p_value']:.4f}")
    
    # Generate smart tickets
    print("\n[5/6] Generating smart tickets...")
    generator = SmartTicketGenerator(all_data)
    smart_tickets = generator.generate_optimized_tickets(
        n_tickets=n_smart_tickets, n_candidates=5000
    )
    print_tickets(smart_tickets, f"Top {n_smart_tickets} Smart Tickets")
    
    # Generate README
    print("\n[6/6] Generating README.md...")
    generate_readme(
        model_results=model_results,
        y_test=y_test,
        metrics_comparison=comparison,
        randomness_results=randomness_results,
        smart_tickets=smart_tickets,
        beam_width=beam_width,
    )
    
    # Print full metrics report
    print_metrics_report(comparison, randomness_results)
    
    print("\n" + "=" * 70)
    print("                    EVALUATION COMPLETE")
    print("=" * 70)
    print("\nFiles generated:")
    print("  - README.md (comprehensive report)")
    print("\nNext steps:")
    print("  - Review the metrics to understand model performance")
    print("  - Use smart tickets if you decide to play")
    print("  - Remember: lottery is random, play responsibly!")
    print()


if __name__ == "__main__":
    run_full_evaluation()
