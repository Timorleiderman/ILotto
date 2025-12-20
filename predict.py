#!/usr/bin/env python3
"""
Prediction script for ILotto.

Generates predictions from all available models and outputs them to PREDICTION.md.
Combines model predictions with smart ticket scoring.

Usage:
    python predict.py                    # All models
    python predict.py --models multi_output transformer
    python predict.py --count 10         # 10 tickets per model
"""

import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from mdutils.mdutils import MdUtils

from helpers import fetch_dataset, train_test_split
from train import get_compiled_model
from smart_generator import (
    HybridGenerator,
    SmartTicketGenerator,
)
from logger import setup_logger

setup_logger()

# Available models
AVAILABLE_MODELS = ["original", "multi_output", "transformer"]


def load_model(model_name: str):
    """Load a trained model by name."""
    model = get_compiled_model(architecture=model_name)
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 10, 7))
    _ = model(dummy_input)
    
    # Load weights
    weights_path = f"model/{model_name}_best.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model
    else:
        print(f"  Warning: No weights found at {weights_path}")
        return None


def get_model_prediction(model, X_latest: np.ndarray) -> dict:
    """Get prediction from a single model."""
    pred_probs = model.predict(X_latest, verbose=0)
    pred_probs = np.squeeze(pred_probs)
    
    # Greedy prediction
    greedy = np.argmax(pred_probs, axis=1) + 1  # 1-indexed
    
    # Get probability distribution
    main_probs = np.mean(pred_probs[:6, :37], axis=0)
    main_probs = main_probs / main_probs.sum()
    
    # Top 5 numbers by probability
    top_indices = np.argsort(main_probs)[-5:][::-1]
    top_probs = [(idx + 1, main_probs[idx]) for idx in top_indices]
    
    return {
        "greedy": greedy,
        "probs": pred_probs,
        "top_numbers": top_probs,
    }


def generate_predictions(
    models: list,
    n_tickets: int = 5,
    model_weight: float = 0.5,
    temperature: float = 1.5,
) -> dict:
    """Generate predictions from all specified models."""
    
    print("\n" + "=" * 70)
    print("                    ILOTTO PREDICTIONS")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading lottery data...")
    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    historical_data = lotto_ds.values - 1  # 0-indexed
    
    # Prepare latest input for prediction
    X_latest = X_test[0][1:]
    X_latest = np.concatenate([X_latest, y_test[0].reshape(1, 7)], axis=0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
    
    print(f"      Loaded {len(lotto_ds)} historical draws")
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "smart_tickets": [],
        "hybrid_tickets": {},
    }
    
    # Generate predictions from each model
    print(f"\n[2/4] Generating predictions from {len(models)} models...")
    
    for model_name in models:
        print(f"\n      Loading {model_name}...")
        model = load_model(model_name)
        
        if model is None:
            print(f"      Skipping {model_name} (no weights)")
            continue
        
        # Get model prediction
        pred = get_model_prediction(model, X_latest)
        results["models"][model_name] = pred
        print(f"      {model_name}: {pred['greedy']}")
        
        # Generate hybrid tickets
        print("      Generating hybrid tickets...")
        hybrid_gen = HybridGenerator(
            model=model,
            historical_data=historical_data,
            model_weight=model_weight,
        )
        
        hybrid_tickets = hybrid_gen.generate_hybrid_tickets(
            X_input=X_latest,
            n_tickets=n_tickets,
            n_candidates=3000,
            temperature=temperature,
        )
        results["hybrid_tickets"][model_name] = hybrid_tickets
    
    # Generate pure smart tickets
    print("\n[3/4] Generating smart tickets (no model)...")
    smart_gen = SmartTicketGenerator(historical_data)
    smart_tickets = smart_gen.generate_optimized_tickets(
        n_tickets=n_tickets,
        n_candidates=5000,
    )
    results["smart_tickets"] = smart_tickets
    
    return results


def write_prediction_file(results: dict, output_file: str = "PREDICTION"):
    """Write predictions to markdown file."""
    
    print(f"\n[4/4] Writing {output_file}.md...")
    
    md = MdUtils(file_name=output_file, title="ILotto - Lottery Predictions")
    
    # Header
    md.new_paragraph(f"**Generated:** {results['timestamp']}")
    md.new_paragraph(
        "Predictions from multiple neural network architectures combined with "
        "smart ticket scoring to maximize expected payout."
    )
    
    # Model Predictions Summary
    md.new_header(level=1, title="Model Predictions")
    
    if results["models"]:
        # Summary table
        table = ["Model", "Prediction (Greedy)", "Top 5 Numbers by Probability"]
        for model_name, pred in results["models"].items():
            greedy_str = " ".join(str(n) for n in pred["greedy"])
            top_str = ", ".join(f"{n}({p:.1%})" for n, p in pred["top_numbers"])
            table.extend([model_name, f"[{greedy_str}]", top_str])
        
        md.new_table(
            columns=3,
            rows=len(results["models"]) + 1,
            text=table,
            text_align="left"
        )
    else:
        md.new_paragraph("No models available.")
    
    # Hybrid Tickets (per model)
    md.new_header(level=1, title="Hybrid Tickets (Model + Smart Scoring)")
    md.new_paragraph(
        "These tickets combine neural network predictions with smart scoring. "
        "They are sampled from the model's probability distribution and filtered "
        "to avoid popular combinations."
    )
    
    for model_name, tickets in results["hybrid_tickets"].items():
        md.new_header(level=2, title=f"{model_name.replace('_', ' ').title()} Model")
        
        table = ["#", "Numbers", "Bonus", "Combined", "Model", "Smart"]
        for i, (ticket, scores) in enumerate(tickets, 1):
            nums = " ".join(f"{n:2d}" for n in ticket[:6])
            table.extend([
                str(i),
                f"[{nums}]",
                str(ticket[6]),
                f"{scores['combined']:.3f}",
                f"{scores['model_score']:.3f}",
                f"{scores['total']:.3f}",
            ])
        
        md.new_table(
            columns=6,
            rows=len(tickets) + 1,
            text=table,
            text_align="center"
        )
    
    # Smart Tickets (no model)
    md.new_header(level=1, title="Smart Tickets (No Model)")
    md.new_paragraph(
        "Pure heuristic-based tickets that avoid popular combinations. "
        "These favor high numbers (32-37), avoid sequences, and ensure good spread."
    )
    
    table = ["#", "Numbers", "Bonus", "Score", "High", "Spread", "Seq"]
    for i, (ticket, scores) in enumerate(results["smart_tickets"], 1):
        nums = " ".join(f"{n:2d}" for n in ticket[:6])
        table.extend([
            str(i),
            f"[{nums}]",
            str(ticket[6]),
            f"{scores['total']:.3f}",
            f"{scores['high_numbers']:.2f}",
            f"{scores['spread']:.2f}",
            f"{scores['sequence_avoidance']:.2f}",
        ])
    
    md.new_table(
        columns=7,
        rows=len(results["smart_tickets"]) + 1,
        text=table,
        text_align="center"
    )
    
    # Recommendations
    md.new_header(level=1, title="Recommendations")
    
    # Find best hybrid ticket across all models
    best_ticket = None
    best_score = 0
    best_model = None
    
    for model_name, tickets in results["hybrid_tickets"].items():
        if tickets and tickets[0][1]["combined"] > best_score:
            best_ticket = tickets[0][0]
            best_score = tickets[0][1]["combined"]
            best_model = model_name
    
    if best_ticket:
        nums = " ".join(f"{n:2d}" for n in best_ticket[:6])
        md.new_paragraph(f"**Best Hybrid Ticket** (from {best_model}):")
        md.new_line(f"[{nums}] + Bonus: {best_ticket[6]}", bold_italics_code="b")
    
    if results["smart_tickets"]:
        best_smart = results["smart_tickets"][0][0]
        nums = " ".join(f"{n:2d}" for n in best_smart[:6])
        md.new_paragraph("**Best Smart Ticket:**")
        md.new_line(f"[{nums}] + Bonus: {best_smart[6]}", bold_italics_code="b")
    
    # Disclaimer
    md.new_header(level=1, title="Disclaimer")
    md.new_paragraph(
        "**Important:** Lottery numbers are random by design. No prediction method "
        "can reliably predict future draws. These predictions are for educational "
        "and entertainment purposes only. The 'smart' tickets don't increase your "
        "probability of winning - they only maximize expected payout IF you win by "
        "avoiding popular combinations. Please gamble responsibly."
    )
    
    md.create_md_file()
    print(f"      Created {output_file}.md")


def main():
    parser = argparse.ArgumentParser(
        description="Generate lottery predictions from all models"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help="Models to use (default: all)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="Number of tickets per model (default: 5)",
    )
    parser.add_argument(
        "--model-weight", "-w",
        type=float,
        default=0.5,
        help="Weight for model vs smart scoring (default: 0.5)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.5,
        help="Sampling temperature (default: 1.5)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="PREDICTION",
        help="Output filename without extension (default: PREDICTION)",
    )
    
    args = parser.parse_args()
    
    # Generate predictions
    results = generate_predictions(
        models=args.models,
        n_tickets=args.count,
        model_weight=args.model_weight,
        temperature=args.temperature,
    )
    
    # Write to file
    write_prediction_file(results, args.output)
    
    print("\n" + "=" * 70)
    print("                    PREDICTIONS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {args.output}.md")
    print("\nQuick view of best tickets:")
    
    # Print best from each model
    for model_name, tickets in results["hybrid_tickets"].items():
        if tickets:
            t, s = tickets[0]
            nums = " ".join(f"{n:2d}" for n in t[:6])
            print(f"  {model_name:15} [{nums}] + {t[6]}  (score: {s['combined']:.3f})")
    
    if results["smart_tickets"]:
        t, s = results["smart_tickets"][0]
        nums = " ".join(f"{n:2d}" for n in t[:6])
        print(f"  {'smart':15} [{nums}] + {t[6]}  (score: {s['total']:.3f})")
    
    print()


if __name__ == "__main__":
    main()
