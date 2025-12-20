"""
Visualization module for ILotto analysis.

Provides charts and visual analytics for lottery data and model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from typing import Dict, List, Tuple, Optional
import os


# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
}


def plot_number_frequency(
    data: np.ndarray,
    title: str = "Lottery Number Frequency Distribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot frequency distribution of main lottery numbers.
    
    Args:
        data: Historical lottery data, shape (n_draws, 7)
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    # Count frequencies for main balls (columns 0-5)
    main_balls = data[:, :6].flatten()
    freq = Counter(main_balls)
    
    numbers = list(range(1, 38))
    frequencies = [freq.get(n, 0) for n in numbers]
    
    # Expected frequency (uniform distribution)
    expected = len(main_balls) / 37
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color bars based on deviation from expected
    colors = []
    for f in frequencies:
        if f > expected * 1.1:
            colors.append(COLORS['success'])  # Above expected
        elif f < expected * 0.9:
            colors.append(COLORS['danger'])  # Below expected
        else:
            colors.append(COLORS['primary'])  # Near expected
    
    ax.bar(numbers, frequencies, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=expected, color=COLORS['warning'], linestyle='--', linewidth=2, label=f'Expected ({expected:.0f})')
    
    ax.set_xlabel('Number', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(numbers)
    ax.legend()
    
    # Add legend for colors
    legend_elements = [
        mpatches.Patch(color=COLORS['success'], label='Above expected (>10%)'),
        mpatches.Patch(color=COLORS['primary'], label='Near expected'),
        mpatches.Patch(color=COLORS['danger'], label='Below expected (<10%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_bonus_frequency(
    data: np.ndarray,
    title: str = "Bonus Ball Frequency Distribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot frequency distribution of bonus ball."""
    bonus_balls = data[:, 6]
    freq = Counter(bonus_balls)
    
    numbers = list(range(1, 8))
    frequencies = [freq.get(n, 0) for n in numbers]
    expected = len(bonus_balls) / 7
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(numbers, frequencies, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax.axhline(y=expected, color=COLORS['warning'], linestyle='--', linewidth=2, label=f'Expected ({expected:.0f})')
    
    ax.set_xlabel('Bonus Number', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(numbers)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_number_heatmap(
    data: np.ndarray,
    title: str = "Number Co-occurrence Heatmap",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot heatmap showing how often pairs of numbers appear together.
    """
    from itertools import combinations
    
    # Count pair occurrences
    n_numbers = 37
    cooccurrence = np.zeros((n_numbers, n_numbers))
    
    for draw in data:
        main_balls = draw[:6]
        for i, j in combinations(main_balls, 2):
            i, j = int(i) - 1, int(j) - 1  # Convert to 0-indexed
            cooccurrence[i, j] += 1
            cooccurrence[j, i] += 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(cooccurrence, cmap='YlOrRd')
    
    ax.set_xlabel('Number', fontsize=12)
    ax.set_ylabel('Number', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(0, 37, 5))
    ax.set_yticks(np.arange(0, 37, 5))
    ax.set_xticklabels(np.arange(1, 38, 5))
    ax.set_yticklabels(np.arange(1, 38, 5))
    
    plt.colorbar(im, ax=ax, label='Co-occurrence count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_match_distribution(
    match_distribution: Dict[int, int],
    baseline_distribution: Optional[Dict[int, int]] = None,
    title: str = "Match Distribution: Model vs Random Baseline",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of match distributions between model and baseline.
    """
    matches = list(range(7))
    model_counts = [match_distribution.get(i, 0) for i in matches]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(matches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_counts, width, label='Model', color=COLORS['primary'])
    
    if baseline_distribution:
        baseline_counts = [baseline_distribution.get(i, 0) for i in matches]
        ax.bar(x + width/2, baseline_counts, width, label='Random Baseline', color=COLORS['secondary'])
    
    ax.set_xlabel('Number of Matches', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(matches)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_model_comparison(
    model_metrics: Dict,
    baseline_avg: float,
    baseline_std: float,
    title: str = "Model Performance vs Random Baseline",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of model performance against random baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Average matches comparison
    ax1 = axes[0]
    
    categories = ['Model', 'Random\nBaseline', 'Expected\n(Theoretical)']
    values = [model_metrics['avg_matches'], baseline_avg, model_metrics['expected_random']]
    errors = [0, baseline_std, 0]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['info']]
    
    bars = ax1.bar(categories, values, color=colors, yerr=errors, capsize=5)
    ax1.set_ylabel('Average Matches per Draw', fontsize=12)
    ax1.set_title('Average Match Comparison', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right plot: Win rates
    ax2 = axes[1]
    
    tiers = list(model_metrics['win_rates'].keys())
    rates = list(model_metrics['win_rates'].values())
    
    bars = ax2.bar(tiers, rates, color=COLORS['success'])
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_xlabel('Match Tier', fontsize=12)
    ax2.set_title('Win Rates by Tier', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, rates):
        ax2.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_randomness_tests(
    test_results: Dict,
    title: str = "Statistical Randomness Tests",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize results of randomness tests.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tests = []
    p_values = []
    colors = []
    
    for test_name, result in test_results.items():
        tests.append(result['test'].replace(' ', '\n'))
        p_values.append(result['p_value'])
        colors.append(COLORS['success'] if result['is_random'] else COLORS['danger'])
    
    y_pos = np.arange(len(tests))
    
    bars = ax.barh(y_pos, p_values, color=colors)
    ax.axvline(x=0.05, color=COLORS['warning'], linestyle='--', linewidth=2, label='Significance threshold (0.05)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests, fontsize=10)
    ax.set_xlabel('P-Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add p-value labels
    for bar, pval in zip(bars, p_values):
        ax.annotate(f'{pval:.4f}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['success'], label='Random (p > 0.05)'),
        mpatches.Patch(color=COLORS['danger'], label='Non-random (p < 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_ticket_scores(
    tickets: List[Tuple[List[int], Dict]],
    title: str = "Smart Ticket Scores",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize scores for smart generated tickets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Total scores bar chart
    ax1 = axes[0]
    
    ticket_labels = [f"Ticket {i+1}" for i in range(len(tickets))]
    total_scores = [t[1]['total'] for t in tickets]
    
    bars = ax1.barh(ticket_labels, total_scores, color=COLORS['primary'])
    ax1.set_xlabel('Unpopularity Score', fontsize=12)
    ax1.set_title('Overall Ticket Scores', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    
    for bar, score in zip(bars, total_scores):
        ax1.annotate(f'{score:.3f}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    
    # Right: Stacked bar showing score components
    ax2 = axes[1]
    
    components = ['high_numbers', 'spread', 'sequence_avoidance', 'pattern_avoidance', 'pair_rarity']
    component_labels = ['High\nNumbers', 'Spread', 'Sequence\nAvoidance', 'Pattern\nAvoidance', 'Pair\nRarity']
    
    x = np.arange(len(tickets))
    width = 0.15
    
    for i, (comp, label) in enumerate(zip(components, component_labels)):
        scores = [t[1].get(comp, 0) for t in tickets]
        ax2.bar(x + i*width, scores, width, label=label)
    
    ax2.set_xlabel('Ticket', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Score Components by Ticket', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([f"#{i+1}" for i in range(len(tickets))])
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0, 1.1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    history: pd.DataFrame,
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1 = axes[0]
    if 'loss' in history.columns:
        ax1.plot(history['loss'], label='Training Loss', color=COLORS['primary'])
    if 'val_loss' in history.columns:
        ax1.plot(history['val_loss'], label='Validation Loss', color=COLORS['secondary'])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Accuracy/Metric
    ax2 = axes[1]
    metric_cols = [c for c in history.columns if 'acc' in c.lower() or 'top' in c.lower()]
    for col in metric_cols:
        label = col.replace('_', ' ').title()
        color = COLORS['primary'] if 'val' not in col else COLORS['secondary']
        ax2.plot(history[col], label=label, color=color)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Metrics Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def generate_all_visualizations(
    data: np.ndarray,
    metrics_comparison: Dict,
    randomness_results: Dict,
    smart_tickets: List,
    output_dir: str = "visualizations",
    show: bool = False
):
    """
    Generate all visualizations and save to directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}/...")
    
    # Number frequency
    plot_number_frequency(
        data, 
        save_path=os.path.join(output_dir, "number_frequency.png"),
        show=show
    )
    print("  - number_frequency.png")
    
    # Bonus frequency  
    plot_bonus_frequency(
        data,
        save_path=os.path.join(output_dir, "bonus_frequency.png"),
        show=show
    )
    print("  - bonus_frequency.png")
    
    # Heatmap
    plot_number_heatmap(
        data,
        save_path=os.path.join(output_dir, "cooccurrence_heatmap.png"),
        show=show
    )
    print("  - cooccurrence_heatmap.png")
    
    # Match distribution
    plot_match_distribution(
        metrics_comparison['model']['match_distribution'],
        save_path=os.path.join(output_dir, "match_distribution.png"),
        show=show
    )
    print("  - match_distribution.png")
    
    # Model comparison
    plot_model_comparison(
        metrics_comparison['model'],
        metrics_comparison['baseline_avg_matches'],
        metrics_comparison['baseline_std'],
        save_path=os.path.join(output_dir, "model_comparison.png"),
        show=show
    )
    print("  - model_comparison.png")
    
    # Randomness tests
    plot_randomness_tests(
        randomness_results,
        save_path=os.path.join(output_dir, "randomness_tests.png"),
        show=show
    )
    print("  - randomness_tests.png")
    
    # Smart tickets
    plot_ticket_scores(
        smart_tickets,
        save_path=os.path.join(output_dir, "ticket_scores.png"),
        show=show
    )
    print("  - ticket_scores.png")
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    from helpers import fetch_dataset, train_test_split
    from metrics import LotteryMetrics, RandomnessTests
    from smart_generator import SmartTicketGenerator
    
    print("Loading data...")
    lotto_ds = fetch_dataset()
    all_data = lotto_ds.values  # Keep 1-indexed for display
    
    print("Computing metrics...")
    metrics = LotteryMetrics()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    
    # Use random predictions as demo
    random_preds = metrics.generate_random_predictions(len(y_test), seed=42)
    comparison = metrics.compare_with_baseline(random_preds, y_test)
    
    print("Running randomness tests...")
    randomness = RandomnessTests(all_data - 1)  # 0-indexed for tests
    randomness_results = randomness.run_all_tests()
    
    print("Generating smart tickets...")
    generator = SmartTicketGenerator(all_data - 1)
    smart_tickets = generator.generate_optimized_tickets(n_tickets=5)
    
    print("\nGenerating visualizations...")
    generate_all_visualizations(
        all_data,
        comparison,
        randomness_results,
        smart_tickets,
        output_dir="visualizations",
        show=False
    )
