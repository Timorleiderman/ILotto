"""
Training script for ILotto models.

Supports multiple architectures:
- original: The original LSTM Seq2Seq with attention
- multi_output: Separate heads for each ball position
- transformer: Transformer-based architecture
- set_prediction: Treats output as unordered set

Usage:
    # Train original model
    python train.py

    # Train specific architecture
    python train.py --model transformer --epochs 100

    # Train all models and compare
    python train.py --compare-all
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from typing import Optional, Tuple, Dict

from ilotto import ILotto
from models import (
    MultiOutputLotto, 
    TransformerLotto, 
    SetPredictionLotto,
    DiversityLoss,
)
from logger import setup_logger
from helpers import fetch_dataset, train_test_split
from metrics import LotteryMetrics

setup_logger()
logger = logging.getLogger(__name__)

# Default hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def get_compiled_model(
    architecture: str = "original",
    learning_rate: float = LEARNING_RATE,
    use_diversity_loss: bool = False,
) -> tf.keras.Model:
    """
    Create and compile a model of the specified architecture.
    
    Args:
        architecture: One of 'original', 'multi_output', 'transformer', 'set_prediction'
        learning_rate: Learning rate for optimizer
        use_diversity_loss: Whether to use diversity loss to prevent mode collapse
    
    Returns:
        Compiled Keras model
    """
    logger.info(f"Creating {architecture} model...")
    
    if architecture == "original":
        model = ILotto()
        loss = DiversityLoss() if use_diversity_loss else "sparse_categorical_crossentropy"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10_acc"),
            ],
        )
        
    elif architecture == "multi_output":
        model = MultiOutputLotto()
        loss = DiversityLoss() if use_diversity_loss else "sparse_categorical_crossentropy"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
            ],
        )
        
    elif architecture == "transformer":
        model = TransformerLotto()
        loss = DiversityLoss() if use_diversity_loss else "sparse_categorical_crossentropy"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
            ],
        )
        
    elif architecture == "set_prediction":
        model = SetPredictionLotto()
        # Set prediction uses binary cross entropy for main balls
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def prepare_data_for_architecture(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data format for specific architecture.
    
    Some architectures need different label formats.
    """
    if architecture == "set_prediction":
        # Convert to multi-hot encoding for main balls
        def to_multihot(y, n_classes=37):
            multihot = np.zeros((len(y), n_classes), dtype=np.float32)
            for i, row in enumerate(y):
                for ball in row[:6]:  # Only main balls
                    multihot[i, int(ball)] = 1.0
            return multihot
        
        y_train_main = to_multihot(y_train)
        y_test_main = to_multihot(y_test)
        
        # Bonus ball stays as is
        y_train_bonus = y_train[:, 6]
        y_test_bonus = y_test[:, 6]
        
        return X_train, (y_train_main, y_train_bonus), X_test, (y_test_main, y_test_bonus)
    
    return X_train, y_train, X_test, y_test


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_name: str = "model",
    checkpoint_dir: str = "model",
    early_stopping: bool = True,
    patience: int = 10,
) -> pd.DataFrame:
    """
    Train a model with proper callbacks and logging.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_test, y_test: Validation data
        epochs: Number of training epochs
        batch_size: Training batch size
        model_name: Name for saving model
        checkpoint_dir: Directory for checkpoints
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
    
    Returns:
        Training history as DataFrame
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks
    callback_list = []
    
    # Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.weights.h5")
    ckp = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    callback_list.append(ckp)
    
    # Early stopping
    if early_stopping:
        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
        callback_list.append(es)
    
    # Learning rate reduction
    lr_reduce = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    callback_list.append(lr_reduce)
    
    # TensorBoard (optional)
    tb_log_dir = os.path.join(checkpoint_dir, "logs", model_name)
    tensorboard = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
    callback_list.append(tensorboard)
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")
    
    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        callbacks=callback_list,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    
    # Save final model
    model_path = os.path.join(checkpoint_dir, f"{model_name}.keras")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(checkpoint_dir, f"{model_name}_history.csv")
    hist_df.to_csv(hist_path, index=False)
    logger.info(f"Training history saved to {hist_path}")
    
    return hist_df


def plot_training_history(
    history: pd.DataFrame,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1 = axes[0]
    ax1.plot(history["loss"], label="Training", linewidth=2)
    if "val_loss" in history.columns:
        ax1.plot(history["val_loss"], label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics
    ax2 = axes[1]
    metric_cols = [c for c in history.columns if "acc" in c.lower() and "val" not in c]
    val_metric_cols = [c for c in history.columns if "acc" in c.lower() and "val" in c]
    
    for col in metric_cols:
        ax2.plot(history[col], label=f"Train {col}", linewidth=2)
    for col in val_metric_cols:
        ax2.plot(history[col], label=f"Val {col}", linewidth=2, linestyle="--")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    architecture: str = "original",
) -> Dict:
    """
    Evaluate model and compare with random baseline.
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    predictions = model.predict(X_test)
    
    if architecture == "set_prediction":
        # For set prediction, get top-6 indices
        main_probs, bonus_probs = predictions
        pred_main = np.argsort(main_probs, axis=1)[:, -6:]  # Top 6
        pred_bonus = np.argmax(bonus_probs, axis=1)
        pred = np.column_stack([pred_main, pred_bonus])
    else:
        # For other architectures, argmax each position
        pred = np.argmax(predictions, axis=2)
    
    # Calculate metrics
    metrics = LotteryMetrics()
    comparison = metrics.compare_with_baseline(pred, y_test)
    
    logger.info(f"Model avg matches: {comparison['model']['avg_matches']:.3f}")
    logger.info(f"Baseline avg matches: {comparison['baseline_avg_matches']:.3f}")
    logger.info(f"Improvement: {comparison['improvement_percent']:+.2f}%")
    
    return comparison


def train_and_compare_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """
    Train all architectures and compare results.
    """
    architectures = ["original", "multi_output", "transformer"]
    results = {}
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch.upper()} architecture")
        print(f"{'='*60}\n")
        
        # Prepare data
        X_tr, y_tr, X_te, y_te = prepare_data_for_architecture(
            arch, X_train, y_train, X_test, y_test
        )
        
        # Create and train model
        model = get_compiled_model(arch, use_diversity_loss=True)
        history = train_model(
            model, X_tr, y_tr, X_te, y_te,
            epochs=epochs,
            batch_size=batch_size,
            model_name=arch,
            early_stopping=True,
        )
        
        # Plot training
        plot_training_history(
            history,
            model_name=arch.replace("_", " ").title(),
            save_path=f"model/{arch}_training.png",
            show=False,
        )
        
        # Evaluate
        comparison = evaluate_model(model, X_test, y_test, arch)
        results[arch] = {
            "history": history,
            "comparison": comparison,
        }
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Architecture':<20} {'Avg Matches':<15} {'vs Random':<15}")
    print("-"*50)
    
    for arch, result in results.items():
        comp = result["comparison"]
        print(f"{arch:<20} {comp['model']['avg_matches']:<15.3f} {comp['improvement_percent']:+.2f}%")
    
    # Random baseline
    print(f"{'random (baseline)':<20} {results['original']['comparison']['baseline_avg_matches']:<15.3f} +0.00%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ILotto models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="original",
        choices=["original", "multi_output", "transformer", "set_prediction"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--diversity-loss",
        action="store_true",
        help="Use diversity loss to prevent mode collapse",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Train all architectures and compare",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading lottery data...")
    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    if args.compare_all:
        # Train and compare all architectures
        train_and_compare_all(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        # Train single architecture
        architecture = args.model
        
        # Prepare data
        X_tr, y_tr, X_te, y_te = prepare_data_for_architecture(
            architecture, X_train, y_train, X_test, y_test
        )
        
        # Create model
        model = get_compiled_model(
            architecture,
            learning_rate=args.learning_rate,
            use_diversity_loss=args.diversity_loss,
        )
        
        # Print model summary
        print(f"\nModel: {architecture}")
        model.build((None, 10, 7))
        model.summary()
        
        # Train
        history = train_model(
            model,
            X_tr, y_tr, X_te, y_te,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=architecture,
            early_stopping=not args.no_early_stopping,
        )
        
        # Plot training
        plot_training_history(
            history,
            model_name=architecture.replace("_", " ").title(),
            save_path=f"model/{architecture}_training.png",
            show=True,
        )
        
        # Evaluate
        evaluate_model(model, X_test, y_test, architecture)
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Model saved to: model/{architecture}.keras")
        print(f"Training plot saved to: model/{architecture}_training.png")
        print("="*60)


if __name__ == "__main__":
    main()
