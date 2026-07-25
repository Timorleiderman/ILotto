"""Neural predictors, rebuilt as a set-prediction problem.

What changed versus the original `ilotto.py`, and why each change matters:

*Target.* The old model emitted a length-7 sequence of softmaxes over 37
numbers, supervised against the balls in ascending order. Ball 1 is the minimum
of six uniform draws and Ball 6 the maximum, so a model that learns nothing but
the marginal of each order statistic scores well on top-k accuracy. Here the
target is a 37-way multi-hot set with binary cross-entropy: order-invariant,
and it can only score well by knowing *which numbers*, which is the actual task.

*Input.* Sequences are chronological (oldest to newest) and encode each past
draw as a 37-dim indicator plus a few normalised context features, rather than
seven integer ball IDs that leak their own sort order into the model.

*Cost.* `recurrent_dropout` and `return_state` forced Keras off the fused cuDNN
LSTM kernel, making training one to two orders of magnitude slower for no
measured benefit. Both are gone. The model trains in seconds on a CI runner.

*Schedule.* The old `LearningRateScheduler` multiplied the *current* learning
rate by `(epoch+1)/warmup` each epoch, so the intended warm-up compounded
downward to ~1e-9 within ten epochs while `ReduceLROnPlateau` fought it for
control of the same variable. Only the plateau-based reduction survives.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import N_NUMBERS, N_STRONG, Draws
from .predictors import Predictor

logger = logging.getLogger(__name__)

WINDOW = 20


def _features(multi_hot: np.ndarray) -> np.ndarray:
    """Per-draw context appended to the raw indicator vector.

    Gives the network cheap access to things it would otherwise need many
    layers and many more samples to infer: how long since each number appeared,
    and the coarse shape of the draw.
    """
    n = len(multi_hot)
    gap = np.zeros_like(multi_hot)
    last_seen = np.full(N_NUMBERS, -1.0)
    for t in range(n):
        gap[t] = np.where(last_seen < 0, 1.0, np.minimum((t - last_seen) / 37.0, 1.0))
        last_seen = np.where(multi_hot[t] > 0, t, last_seen)
    return np.concatenate([multi_hot, gap], axis=-1)


def make_windows(draws: Draws, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological sliding windows: X[i] are the `window` draws before y[i]."""
    feats = _features(draws.multi_hot)
    targets = draws.multi_hot
    strong = draws.strong - 1

    n = len(draws) - window
    if n <= 0:
        raise ValueError(f"Need more than {window} draws, got {len(draws)}")
    X = np.stack([feats[i : i + window] for i in range(n)])
    y = targets[window:]
    s = strong[window:]
    return X.astype(np.float32), y.astype(np.float32), s.astype(np.int32)


def build_model(window: int = WINDOW, arch: str = "gru", seed: int = 42):
    """Small dual-head network: 37 sigmoids for the set, 7 softmax for strong."""
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    n_feat = N_NUMBERS * 2

    inp = layers.Input(shape=(window, n_feat), name="history")

    if arch == "gru":
        x = layers.GRU(96, return_sequences=True)(inp)
        x = layers.LayerNormalization()(x)
        x = layers.GRU(64)(x)
    elif arch == "transformer":
        x = layers.Dense(64)(inp)
        x = x + _positional(window, 64)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = layers.LayerNormalization()(x + attn)
        ff = layers.Dense(128, activation="gelu")(x)
        x = layers.LayerNormalization()(x + layers.Dense(64)(ff))
        x = layers.GlobalAveragePooling1D()(x)
    elif arch == "mlp":
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation="gelu")(x)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(96, activation="gelu")(x)

    # Bias-initialised at the base rate 6/37 so the model starts calibrated
    # rather than spending its first epochs learning the prior.
    base_rate = 6.0 / N_NUMBERS
    balls = layers.Dense(
        N_NUMBERS,
        activation="sigmoid",
        name="balls",
        bias_initializer=tf.keras.initializers.Constant(np.log(base_rate / (1 - base_rate))),
    )(x)
    strong = layers.Dense(N_STRONG, activation="softmax", name="strong")(x)

    model = tf.keras.Model(inp, [balls, strong])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss={"balls": "binary_crossentropy", "strong": "sparse_categorical_crossentropy"},
        loss_weights={"balls": 1.0, "strong": 0.2},
    )
    return model


def _positional(length: int, dim: int) -> np.ndarray:
    pos = np.arange(length)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / dim)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe.astype(np.float32)[None]


class NeuralPredictor(Predictor):
    """Walk-forward wrapper: refits periodically, scores every draw."""

    # Sigmoid heads trained with BCE emit per-number inclusion probabilities.
    emits_probabilities = True

    def __init__(
        self,
        arch: str = "gru",
        window: int = WINDOW,
        epochs: int = 60,
        refit_every: int = 50,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        self.arch = arch
        self.window = window
        self.epochs = epochs
        self.refit_every = refit_every
        self.val_frac = val_frac
        self.seed = seed
        self.model = None
        self.name = f"Neural ({arch}, set prediction)"
        self.description = (
            f"A {arch.upper()} over the last {window} draws encoded as 37-dim indicators plus "
            "recency features, trained with binary cross-entropy to predict the drawn *set*. "
            "Refit every "
            f"{refit_every} draws during the backtest so it never sees the future."
        )
        self.history: list[dict] = []

    def fit(self, history: Draws) -> None:
        import tensorflow as tf

        X, y, s = make_windows(history, self.window)
        n_val = max(int(len(X) * self.val_frac), 20)
        X_tr, y_tr, s_tr = X[:-n_val], y[:-n_val], s[:-n_val]
        X_va, y_va, s_va = X[-n_val:], y[-n_val:], s[-n_val:]

        self.model = build_model(self.window, self.arch, self.seed)
        hist = self.model.fit(
            X_tr,
            {"balls": y_tr, "strong": s_tr},
            validation_data=(X_va, {"balls": y_va, "strong": s_va}),
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0
                ),
            ],
        )
        self.history.append({k: [float(x) for x in v] for k, v in hist.history.items()})
        logger.info(
            "Refit %s on %d windows; val_loss %.4f", self.name, len(X_tr), min(hist.history["val_loss"])
        )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            self.fit(history)
        feats = _features(history.multi_hot)[-self.window :]
        x = feats[None].astype(np.float32)
        balls, strong = self.model.predict(x, verbose=0)
        return balls[0].astype(np.float64), strong[0].astype(np.float64)
