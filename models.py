"""
Alternative model architectures for ILotto.

This module provides different neural network architectures for lottery prediction,
primarily for educational purposes to explore different ML approaches.

Key insight: Lottery is random, so no model will beat random baseline.
But these architectures demonstrate different ML concepts.
"""

import tensorflow as tf
from tensorflow.keras import Model, saving
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
import numpy as np


# Configuration
MAIN_NUMBERS = 37
BONUS_NUMBERS = 7
SEQUENCE_LENGTH = 10  # Number of previous draws to consider
OUTPUT_LENGTH = 7  # 6 main + 1 bonus


@saving.register_keras_serializable(package="ILotto")
class MultiOutputLotto(Model):
    """
    Multi-output model that predicts each ball position independently.
    
    This avoids the mode collapse problem by treating each position as
    a separate classification task.
    """
    
    def __init__(
        self,
        n_main: int = MAIN_NUMBERS,
        n_bonus: int = BONUS_NUMBERS,
        embed_dim: int = 32,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_main = n_main
        self.n_bonus = n_bonus
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Shared embedding for all ball positions
        self.main_embedding = Embedding(n_main, embed_dim)
        self.bonus_embedding = Embedding(n_bonus, embed_dim)
        
        # Shared LSTM encoder
        self.lstm1 = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.lstm2 = LSTM(lstm_units, dropout=dropout_rate)
        
        # Separate output heads for each ball position
        self.main_heads = [
            Dense(n_main, activation='softmax', name=f'main_ball_{i}')
            for i in range(6)
        ]
        self.bonus_head = Dense(n_bonus, activation='softmax', name='bonus_ball')
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch, seq_len, 7)
        
        # Embed main balls (first 6) and bonus ball separately
        main_inputs = inputs[:, :, :6]  # (batch, seq_len, 6)
        bonus_inputs = inputs[:, :, 6:7]  # (batch, seq_len, 1)
        
        # Embed and concatenate
        main_embedded = self.main_embedding(main_inputs)  # (batch, seq_len, 6, embed_dim)
        bonus_embedded = self.bonus_embedding(bonus_inputs)  # (batch, seq_len, 1, embed_dim)
        
        # Reshape for LSTM
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        main_flat = tf.reshape(main_embedded, (batch_size, seq_len, 6 * self.embed_dim))
        bonus_flat = tf.reshape(bonus_embedded, (batch_size, seq_len, self.embed_dim))
        
        combined = Concatenate()([main_flat, bonus_flat])
        
        # LSTM encoding
        x = self.lstm1(combined, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout(x, training=training)
        
        # Generate outputs for each position
        main_outputs = [head(x) for head in self.main_heads]
        bonus_output = self.bonus_head(x)
        
        # Pad bonus output to match main output size for stacking
        # Bonus has 7 classes, main has 37 - pad bonus with zeros
        bonus_padded = tf.pad(bonus_output, [[0, 0], [0, self.n_main - self.n_bonus]])
        
        # Stack outputs: (batch, 7, 37)
        return tf.stack(main_outputs + [bonus_padded], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_main': self.n_main,
            'n_bonus': self.n_bonus,
            'embed_dim': self.embed_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
        })
        return config


@saving.register_keras_serializable(package="ILotto")
class TransformerLotto(Model):
    """
    Transformer-based model for lottery prediction.
    
    Uses self-attention to capture relationships between numbers
    across different draws.
    """
    
    def __init__(
        self,
        n_main: int = MAIN_NUMBERS,
        n_bonus: int = BONUS_NUMBERS,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_main = n_main
        self.n_bonus = n_bonus
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Embeddings
        self.number_embedding = Embedding(n_main, embed_dim)
        self.position_embedding = Embedding(100, embed_dim)  # Position in sequence
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append({
                'attention': MultiHeadAttention(
                    num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
                ),
                'ffn': tf.keras.Sequential([
                    Dense(ff_dim, activation='gelu'),
                    Dropout(dropout_rate),
                    Dense(embed_dim),
                ]),
                'norm1': LayerNormalization(epsilon=1e-6),
                'norm2': LayerNormalization(epsilon=1e-6),
                'dropout1': Dropout(dropout_rate),
                'dropout2': Dropout(dropout_rate),
            })
        
        # Output
        self.global_pool = GlobalAveragePooling1D()
        self.output_dense = Dense(256, activation='gelu')
        self.dropout_out = Dropout(dropout_rate)
        
        # Separate heads for each output position
        self.output_heads = [Dense(n_main, activation='softmax') for _ in range(6)]
        self.bonus_head = Dense(n_bonus, activation='softmax')
    
    def call(self, inputs, training=None):
        # inputs: (batch, seq_len, 7)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Flatten sequence: (batch, seq_len * 7)
        x = tf.reshape(inputs, (batch_size, seq_len * 7))
        
        # Embed numbers
        x = self.number_embedding(x)  # (batch, seq_len * 7, embed_dim)
        
        # Add position embeddings
        positions = tf.range(seq_len * 7)
        pos_embed = self.position_embedding(positions)
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            # Self attention
            attn_output = block['attention'](x, x, training=training)
            attn_output = block['dropout1'](attn_output, training=training)
            x = block['norm1'](x + attn_output)
            
            # Feed forward
            ffn_output = block['ffn'](x)
            ffn_output = block['dropout2'](ffn_output, training=training)
            x = block['norm2'](x + ffn_output)
        
        # Pool and output
        x = self.global_pool(x)
        x = self.output_dense(x)
        x = self.dropout_out(x, training=training)
        
        # Generate predictions
        main_outputs = [head(x) for head in self.output_heads]
        bonus_output = self.bonus_head(x)
        
        # Pad bonus to match main size
        bonus_padded = tf.pad(bonus_output, [[0, 0], [0, self.n_main - self.n_bonus]])
        
        return tf.stack(main_outputs + [bonus_padded], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_main': self.n_main,
            'n_bonus': self.n_bonus,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        })
        return config


@saving.register_keras_serializable(package="ILotto")  
class SetPredictionLotto(Model):
    """
    Set prediction model - predicts numbers as an unordered set.
    
    Uses a different approach: instead of predicting exact positions,
    predicts which numbers will appear in the draw (binary classification
    for each number).
    
    This is more aligned with how lottery works - position doesn't matter.
    """
    
    def __init__(
        self,
        n_main: int = MAIN_NUMBERS,
        n_bonus: int = BONUS_NUMBERS,
        embed_dim: int = 32,
        hidden_units: list = [128, 64],
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_main = n_main
        self.n_bonus = n_bonus
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Embedding
        self.embedding = Embedding(n_main, embed_dim)
        
        # LSTM for sequence encoding
        self.lstm = LSTM(hidden_units[0], dropout=dropout_rate)
        
        # Dense layers
        self.dense_layers = []
        for units in hidden_units[1:]:
            self.dense_layers.append(Dense(units, activation='relu'))
            self.dense_layers.append(Dropout(dropout_rate))
        
        # Output: probability for each main number (multi-label, not softmax)
        self.main_output = Dense(n_main, activation='sigmoid', name='main_probs')
        self.bonus_output = Dense(n_bonus, activation='softmax', name='bonus')
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Embed all numbers
        x = self.embedding(inputs)  # (batch, seq_len, 7, embed_dim)
        x = tf.reshape(x, (batch_size, seq_len, 7 * self.embed_dim))
        
        # LSTM encoding
        x = self.lstm(x, training=training)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training) if isinstance(layer, Dropout) else layer(x)
        
        # Outputs
        main_probs = self.main_output(x)  # (batch, 37) - probability for each number
        bonus_probs = self.bonus_output(x)  # (batch, 7)
        
        return main_probs, bonus_probs
    
    def predict_top_k(self, inputs, k: int = 6):
        """Predict top-k most likely numbers."""
        main_probs, bonus_probs = self(inputs, training=False)
        
        # Get top-k main numbers
        top_k_values, top_k_indices = tf.math.top_k(main_probs, k=k)
        
        # Get most likely bonus
        bonus = tf.argmax(bonus_probs, axis=1)
        
        return top_k_indices.numpy(), bonus.numpy()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_main': self.n_main,
            'n_bonus': self.n_bonus,
            'embed_dim': self.embed_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
        })
        return config


class DiversityLoss(tf.keras.losses.Loss):
    """
    Custom loss that penalizes predicting the same number repeatedly.
    
    Helps prevent mode collapse where the model always predicts the same numbers.
    """
    
    def __init__(self, diversity_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.diversity_weight = diversity_weight
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        # Standard cross entropy
        ce = self.ce_loss(y_true, y_pred)
        
        # Diversity penalty: encourage different predictions for each position
        # Compute average prediction across positions
        avg_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
        
        # Penalize if predictions are too similar to average
        diversity_penalty = tf.reduce_mean(
            tf.reduce_sum(y_pred * avg_pred, axis=-1)
        )
        
        return ce + self.diversity_weight * diversity_penalty


def create_model(architecture: str = 'original', **kwargs) -> Model:
    """
    Factory function to create different model architectures.
    
    Args:
        architecture: One of 'original', 'multi_output', 'transformer', 'set_prediction'
        **kwargs: Architecture-specific parameters
    
    Returns:
        Compiled model
    """
    if architecture == 'original':
        from ilotto import ILotto
        model = ILotto(**kwargs)
        
    elif architecture == 'multi_output':
        model = MultiOutputLotto(**kwargs)
        
    elif architecture == 'transformer':
        model = TransformerLotto(**kwargs)
        
    elif architecture == 'set_prediction':
        model = SetPredictionLotto(**kwargs)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def compile_model(
    model: Model, 
    architecture: str = 'original',
    learning_rate: float = 0.001,
    use_diversity_loss: bool = False,
):
    """
    Compile model with appropriate loss and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if architecture == 'set_prediction':
        # Binary cross entropy for multi-label main ball prediction
        model.compile(
            optimizer=optimizer,
            loss={
                'main_probs': 'binary_crossentropy',
                'bonus': 'sparse_categorical_crossentropy',
            },
            metrics={
                'main_probs': 'accuracy',
                'bonus': 'accuracy',
            }
        )
    else:
        # Standard classification for position-based models
        loss = DiversityLoss() if use_diversity_loss else 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
        )
    
    return model


def temperature_sample(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Sample from probability distribution with temperature.
    
    Higher temperature = more random/diverse
    Lower temperature = more deterministic (closer to argmax)
    
    Args:
        probs: Probability distribution (batch, n_classes)
        temperature: Sampling temperature
    
    Returns:
        Sampled indices
    """
    if temperature == 0:
        return np.argmax(probs, axis=-1)
    
    # Apply temperature
    log_probs = np.log(probs + 1e-10) / temperature
    exp_probs = np.exp(log_probs)
    probs = exp_probs / np.sum(exp_probs, axis=-1, keepdims=True)
    
    # Sample
    samples = []
    for p in probs:
        sample = np.random.choice(len(p), p=p)
        samples.append(sample)
    
    return np.array(samples)


if __name__ == "__main__":
    # Test architectures
    print("Testing model architectures...")
    
    # Create dummy data
    batch_size = 4
    seq_length = 10
    X = np.random.randint(0, 37, size=(batch_size, seq_length, 7))
    
    # Test MultiOutputLotto
    print("\n1. MultiOutputLotto:")
    model1 = MultiOutputLotto()
    out1 = model1(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {out1.shape}")
    
    # Test TransformerLotto
    print("\n2. TransformerLotto:")
    model2 = TransformerLotto()
    out2 = model2(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {out2.shape}")
    
    # Test SetPredictionLotto
    print("\n3. SetPredictionLotto:")
    model3 = SetPredictionLotto()
    main_out, bonus_out = model3(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Main output shape: {main_out.shape}")
    print(f"   Bonus output shape: {bonus_out.shape}")
    
    print("\nAll architectures working!")
