import logging
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

from ilotto import ILotto
from helpers import fetch_dataset, train_test_split


logger = logging.getLogger(__name__)


def plot_results(hist):
    print(hist["val_sparse_top_k"].max())

    plt.figure(figsize=(8, 6))
    plt.semilogy(hist["sparse_top_k"], "-r", label="Training")
    plt.semilogy(hist["val_sparse_top_k"], "-b", label="Validation")
    plt.ylabel("Sparse Top K Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()


def get_compiled_model(epochs, initial_learning_rate=0.001, lr_max=1e-4, lr_min=1e-6):
    # Define your model
    model = ILotto()

    # Define a cosine decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,  # Start LR
        decay_steps=epochs,  # Total epochs
        alpha=0.0,  # Minimum LR factor (0 means LR reaches 0)
    )

    # Use the schedule in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Define metric
    sparse_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name="sparse_top_k"
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[sparse_top_k],
    )

    return model


def train(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=200,
    batch_size=32,
    lr_max=1e-4,
    lr_min=1e-6,
    checkpoint_path="training_2/cp-{epoch:04d}.weights.h5",
):

    # Define checkpoint callback
    ckp = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_sparse_top_k",
        verbose=1,
        save_weights_only=True,
        mode="max",
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        callbacks=[ckp],  # Removed lr_schedule from callbacks
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Convert history to DataFrame
    hist = pd.DataFrame(history.history)

    # Save model weights and full model
    model.save_weights(checkpoint_path.format(epoch=0))
    model.save("model/Ilotto.keras")

    return hist


if __name__ == "__main__":

    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)

    checkpoint_path = "training_0/cp-{epoch:04d}.weights.h5"
    epochs = 300
    lr_max = 1e-4
    lr_min = 1e-6
    batch_size = 32
    model = get_compiled_model(epochs, lr_max=lr_max, lr_min=lr_min)
    hist = train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        lr_max=lr_max,
        lr_min=lr_min,
        checkpoint_path=checkpoint_path,
    )
