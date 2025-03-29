import logging
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

from ilotto import ILotto
from logger import setup_logger
from helpers import fetch_dataset, train_test_split

setup_logger()
logger = logging.getLogger(__name__)

EPOCHS = 60
BATCH_SIZE = 32
MODEL_CP_PATH = "model/training/cp.weights.h5"
SAVED_MODEL_PATH = "model/training/ilotto.keras"

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


def get_compiled_model():
    # Define your model
    model = ILotto()

    # Define metric
    sparse_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name="sparse_top_k"
    )

    # Compile the model
    model.compile(
        optimizer="adam",
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
    epochs=250,
    batch_size=32,
    checkpoint_path="training/cp.weights.h5",
    save_model_path="training/model.h5",
):

    # Define checkpoint callback
    ckp = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_sparse_top_k",
        verbose=1,
        save_weights_only=True,
        save_best_only=False,
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

    # Save model weights
    model.save_weights(checkpoint_path)
    model.save(save_model_path)
    return hist


if __name__ == "__main__":

    orig_lotto_csv="input/Orig_IL_lotto.csv"
    lotto_csv_file="input/lotto_IL_filtered.csv"

    lotto_ds = fetch_dataset(orig_lotto_csv, lotto_csv_file)
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)


    model = get_compiled_model()
    hist = train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_path=MODEL_CP_PATH,
        save_model_path=SAVED_MODEL_PATH
    )
