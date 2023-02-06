import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from ilotto import ILotto, CosineAnnealingScheduler


def train(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, lr_max=1e-4, lr_min=1e-6, checkpoint_path="training_2/cp-{epoch:04d}.ckpt"):
    
    model = ILotto()  
    sparse_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k = 5, name = 'sparse_top_k')
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [sparse_top_k])
    
    # callbacks
    cas = CosineAnnealingScheduler(epochs, lr_max, lr_min)
    ckp = callbacks.ModelCheckpoint(checkpoint_path, 
                                    monitor = 'val_sparse_top_k',
                                    verbose = 1, 
                                    save_weights_only = True,
                                    mode = 'max')

    history = model.fit(X_train, y_train, 
                        validation_data = (X_test, y_test), 
                        callbacks = [
                            ckp,
                            cas
                            ], 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        verbose = 1)

    hist = pd.DataFrame(history.history)
    model.save_weights(checkpoint_path.format(epoch=0))
    
    return  hist

if __name__ == "__main__":
    from helpers import fetch_dataset, train_test_split
    from evaluate import evaluate
    
    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)
    checkpoint_path="training_0/cp-{epoch:04d}.ckpt"
    hist = train(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, lr_max=1e-4, lr_min=1e-6, checkpoint_path=checkpoint_path)
    evaluate(checkpoint_path, X_test, y_test)