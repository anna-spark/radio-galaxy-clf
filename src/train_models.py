"""
train_models.py

Constructs and trains the SCNN from Brand et al. (2023) on preprocessed
galaxy datasets. Supports standard (unprocessed), rotationally standardized,
and augmented training modes.

Usage:
    python train_models.py          # trains on watershed data by default
    Set data_type at the bottom of the file to switch datasets:
      ''   → unprocessed
      'ws' → watershed
      'th' → binary thresholding
      'mt' → adaptive mean thresholding
      'gt' → adaptive Gaussian thresholding
      'aug'→ augmented
"""

import os
import numpy as np
from functools import partial
from tensorflow import keras
from sklearn.model_selection import train_test_split


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------
# Model architecture
# -----------------------------------------------------------------------

def construct_standard():
    """
    Build the SCNN from Brand et al. (2023).

    Three conv blocks (64 → 128 → 256 filters, 7x7 kernels) followed by
    two ELU dense layers with 0.5 dropout. Nadam optimizer, lr=1e-4.

    Returns
    -------
    keras.Sequential
        Compiled model ready for training.
    """
    base_conv = partial(keras.layers.Conv2D, kernel_size=7, activation='relu', padding='same')

    model = keras.models.Sequential()
    filters = 64
    model.add(base_conv(filters=filters, kernel_size=14, input_shape=[150, 150, 1]))
    for _ in range(2):
        filters <<= 1
        model.add(keras.layers.MaxPooling2D(2))
        model.add(base_conv(filters=filters))
        model.add(base_conv(filters=filters))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(150, activation='elu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(75, activation='elu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Nadam(learning_rate=0.0001),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    return model


# -----------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------

def _train(model_name, x_train, y_train, x_val, y_val, x_test, y_test, run, data_type):
    """
    Shared training loop. Saves best checkpoint, uses early stopping (patience=5).

    Returns (accuracy, loss) on the test set.
    """
    _ensure_dir('../../lr_logs/')
    _ensure_dir('../../models/')

    model = construct_standard()
    log_dir   = os.path.join(os.curdir, f'../../lr_logs/{model_name}_run{run}_{data_type}')
    ckpt_path = f'../../models/{model_name}_model{run}_{data_type}.h5'

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True),
        keras.callbacks.TensorBoard(log_dir),
    ]
    model.fit(
        x_train, y_train, epochs=100,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )
    best = keras.models.load_model(ckpt_path)
    return best.evaluate(x_test, y_test)


def train_standard(x_train, y_train, x_val, y_val, x_test, y_test, run, data_type='ws'):
    """Train on unprocessed or standardized images."""
    return _train('standard', x_train, y_train, x_val, y_val, x_test, y_test, run, data_type)


def train_derotated_standard(x_train, y_train, x_val, y_val, x_test, y_test, run, data_type='ws'):
    """Train on rotationally standardized (derotated) images."""
    return _train('derotated_standard', x_train, y_train, x_val, y_val, x_test, y_test, run, data_type)


def train_augmented_standard(x_train, y_train, x_val, y_val, x_test, y_test, run, data_type='ws'):
    """Train on rotationally augmented images (6x dataset size)."""
    return _train('aug_standard', x_train, y_train, x_val, y_val, x_test, y_test, run, data_type)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    data_type = 'ws'   # change to '', 'th', 'mt', 'gt', or 'aug'
    data_dir  = '../data'
    suffix    = f'_{data_type}' if data_type else ''

    print(f"\nLoading dataset: '{data_type or 'base'}'\n")

    X_train = np.load(f'{data_dir}/galaxy_X_train1{suffix}.npy')
    y_train = np.load(f'{data_dir}/galaxy_y_train{suffix}.npy')
    X_val   = np.load(f'{data_dir}/galaxy_X_val1{suffix}.npy')
    y_val   = np.load(f'{data_dir}/galaxy_y_val{suffix}.npy')
    X_test  = np.load(f'{data_dir}/galaxy_X_test1{suffix}.npy')
    y_test  = np.load(f'{data_dir}/galaxy_y_test{suffix}.npy')

    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, -1)
        X_val   = np.expand_dims(X_val,   -1)
        X_test  = np.expand_dims(X_test,  -1)

    acc, loss = train_standard(X_train, y_train, X_val, y_val, X_test, y_test,
                               run=1, data_type=data_type)
    print(f"\nTest accuracy: {acc:.4f}  |  loss: {loss:.4f}\n")
