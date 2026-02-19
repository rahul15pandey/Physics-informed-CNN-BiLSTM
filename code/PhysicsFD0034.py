# -*- coding: utf-8 -*-
"""
Optimized Physics-informed FCN for CMAPSS (FD003 & FD004)
Now with full logging and saving
"""

import os
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import CMAPSSDataset

# =============================
# Directory Setup
# =============================

# Base folder: ...\Physics-informed-CNN-BiLSTM
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

MODEL_DIR   = os.path.join(PROJECT_ROOT, "saved_models")
LOG_DIR     = os.path.join(PROJECT_ROOT, "training_logs")
PRED_DIR    = os.path.join(PROJECT_ROOT, "predictions")
FIG_DIR     = os.path.join(PROJECT_ROOT, "figures")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "summary")

for d in [MODEL_DIR, LOG_DIR, PRED_DIR, FIG_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================
# Loss Definition
# =============================

physics_alpha = 0.002
physics_gamma = 0.0005
late_prediction_weight = 0.08
score_weight = 0.02

def _diff_mask(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    mask = K.cast(true_diffs < 0.0, K.floatx())
    return true_diffs, pred_diffs, mask

@register_keras_serializable()
def physics_loss(y_true, y_pred):

    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    mse = K.mean(K.square(y_true_flat - y_pred_flat))

    over_est = K.relu(y_pred_flat - y_true_flat)
    asymmetric_loss = K.mean(K.square(over_est))

    error = y_pred_flat - y_true_flat
    pos = K.exp(error / 10.0) - 1.0
    neg = K.exp(-error / 13.0) - 1.0
    score_loss = K.mean(K.switch(error > 0, pos, neg))

    true_diffs, pred_diffs, mask = _diff_mask(y_true_flat, y_pred_flat)

    mono_penalty = K.sum(K.relu(pred_diffs * mask)) / (K.sum(mask) + K.epsilon())
    slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * mask)) / (
        K.sum(mask) + K.epsilon()
    )

    return (
        mse
        + late_prediction_weight * asymmetric_loss
        + score_weight * score_loss
        + physics_alpha * mono_penalty
        + physics_gamma * slope_penalty
    )

# =============================
# Smoothing
# =============================

def apply_exponential_smoothing(data, alpha=0.1):
    smoothed = np.copy(data)
    for i in range(smoothed.shape[0]):
        for j in range(smoothed.shape[2]):
            x = smoothed[i, :, j]
            s = np.zeros_like(x)
            s[0] = x[0]
            for t in range(1, len(x)):
                s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
            smoothed[i, :, j] = s
    return smoothed

# =============================
# Squeeze-and-Excitation Attention
# =============================

def se_block(x, ratio=8):
    """Squeeze-and-Excitation: learns per-channel importance weights.

    Adaptively recalibrates feature maps so the network focuses on the
    most informative degradation channels.  Ref: Hu et al., CVPR 2018.
    """
    filters = int(x.shape[-1])
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Dense(max(filters // ratio, 4), activation='relu',
                            kernel_initializer='he_normal')(se)
    se = keras.layers.Dense(filters, activation='sigmoid',
                            kernel_initializer='he_normal')(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.Multiply()([x, se])

# =============================
# Model
# =============================

def build_model(input_shape):

    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (11, 1), padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    x = keras.layers.Conv2D(64, (9, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    x = keras.layers.Conv2D(128, (5, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    x = keras.layers.Conv2D(256, (3, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    avg_pool = keras.layers.GlobalAveragePooling2D()(x)
    max_pool = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([avg_pool, max_pool])

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(1, activation="relu")(x)

    return keras.models.Model(inputs, outputs)

# =============================
# Hyperparameters
# =============================

batch_size = 256
epochs = 200
learning_rate = 0.001
smoothing_alpha = 0.1

# If True, load existing saved weights (if present) and skip training.
# Set to False to force retraining even if weights exist.
reuse_saved_weights = True

fd_summary = []

# =============================
# Training Loop
# =============================

for FD in ["3", "4"]:

    print(f"\n========== Training FD00{FD} ==========")

    if FD == "3":
        sequence_length = 38
        features = ["s2","s3","s4","s6","s7","s8","s9","s10",
                    "s11","s12","s13","s14","s15","s17","s20","s21"]
    else:
        sequence_length = 19
        features = ["s1","s2","s3","s4","s5","s6","s7","s8","s9",
                    "s10","s11","s12","s13","s14","s15","s16",
                    "s17","s18","s20","s21"]

    dataset = CMAPSSDataset.CMAPSSDataset(
        fd_number=FD,
        batch_size=batch_size,
        sequence_length=sequence_length,
        deleted_engine=[1000],
        feature_columns=features,
    )

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    X_train = dataset.get_feature_slice(train_data)
    Y_train = dataset.get_label_slice(train_data)

    X_test, Y_test = dataset.get_last_data_slice(test_data)

    Y_train[Y_train > 115] = 115
    Y_test[Y_test > 115] = 115

    X_train = apply_exponential_smoothing(X_train, smoothing_alpha)
    X_test = apply_exponential_smoothing(X_test, smoothing_alpha)

    X_train = np.reshape(X_train, (-1, X_train.shape[1], 1, X_train.shape[2]))
    X_test = np.reshape(X_test, (-1, X_test.shape[1], 1, X_test.shape[2]))

    # ================= MODEL BUILD / LOAD =================
    model = build_model(X_train.shape[1:])

    # Cosine-decay LR schedule + gradient clipping for stable convergence
    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps = epochs * steps_per_epoch
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=1e-5,
    )
    optimizer = keras.optimizers.Adam(learning_rate=cosine_lr, clipnorm=1.0)

    model.compile(
        loss=physics_loss,
        optimizer=optimizer,
        metrics=[keras.metrics.RootMeanSquaredError(name="RMSE")]
    )

    # Paths for saving model and weights
    model_path = os.path.join(MODEL_DIR, f"FD00{FD}_model.h5")
    weights_path = os.path.join(MODEL_DIR, f"FD00{FD}_model.h5")

    history = None

    if reuse_saved_weights and os.path.exists(weights_path):
        print(f"Found existing weights for FD00{FD}, loading from {weights_path} and skipping training.")
        model.load_weights(weights_path)
    else:
        print(f"No saved weights for FD00{FD} (or reuse_saved_weights=False). Starting training.")

        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            shuffle=True,
            verbose=1
        )

        # ================= SAVE MODEL & WEIGHTS =================
        model.save(model_path)
        model.save_weights(weights_path)

        # ================= SAVE LOG =================
        log_df = pd.DataFrame(history.history)
        log_path = os.path.join(LOG_DIR, f"FD00{FD}_training_log.csv")
        log_df.to_csv(log_path, index=False)

        # ================= SAVE LOSS PLOT =================
        plt.figure()
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title(f"FD00{FD} Loss Curve")
        fig_path = os.path.join(FIG_DIR, f"FD00{FD}_loss.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    # ================= SAVE PREDICTIONS =================
    Y_pred = model.predict(X_test)

    pred_df = pd.DataFrame({
        "True_RUL": Y_test.flatten(),
        "Predicted_RUL": Y_pred.flatten()
    })

    pred_path = os.path.join(PRED_DIR, f"FD00{FD}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # ================= METRICS =================
    rmse = np.sqrt(((Y_test - Y_pred) ** 2).mean())

    score = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] > Y_test[i]:
            score += math.exp((Y_pred[i] - Y_test[i]) / 10) - 1
        else:
            score += math.exp((Y_test[i] - Y_pred[i]) / 13) - 1

    print(f"FD00{FD} RMSE: {rmse:.4f}")
    print(f"FD00{FD} NASA Score: {score:.2f}")

    fd_summary.append({
        "FD": f"FD00{FD}",
        "RMSE": rmse,
        "NASA_Score": score
    })

# ================= SUMMARY =================

summary_df = pd.DataFrame(fd_summary)
summary_csv_path = os.path.join(SUMMARY_DIR, "FD003_FD004_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("\nSaved everything successfully.")
