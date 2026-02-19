"""
Ablation study for Physics-Informed Dual-Pooling FCN (PI-DP-FCN).

Systematically evaluates the contribution of each loss component:
  1. MSE only (baseline)
  2. MSE + Asymmetric penalty
  3. MSE + Monotonicity
  4. MSE + Monotonicity + Slope
  5. MSE + Asymmetric + Monotonicity + Slope
  6. Full proposed (MSE + Asym + Mono + Slope + Score)  [FD003/004 only]

Generates publication-quality plots including:
  - Grouped bar charts comparing all metrics side by side
  - Radar / spider chart for multi-metric overview
  - Loss convergence curves (train + val) with log scale option
  - Prediction-vs-true scatter for each configuration
  - LaTeX-ready results table
"""

import argparse
import datetime
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
from scipy.signal import lfilter, lfilter_zi
import CMAPSSDataset

# ---------- GPU memory growth ----------
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

# =========================================================================
# Global plot style (publication quality)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colour palette (colour-blind friendly, from seaborn "colorblind")
COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]
HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]

# =========================================================================
# Metrics and helpers
# =========================================================================

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse_np(predictions, targets):
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def nasa_score_np(Y_test, Y_pred):
    """Compute NASA asymmetric scoring metric (scalar)."""
    s = 0.0
    yt = Y_test.flatten()
    yp = Y_pred.flatten()
    for i in range(len(yp)):
        if yp[i] > yt[i]:
            s += math.exp((yp[i] - yt[i]) / 10.0) - 1.0
        else:
            s += math.exp((yt[i] - yp[i]) / 13.0) - 1.0
    return float(s)


def physics_metrics_np(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    true_diffs = y_true_flat[1:] - y_true_flat[:-1]
    pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]
    mask = (true_diffs < 0).astype(np.float32)
    denom = np.sum(mask) + 1e-8
    mono_violation = np.maximum(pred_diffs, 0) * mask
    mono_mean = float(mono_violation.sum() / denom)
    slope_sqerr = ((pred_diffs - true_diffs) ** 2) * mask
    slope_rmse = float(math.sqrt(slope_sqerr.sum() / denom))
    return mono_mean, slope_rmse


def error_range_1out(Y_test, Y_pred):
    return float((Y_test - Y_pred).min()), float((Y_test - Y_pred).max())


# =========================================================================
# EMA Smoothing
# =========================================================================

def apply_exponential_smoothing(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Vectorised EMA along the time axis using scipy IIR filter.

    Matches the original formula: s[0]=x[0], s[t]=alpha*x[t]+(1-alpha)*s[t-1].
    Uses lfilter_zi to set proper initial conditions so y[0]=x[0].
    """
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = lfilter_zi(b, a)  # shape (1,)
    smoothed = np.empty_like(data)
    for j in range(data.shape[2]):
        x_2d = data[:, :, j].astype(np.float64)
        zi_2d = zi * x_2d[:, 0:1]  # (n_samples, 1)
        smoothed[:, :, j], _ = lfilter(b, a, x_2d, axis=1, zi=zi_2d)
    return smoothed.astype(np.float32)


# =========================================================================
# Loss factory (now supports asymmetric + score terms)
# =========================================================================

def make_physics_loss(
    alpha: float = 0.0,
    gamma: float = 0.0,
    asym_weight: float = 0.0,
    score_weight: float = 0.0,
    smooth_weight: float = 0.0,
):
    """Build a composite physics-informed loss with toggleable components."""

    def _diff_mask(y_true_flat, y_pred_flat):
        true_diffs = y_true_flat[1:] - y_true_flat[:-1]
        pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]
        same_engine_mask = K.cast(true_diffs < 0.0, K.floatx())
        return true_diffs, pred_diffs, same_engine_mask

    def loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        # (a) MSE
        mse = K.mean(K.square(y_true_flat - y_pred_flat))

        total = mse

        # (b) Asymmetric over-estimation penalty
        if asym_weight > 0:
            over_est = K.relu(y_pred_flat - y_true_flat)
            total = total + asym_weight * K.mean(K.square(over_est))

        # (c) Differentiable score proxy
        if score_weight > 0:
            error = y_pred_flat - y_true_flat
            pos = K.exp(error / 10.0) - 1.0
            neg = K.exp(-error / 13.0) - 1.0
            score_loss = K.mean(K.switch(error > 0, pos, neg))
            total = total + score_weight * score_loss

        # (d) Monotonicity + Slope
        if alpha > 0 or gamma > 0 or smooth_weight > 0:
            true_diffs, pred_diffs, mask = _diff_mask(y_true_flat, y_pred_flat)
            masked_pred_diffs = pred_diffs * mask
            if alpha > 0:
                mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(mask) + K.epsilon())
                total = total + alpha * mono_penalty
            if gamma > 0:
                slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * mask)) / (
                    K.sum(mask) + K.epsilon()
                )
                total = total + gamma * slope_penalty
            if smooth_weight > 0:
                ddiffs = pred_diffs[1:] - pred_diffs[:-1]
                total = total + smooth_weight * K.mean(K.square(ddiffs))

        return total

    return loss


# =========================================================================
# Squeeze-and-Excitation (SE) channel attention
# =========================================================================

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


# =========================================================================
# Model builder (Dual-Pooling architecture matching the main model)
# =========================================================================

def build_fcn(
    input_shape,
    num_filter1=32,
    num_filter2=64,
    num_filter3=128,
    k1=11,
    k2=9,
    k3=5,
    use_dual_pool=True,
    dropout_rate=0.5,
):
    """Build the PI-DP-FCN ablation model (matches Physics_olu.py architecture for FD001/002)."""
    in0 = keras.Input(shape=input_shape)

    # Block 1 + SE
    x = keras.layers.Conv2D(num_filter1, (k1, 1), strides=1, padding="same")(in0)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Block 2 + SE
    x = keras.layers.Conv2D(num_filter2, (k2, 1), strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Block 3 + SE
    x = keras.layers.Conv2D(num_filter3, (k3, 1), strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Pooling
    if use_dual_pool:
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        max_pool = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Concatenate()([avg_pool, max_pool])
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)

    # Head
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(1, activation="relu")(x)
    return keras.models.Model(inputs=in0, outputs=[out])


def build_fcn_deep(
    input_shape,
    dropout_rate=0.3,
):
    """Build 4-block deeper FCN for FD003/004 (matches PhysicsFD0034.py architecture).

    Architecture: Conv32(11) -> Conv64(9) -> Conv128(5) -> Conv256(3)
    with Dual Pooling and dropout head.
    """
    in0 = keras.Input(shape=input_shape)

    # Block 1 + SE
    x = keras.layers.Conv2D(32, (11, 1), padding="same")(in0)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Block 2 + SE
    x = keras.layers.Conv2D(64, (9, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Block 3 + SE
    x = keras.layers.Conv2D(128, (5, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Block 4 (extra depth for multi-condition FD003/004) + SE
    x = keras.layers.Conv2D(256, (3, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)

    # Dual Pooling
    avg_pool = keras.layers.GlobalAveragePooling2D()(x)
    max_pool = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([avg_pool, max_pool])

    # Head
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(1, activation="relu")(x)
    return keras.models.Model(inputs=in0, outputs=[out])


# =========================================================================
# Data loader
# =========================================================================

def load_data(
    fd: str, num_test: int, batch_size: int, smoothing_alpha: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fd == "1":
        sequence_length = 31
        fd_features = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    elif fd == "2":
        sequence_length = 21
        fd_features = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
    elif fd == "3":
        sequence_length = 38
        fd_features = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    elif fd == "4":
        sequence_length = 19
        fd_features = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s20", "s21"]
    else:
        raise ValueError("Unsupported FD")

    datasets = CMAPSSDataset.CMAPSSDataset(
        fd_number=fd,
        batch_size=batch_size,
        sequence_length=sequence_length,
        deleted_engine=[1000],
        feature_columns=fd_features,
    )
    train_data = datasets.get_train_data()
    train_feature_slice = datasets.get_feature_slice(train_data)
    train_label_slice = datasets.get_label_slice(train_data)
    test_data = datasets.get_test_data()
    if num_test == 100:
        test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
    else:
        test_feature_slice = datasets.get_feature_slice(test_data)
        test_label_slice = datasets.get_label_slice(test_data)

    # EMA smoothing
    print(f"  Applying EMA smoothing (alpha={smoothing_alpha})...")
    train_feature_slice = apply_exponential_smoothing(train_feature_slice, smoothing_alpha)
    test_feature_slice = apply_exponential_smoothing(test_feature_slice, smoothing_alpha)

    train_label_slice[train_label_slice > 115] = 115
    test_label_slice[test_label_slice > 115] = 115

    X_train = np.reshape(train_feature_slice, (-1, train_feature_slice.shape[1], 1, train_feature_slice.shape[2]))
    Y_train = train_label_slice
    X_test = np.reshape(test_feature_slice, (-1, test_feature_slice.shape[1], 1, test_feature_slice.shape[2]))
    Y_test = test_label_slice
    return X_train, Y_train, X_test, Y_test


# =========================================================================
# Training + evaluation for one config
# =========================================================================

def run_config(X_train, Y_train, X_test, Y_test, loss_fn, cfg: Dict, args,
               model_builder=None) -> Dict:
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Paths for caching ---
    model_dir = os.path.join(args.out_models, cfg["name"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_{cfg['name']}_fd{args.fd}.h5")
    weights_path = os.path.join(model_dir, f"weights_{cfg['name']}_fd{args.fd}.weights.h5")
    history_path = os.path.join(model_dir, f"history_{cfg['name']}_fd{args.fd}.json")
    preds_path = os.path.join(model_dir, f"preds_{cfg['name']}_fd{args.fd}.npz")

    # --- Build model (use custom builder if provided) ---
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    if model_builder is not None:
        model = model_builder(input_shape)
    else:
        model = build_fcn(
            input_shape=input_shape,
            num_filter1=args.num_filter1,
            num_filter2=args.num_filter2,
            num_filter3=args.num_filter3,
            k1=args.k1,
            k2=args.k2,
            k3=args.k3,
            dropout_rate=args.dropout,
        )
    # Cosine-decay LR + gradient clipping for stable convergence
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.epochs * steps_per_epoch,
        alpha=1e-5,
    )
    optimizer = keras.optimizers.Adam(learning_rate=cosine_lr, clipnorm=1.0)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[root_mean_squared_error])

    # --- Check if we can reuse a cached run ---
    cached = (
        not args.force_retrain
        and os.path.exists(model_path)
        and os.path.exists(history_path)
        and os.path.exists(preds_path)
    )

    if cached:
        # ---- LOAD from cache ----
        print(f"  [CACHE HIT] Loading saved model & results from {model_dir}")
        try:
            model = keras.models.load_model(model_path, compile=False)
            model.compile(loss=loss_fn, optimizer=optimizer, metrics=[root_mean_squared_error])
        except Exception:
            # Fallback: load weights into freshly-built model
            model.load_weights(weights_path)

        with open(history_path, "r") as f:
            history_dict = json.load(f)

        saved = np.load(preds_path)
        Y_pred = saved["Y_pred"]
        Y_test_saved = saved["Y_test"]

        rmse_val = rmse_np(Y_pred, Y_test_saved)
        score_val = nasa_score_np(Y_test_saved, Y_pred)
        er_left, er_right = error_range_1out(Y_test_saved, Y_pred)
        mono_mean, slope_rmse_val = physics_metrics_np(Y_test_saved, Y_pred)

        print(f"  Loaded — RMSE={rmse_val:.4f}  Score={score_val:.2f}")

        return {
            "name": cfg["name"],
            "display_name": cfg.get("display_name", cfg["name"]),
            "rmse": rmse_val,
            "score": score_val,
            "er_left": er_left,
            "er_right": er_right,
            "mono": mono_mean,
            "slope": slope_rmse_val,
            "history": history_dict,
            "model_path": model_path,
            "Y_pred": Y_pred,
            "Y_test": Y_test_saved,
        }

    # ---- TRAIN from scratch ----
    print(f"  [TRAINING] No cache found (or --force_retrain). Training...")

    # FD-specific training behavior
    use_shuffle = getattr(args, '_shuffle', False)
    use_callbacks = getattr(args, '_use_callbacks', True)

    cb = []
    if use_callbacks:
        cb = [
            keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=args.patience_reduce_lr, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="loss", patience=args.patience, verbose=1, restore_best_weights=True),
        ]

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=cb,
        shuffle=use_shuffle,
    )

    # ---- SAVE everything ----
    # 1) Full model + separate weights
    model.save(model_path)
    model.save_weights(weights_path)
    print(f"  Saved model  → {model_path}")
    print(f"  Saved weights → {weights_path}")

    # 2) Training history as JSON (convert numpy floats to Python floats)
    history_dict = {}
    for k, v in hist.history.items():
        history_dict[k] = [float(x) for x in v]
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Saved history → {history_path}")

    # 3) Predictions
    Y_pred = model.predict(X_test)
    np.savez_compressed(preds_path, Y_pred=Y_pred, Y_test=Y_test)
    print(f"  Saved preds   → {preds_path}")

    # ---- Compute metrics ----
    rmse_val = rmse_np(Y_pred, Y_test)
    score_val = nasa_score_np(Y_test, Y_pred)
    er_left, er_right = error_range_1out(Y_test, Y_pred)
    mono_mean, slope_rmse_val = physics_metrics_np(Y_test, Y_pred)

    return {
        "name": cfg["name"],
        "display_name": cfg.get("display_name", cfg["name"]),
        "rmse": rmse_val,
        "score": score_val,
        "er_left": er_left,
        "er_right": er_right,
        "mono": mono_mean,
        "slope": slope_rmse_val,
        "history": history_dict,
        "model_path": model_path,
        "Y_pred": Y_pred,
        "Y_test": Y_test,
    }


# =========================================================================
# Publication-quality plotting functions
# =========================================================================

def plot_grouped_bars(results: List[Dict], metrics: List[Tuple[str, str]], out_path: str, title: str):
    """Grouped bar chart comparing multiple metrics across all configs.
    
    Args:
        metrics: list of (metric_key, display_label) tuples
    """
    n_configs = len(results)
    n_metrics = len(metrics)
    x = np.arange(n_configs)
    width = 0.8 / n_metrics

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    axes = axes.flatten()

    for m_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[m_idx]
        vals = [r[metric_key] for r in results]
        labels = [r["display_name"] for r in results]
        bars = ax.bar(
            x, vals,
            width=0.6,
            color=[COLORS[i % len(COLORS)] for i in range(n_configs)],
            edgecolor="black",
            linewidth=0.5,
            hatch=[HATCHES[i % len(HATCHES)] for i in range(n_configs)],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric_label, fontweight="bold")
        ax.set_title(metric_label, fontweight="bold")

        # Value annotations on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar(results: List[Dict], metrics: List[Tuple[str, str]], out_path: str, title: str):
    """Radar / spider chart for multi-metric comparison (normalised to [0, 1])."""
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Collect raw values
    raw_vals = {m_key: [r[m_key] for r in results] for m_key, _ in metrics}

    for i, r in enumerate(results):
        values = []
        for m_key, _ in metrics:
            v = r[m_key]
            vmin = min(raw_vals[m_key])
            vmax = max(raw_vals[m_key])
            rng = vmax - vmin if vmax != vmin else 1.0
            # Invert: lower is better -> higher on radar
            values.append(1.0 - (v - vmin) / rng)
        values += values[:1]
        ax.plot(angles, values, "o-", label=r["display_name"], color=COLORS[i % len(COLORS)], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m_label for _, m_label in metrics], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curves(results: List[Dict], out_path: str, title: str, log_scale: bool = False):
    """Overlaid train + val loss curves with optional log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        epochs_arr = range(1, len(r["history"]["loss"]) + 1)
        ax1.plot(epochs_arr, r["history"]["loss"], color=c, linewidth=1.5, label=r["display_name"])
        if "val_loss" in r["history"]:
            ax2.plot(epochs_arr, r["history"]["val_loss"], color=c, linewidth=1.5, label=r["display_name"])

    for ax, lbl in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(lbl, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter_predictions(results: List[Dict], out_path: str, title: str):
    """Scatter plot: True RUL vs Predicted RUL for each ablation config."""
    n = len(results)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    for idx, r in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        yt = r["Y_test"].flatten()
        yp = r["Y_pred"].flatten()
        ax.scatter(yt, yp, s=18, alpha=0.6, c=COLORS[idx % len(COLORS)], edgecolor="none")
        lims = [0, max(yt.max(), yp.max()) + 5]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(f"{r['display_name']}\nRMSE={r['rmse']:.2f}  Score={r['score']:.0f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_improvement_waterfall(results: List[Dict], metric: str, metric_label: str, out_path: str, title: str):
    """Waterfall chart showing incremental improvement over baseline for one metric."""
    baseline_val = results[0][metric]
    names = [r["display_name"] for r in results]
    vals = [r[metric] for r in results]
    deltas = [0.0] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(8, 5))
    cumulative = baseline_val
    for i, (name, delta) in enumerate(zip(names, deltas)):
        if i == 0:
            ax.bar(name, baseline_val, color=COLORS[0], edgecolor="black", linewidth=0.5)
            ax.text(i, baseline_val, f"{baseline_val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            colour = "#029E73" if delta < 0 else "#D55E00"  # green if decrease (better)
            ax.bar(name, abs(delta), bottom=min(cumulative, cumulative - delta), color=colour, edgecolor="black", linewidth=0.5)
            ax.text(i, cumulative, f"{cumulative:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        cumulative = vals[i]

    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_latex_table(results: List[Dict], out_path: str, fd: str):
    """Write a LaTeX-ready table of ablation results."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Ablation study results on FD{fd}.}}",
        rf"\label{{tab:ablation_fd{fd}}}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{RMSE} $\downarrow$ & \textbf{Score} $\downarrow$ & \textbf{Mono} $\downarrow$ & \textbf{Slope} $\downarrow$ \\",
        r"\midrule",
    ]
    # Find best (min) for bolding
    best_rmse = min(r["rmse"] for r in results)
    best_score = min(r["score"] for r in results)
    best_mono = min(r["mono"] for r in results)
    best_slope = min(r["slope"] for r in results)

    for r in results:
        def _fmt(val, best):
            s = f"{val:.2f}"
            return rf"\textbf{{{s}}}" if abs(val - best) < 1e-6 else s

        line = (
            f"{r['display_name']} & "
            f"{_fmt(r['rmse'], best_rmse)} & "
            f"{_fmt(r['score'], best_score)} & "
            f"{_fmt(r['mono'], best_mono)} & "
            f"{_fmt(r['slope'], best_slope)} \\\\"
        )
        lines.append(line)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX table to {out_path}")


# =========================================================================
# Cross-dataset summary plots (for --fd all)
# =========================================================================

def plot_cross_fd_grouped(all_fd_results: Dict[str, List[Dict]], metric_key: str,
                          metric_label: str, out_path: str):
    """Grouped bar chart: configs (x-axis) grouped by FD dataset, for one metric."""
    fd_keys = sorted(all_fd_results.keys())
    # Use the config names from the first FD (they are the same across FDs)
    config_names = [r["display_name"] for r in all_fd_results[fd_keys[0]]]
    n_fds = len(fd_keys)
    n_configs = len(config_names)
    x = np.arange(n_configs)
    width = 0.8 / n_fds

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_configs), 5))
    for fd_idx, fd_key in enumerate(fd_keys):
        vals = [r[metric_key] for r in all_fd_results[fd_key]]
        offset = (fd_idx - (n_fds - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=f"FD00{fd_key}",
            color=COLORS[fd_idx % len(COLORS)],
            edgecolor="black", linewidth=0.4,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"Ablation: {metric_label} across all FD datasets", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_cross_fd_latex(all_fd_results: Dict[str, List[Dict]], out_path: str):
    """Write a single LaTeX table summarising Full Proposed across all FDs."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cross-dataset results for the Full Proposed model.}",
        r"\label{tab:cross_fd}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{RMSE} $\downarrow$ & \textbf{Score} $\downarrow$ & \textbf{Mono} $\downarrow$ & \textbf{Slope} $\downarrow$ \\",
        r"\midrule",
    ]
    for fd_key in sorted(all_fd_results.keys()):
        # Get the Full Proposed result (last config)
        full = all_fd_results[fd_key][-1]
        lines.append(
            f"FD00{fd_key} & {full['rmse']:.2f} & {full['score']:.2f} & "
            f"{full['mono']:.4f} & {full['slope']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved cross-FD LaTeX table to {out_path}")


# =========================================================================
# Per-FD ablation runner
# =========================================================================

def get_fd_defaults(fd: str):
    """Return dataset-specific defaults for batch_size, lr, dropout, smoothing, epochs.

    All datasets use epochs=200, shuffle=True.
    FD003/004: 4-block deep architecture, batch=256, lr=0.001, dropout=0.3,
               smoothing_alpha=0.1, no callbacks.
    FD001/002: 3-block architecture, batch=1024, lr=0.003, dropout=0.5,
               smoothing_alpha=0.3, with callbacks.
    """
    if fd in ("3", "4"):
        return {
            "batch_size": 256, "lr": 0.001, "dropout": 0.3,
            "smoothing_alpha": 0.1, "epochs": 200,
            "shuffle": True, "use_callbacks": False,
            "use_deep_model": True,
        }
    return {
        "batch_size": 1024, "lr": 0.003, "dropout": 0.5,
        "smoothing_alpha": 0.3, "epochs": 200,
        "shuffle": True, "use_callbacks": True,
        "use_deep_model": False,
    }


def get_configs(fd: str) -> List[Dict]:
    """Return the ablation configurations for a given FD dataset.

    Weight rationale:
    - asym kept LOW (0.05 FD1/2, 0.08 FD3/4) to avoid prediction bias
    - slope << mono so monotonicity dominates while slope refines
    - score proxy at very low weight (FD3/4 only) for NASA-score alignment
    - Full Proposed uses SAME base weights as sub-configs plus extra terms
    """
    if fd in ("1", "2"):
        configs = [
            {
                "name": "mse_only",
                "display_name": "MSE Only",
                "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_asym",
                "display_name": "MSE+Asym",
                "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.05, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_mono",
                "display_name": "MSE+Mono",
                "alpha": 0.001, "gamma": 0.0, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_mono_slope",
                "display_name": "MSE+Mono+Slope",
                "alpha": 0.001, "gamma": 0.0003, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "full_proposed",
                "display_name": "Full Proposed",
                "alpha": 0.001, "gamma": 0.0003, "asym_weight": 0.05, "score_weight": 0.0, "smooth_weight": 0.0,
            },
        ]
    else:
        # FD003/004: deeper model, extra score-proxy config
        configs = [
            {
                "name": "mse_only",
                "display_name": "MSE Only",
                "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_asym",
                "display_name": "MSE+Asym",
                "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.08, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_mono",
                "display_name": "MSE+Mono",
                "alpha": 0.002, "gamma": 0.0, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_mono_slope",
                "display_name": "MSE+Mono+Slope",
                "alpha": 0.002, "gamma": 0.0005, "asym_weight": 0.0, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "mse_asym_mono_slope",
                "display_name": "MSE+Asym+Mono+Slope",
                "alpha": 0.002, "gamma": 0.0005, "asym_weight": 0.08, "score_weight": 0.0, "smooth_weight": 0.0,
            },
            {
                "name": "full_proposed",
                "display_name": "Full Proposed",
                "alpha": 0.002, "gamma": 0.0005, "asym_weight": 0.08, "score_weight": 0.02, "smooth_weight": 0.0,
            },
        ]

    return configs


def run_ablation_for_fd(fd: str, args) -> List[Dict]:
    """Run the full ablation study for one FD dataset and save per-FD outputs."""
    # --- Apply FD-specific defaults (only override if the user didn't set them) ---
    defaults = get_fd_defaults(fd)
    effective_args = argparse.Namespace(**vars(args))
    effective_args.fd = fd
    if not args._user_set_batch_size:
        effective_args.batch_size = defaults["batch_size"]
    if not args._user_set_lr:
        effective_args.lr = defaults["lr"]
    if not args._user_set_dropout:
        effective_args.dropout = defaults["dropout"]
    if not args._user_set_smoothing:
        effective_args.smoothing_alpha = defaults["smoothing_alpha"]

    # FD-specific epochs (200 for FD003/004, 150 for FD001/002)
    if not args._user_set_epochs:
        effective_args.epochs = defaults["epochs"]

    # FD-specific training flags
    effective_args._shuffle = defaults.get("shuffle", False)
    effective_args._use_callbacks = defaults.get("use_callbacks", True)
    use_deep = defaults.get("use_deep_model", False)

    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY – FD00{fd}")
    print(f"  Epochs={effective_args.epochs}  Batch={effective_args.batch_size}  "
          f"LR={effective_args.lr}  Dropout={effective_args.dropout}  "
          f"Smooth={effective_args.smoothing_alpha}")
    print(f"  Shuffle={effective_args._shuffle}  Callbacks={effective_args._use_callbacks}  "
          f"DeepModel={use_deep}")
    print(f"{'='*60}\n")

    X_train, Y_train, X_test, Y_test = load_data(
        fd, effective_args.num_test, effective_args.batch_size,
        effective_args.smoothing_alpha,
    )

    configs = get_configs(fd)
    results = []
    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Config: {cfg['display_name']}")
        print(f"  alpha={cfg['alpha']}, gamma={cfg['gamma']}, asym={cfg['asym_weight']}, "
              f"score={cfg['score_weight']}, smooth={cfg['smooth_weight']}")
        print(f"{'─'*50}")

        loss_fn = make_physics_loss(
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            asym_weight=cfg["asym_weight"],
            score_weight=cfg["score_weight"],
            smooth_weight=cfg["smooth_weight"],
        )

        # Use the deep 4-block model for FD003/004 (matching PhysicsFD0034.py)
        builder = None
        if use_deep:
            builder = lambda shape: build_fcn_deep(shape, dropout_rate=effective_args.dropout)

        res = run_config(X_train, Y_train, X_test, Y_test, loss_fn, cfg, effective_args,
                         model_builder=builder)
        results.append(res)

    # =====================
    # Save CSV results
    # =====================
    df = pd.DataFrame([
        {
            "config": r["display_name"],
            "rmse": r["rmse"],
            "nasa_score": r["score"],
            "mono_violation": r["mono"],
            "slope_rmse": r["slope"],
            "error_range_left": r["er_left"],
            "error_range_right": r["er_right"],
            "model_path": r["model_path"],
        }
        for r in results
    ])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(effective_args.out_ablation_csv, f"ablation_fd{fd}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to {csv_path}")

    # =====================
    # Generate per-FD plots
    # =====================
    prefix = os.path.join(effective_args.out_ablation, f"fd{fd}")
    fd_label = f"FD00{fd}"
    metrics_list = [("rmse", "RMSE"), ("score", "NASA Score"), ("mono", "Mono Violation"), ("slope", "Slope RMSE")]

    plot_grouped_bars(
        results, metrics_list,
        f"{prefix}_grouped_bars.png",
        f"Ablation Study – {fd_label}: All Metrics"
    )
    plot_radar(
        results, metrics_list,
        f"{prefix}_radar.png",
        f"Ablation Study – {fd_label}: Radar Overview"
    )
    plot_loss_curves(
        results, f"{prefix}_loss_curves.png",
        f"Ablation Study – {fd_label}: Loss Convergence"
    )
    plot_loss_curves(
        results, f"{prefix}_loss_curves_log.png",
        f"Ablation Study – {fd_label}: Loss Convergence (Log Scale)",
        log_scale=True,
    )
    plot_scatter_predictions(
        results, f"{prefix}_scatter.png",
        f"Ablation Study – {fd_label}: True vs Predicted RUL"
    )
    plot_improvement_waterfall(
        results, "rmse", "RMSE",
        f"{prefix}_waterfall_rmse.png",
        f"RMSE Improvement Waterfall – {fd_label}"
    )
    plot_improvement_waterfall(
        results, "score", "NASA Score",
        f"{prefix}_waterfall_score.png",
        f"NASA Score Improvement Waterfall – {fd_label}"
    )
    generate_latex_table(results, f"{prefix}_ablation_table.tex", fd)

    print(f"\n  All per-FD plots saved for {fd_label}")

    # Print summary table
    print(f"\n{'─'*70}")
    print(f"{'Config':<25} {'RMSE':>8} {'Score':>10} {'Mono':>8} {'Slope':>8}")
    print(f"{'─'*70}")
    for r in results:
        print(f"{r['display_name']:<25} {r['rmse']:>8.2f} {r['score']:>10.2f} {r['mono']:>8.4f} {r['slope']:>8.4f}")
    print(f"{'─'*70}")

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation study for PI-DP-FCN RUL prediction")
    parser.add_argument("--fd", default="all",
                        help="FD sub-dataset: 1, 2, 3, 4, or 'all' to run all four")
    parser.add_argument("--num_test", type=int, default=100, choices=[100, 10000])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 150 for FD001/002, 200 for FD003/004)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: auto per FD)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto per FD)")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--patience_reduce_lr", type=int, default=15)
    parser.add_argument("--num_filter1", type=int, default=32)
    parser.add_argument("--num_filter2", type=int, default=64)
    parser.add_argument("--num_filter3", type=int, default=128)
    parser.add_argument("--k1", type=int, default=11)
    parser.add_argument("--k2", type=int, default=9)
    parser.add_argument("--k3", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (default: auto per FD)")
    parser.add_argument("--smoothing_alpha", type=float, default=None,
                        help="EMA smoothing alpha (default: auto per FD)")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Force retraining even if cached weights/results exist")
    parser.add_argument("--out_root", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "experiments_result")))
    args = parser.parse_args()

    # Track which args the user explicitly set vs auto-defaults
    args._user_set_batch_size = args.batch_size is not None
    args._user_set_lr = args.lr is not None
    args._user_set_dropout = args.dropout is not None
    args._user_set_smoothing = args.smoothing_alpha is not None
    args._user_set_epochs = args.epochs is not None

    # Fill in defaults so downstream code always has values
    if args.batch_size is None:
        args.batch_size = 1024
    if args.lr is None:
        args.lr = 0.003
    if args.dropout is None:
        args.dropout = 0.5
    if args.smoothing_alpha is None:
        args.smoothing_alpha = 0.3
    if args.epochs is None:
        args.epochs = 150

    # Setup output directories
    args.out_ablation = os.path.join(args.out_root, "ablation")
    args.out_ablation_csv = os.path.join(args.out_root, "ablation_csv")
    args.out_models = os.path.join(args.out_root, "ablation_models")
    for d in [args.out_ablation, args.out_ablation_csv, args.out_models]:
        os.makedirs(d, exist_ok=True)

    # ---- Determine which FDs to run ----
    if args.fd.lower() == "all":
        fd_list = ["1", "2", "3", "4"]
    else:
        fd_list = [args.fd]

    # ---- Run ablation for each FD ----
    all_fd_results: Dict[str, List[Dict]] = {}
    for fd in fd_list:
        results = run_ablation_for_fd(fd, args)
        all_fd_results[fd] = results

    # ---- Cross-dataset summary (when multiple FDs) ----
    if len(fd_list) > 1:
        print(f"\n{'#'*60}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"{'#'*60}")

        # 1) Cross-FD grouped bar charts
        # We need all FDs to have the same configs for grouped bars.
        # Use only configs present in ALL FDs (intersection by display_name).
        common_names = None
        for fd_key in sorted(all_fd_results.keys()):
            names = {r["display_name"] for r in all_fd_results[fd_key]}
            common_names = names if common_names is None else common_names & names

        # Filter results to common configs, preserving order
        filtered_fd_results: Dict[str, List[Dict]] = {}
        for fd_key in sorted(all_fd_results.keys()):
            filtered_fd_results[fd_key] = [
                r for r in all_fd_results[fd_key] if r["display_name"] in common_names
            ]

        cross_prefix = os.path.join(args.out_ablation, "cross_fd")
        for metric_key, metric_label in [("rmse", "RMSE"), ("score", "NASA Score"),
                                          ("mono", "Mono Violation"), ("slope", "Slope RMSE")]:
            plot_cross_fd_grouped(
                filtered_fd_results, metric_key, metric_label,
                f"{cross_prefix}_{metric_key}.png",
            )
        print("  Saved cross-FD grouped bar charts")

        # 2) Cross-FD LaTeX table (Full Proposed only)
        generate_cross_fd_latex(all_fd_results, f"{cross_prefix}_table.tex")

        # 3) Cross-FD CSV
        cross_rows = []
        for fd_key in sorted(all_fd_results.keys()):
            for r in all_fd_results[fd_key]:
                cross_rows.append({
                    "FD": f"FD00{fd_key}",
                    "config": r["display_name"],
                    "rmse": r["rmse"],
                    "nasa_score": r["score"],
                    "mono": r["mono"],
                    "slope": r["slope"],
                })
        cross_df = pd.DataFrame(cross_rows)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cross_csv_path = os.path.join(args.out_ablation_csv, f"ablation_all_fd_{ts}.csv")
        cross_df.to_csv(cross_csv_path, index=False)
        print(f"  Saved cross-FD CSV to {cross_csv_path}")

        # 4) Print grand summary
        print(f"\n{'─'*78}")
        print(f"{'FD':<8} {'Config':<25} {'RMSE':>8} {'Score':>10} {'Mono':>8} {'Slope':>8}")
        print(f"{'─'*78}")
        for fd_key in sorted(all_fd_results.keys()):
            full = all_fd_results[fd_key][-1]  # Full Proposed is always last
            print(f"FD00{fd_key:<4} {full['display_name']:<25} "
                  f"{full['rmse']:>8.2f} {full['score']:>10.2f} "
                  f"{full['mono']:>8.4f} {full['slope']:>8.2f}")
        print(f"{'─'*78}")

    print(f"\n{'='*60}")
    print(f"  ALL ABLATIONS COMPLETE")
    print(f"  Outputs: {args.out_ablation}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
