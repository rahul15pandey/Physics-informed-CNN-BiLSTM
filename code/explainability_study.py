"""
Explainability Study for Physics-Informed Dual-Pooling SE-FCN (PI-DP-SE-FCN).

Generates publication-quality figures for model interpretability:
  1. Grad-CAM heatmaps (convolutional layer spatial attention)
  2. Integrated Gradients saliency maps (per time-step × sensor attribution)
  3. Aggregated sensor importance ranking (bar chart)
  4. Prediction error distribution analysis by RUL range
  5. Physics-consistency analysis (monotonicity & slope tracking)
  6. SE attention weight visualisation (channel importance per block)
  7. Prediction scatter plots (true vs. predicted RUL)

Usage:
    python explainability_study.py --fd 1 --model_path ../model/Physics/FD1/model.h5
    python explainability_study.py --fd all   # runs all FDs with auto-discovered models
"""

import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.saving import register_keras_serializable
from scipy.signal import lfilter, lfilter_zi
import CMAPSSDataset

# ---------- GPU memory growth ----------
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

# =========================================================================
# Plot style
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

COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]


# =========================================================================
# Loss & helpers (for model loading)
# =========================================================================

def _diff_mask(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    mask = K.cast(true_diffs < 0.0, K.floatx())
    return true_diffs, pred_diffs, mask


@register_keras_serializable()
def physics_loss(y_true, y_pred):
    """Generic physics loss for loading saved models."""
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    mse = K.mean(K.square(y_true_flat - y_pred_flat))
    over_est = K.relu(y_pred_flat - y_true_flat)
    asym = K.mean(K.square(over_est))
    true_diffs, pred_diffs, mask = _diff_mask(y_true_flat, y_pred_flat)
    mono = K.sum(K.relu(pred_diffs * mask)) / (K.sum(mask) + K.epsilon())
    slope = K.sum(K.square((pred_diffs - true_diffs) * mask)) / (
        K.sum(mask) + K.epsilon()
    )
    return mse + 0.05 * asym + 0.001 * mono + 0.0003 * slope


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def se_block(x, ratio=8):
    """SE block stub for model loading (not called, just for custom_objects)."""
    pass


# =========================================================================
# EMA Smoothing
# =========================================================================

def apply_exponential_smoothing(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = lfilter_zi(b, a)
    smoothed = np.empty_like(data)
    for j in range(data.shape[2]):
        x_2d = data[:, :, j].astype(np.float64)
        zi_2d = zi * x_2d[:, 0:1]
        smoothed[:, :, j], _ = lfilter(b, a, x_2d, axis=1, zi=zi_2d)
    return smoothed.astype(np.float32)


# =========================================================================
# Dataset configs
# =========================================================================

def get_fd_config(fd: str) -> Tuple[int, List[str], float]:
    """Return (sequence_length, feature_columns, smoothing_alpha)."""
    if fd == "1":
        return 31, ["s2","s3","s4","s6","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"], 0.3
    elif fd == "2":
        return 21, ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"], 0.3
    elif fd == "3":
        return 38, ["s2","s3","s4","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s17","s20","s21"], 0.15
    elif fd == "4":
        return 19, ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s20","s21"], 0.15
    else:
        raise ValueError(f"Unsupported FD: {fd}")


def load_data(fd: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test data for a given FD dataset. Returns (X_test, Y_test, feature_names)."""
    seq_len, features, smooth_alpha = get_fd_config(fd)
    dataset = CMAPSSDataset.CMAPSSDataset(
        fd_number=fd,
        batch_size=1024,
        sequence_length=seq_len,
        deleted_engine=[1000],
        feature_columns=features,
    )
    test_data = dataset.get_test_data()
    X_test_raw, Y_test = dataset.get_last_data_slice(test_data)
    Y_test[Y_test > 115] = 115

    X_test_raw = apply_exponential_smoothing(X_test_raw, alpha=smooth_alpha)
    X_test = np.reshape(X_test_raw, (-1, X_test_raw.shape[1], 1, X_test_raw.shape[2]))
    return X_test, Y_test, features


def load_model_safe(model_path: str) -> keras.Model:
    """Load a saved model with all custom objects."""
    custom = {
        "physics_loss": physics_loss,
        "root_mean_squared_error": root_mean_squared_error,
    }
    try:
        model = keras.models.load_model(model_path, custom_objects=custom, compile=False)
    except Exception:
        model = keras.models.load_model(model_path, compile=False)
    model.compile(loss=physics_loss, metrics=[root_mean_squared_error])
    return model


# =========================================================================
# 1. Grad-CAM
# =========================================================================

def get_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last Conv2D layer in the model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model")
    return last_conv


def grad_cam(model: keras.Model, x: np.ndarray, layer_name: str = None) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single sample.

    Args:
        model: trained Keras model
        x: single sample, shape (1, T, 1, F)
        layer_name: target conv layer (default: last Conv2D)

    Returns:
        heatmap: shape (T, F) normalised to [0, 1]
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x, training=False)
        pred = predictions[:, 0]

    grads = tape.gradient(pred, conv_output)
    # Global average pooling of gradients → channel importance weights
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)

    # Weighted sum of feature maps
    cam = tf.reduce_sum(conv_output * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam)  # only positive contributions
    cam = cam[0].numpy()  # (T, 1) or (T, W)

    # Normalise to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def plot_grad_cam_heatmap(cam: np.ndarray, feature_names: List[str],
                          out_path: str, title: str):
    """Plot Grad-CAM activation map."""
    fig, ax = plt.subplots(figsize=(3, 5))
    im = ax.imshow(cam, aspect="auto", cmap="jet", interpolation="nearest")
    ax.set_ylabel("Time step")
    ax.set_xlabel("Spatial dim")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attention intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved Grad-CAM: {out_path}")


# =========================================================================
# 2. Integrated Gradients
# =========================================================================

def integrated_gradients(model: keras.Model, x: np.ndarray,
                         baseline: np.ndarray = None, steps: int = 50) -> np.ndarray:
    """Compute Integrated Gradients for a single sample.

    Args:
        x: shape (1, T, 1, F)
        baseline: zero baseline by default
        steps: interpolation steps (higher = more accurate)

    Returns:
        ig_map: shape (T, F) — attribution per time-step × sensor
    """
    if baseline is None:
        baseline = np.zeros_like(x)

    x_tf = tf.constant(x, dtype=tf.float32)
    baseline_tf = tf.constant(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    # Build interpolated inputs: (steps+1, T, 1, F)
    interp = baseline_tf + alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * (x_tf - baseline_tf)

    # Batch gradient computation
    with tf.GradientTape() as tape:
        tape.watch(interp)
        preds = model(interp, training=False)
    grads = tape.gradient(preds, interp)  # (steps+1, T, 1, F)

    # Trapezoidal approximation of integral
    avg_grads = tf.reduce_mean(grads, axis=0)  # (T, 1, F)

    # Attribution = (x - baseline) * avg_grads
    ig = (x_tf - baseline_tf) * avg_grads
    # Sum over channel dim (dim=2 which is 1), take absolute value
    ig_map = tf.reduce_sum(tf.abs(ig), axis=2)[0].numpy()  # (T, F)
    return ig_map


def plot_ig_heatmap(ig_map: np.ndarray, feature_names: List[str],
                    out_path: str, title: str):
    """Sensor × Time-step Integrated Gradients heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(feature_names) * 0.5), 5))
    im = ax.imshow(ig_map, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time step in window")
    ax.set_xlabel("Sensor")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|Integrated Gradients|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved IG heatmap: {out_path}")


# =========================================================================
# 3. Aggregated Sensor Importance (averaged over test set)
# =========================================================================

def compute_sensor_importance(model: keras.Model, X_test: np.ndarray,
                              n_samples: int = 50, steps: int = 30) -> np.ndarray:
    """Compute mean |IG| per sensor across n_samples test samples."""
    n = min(n_samples, X_test.shape[0])
    indices = np.linspace(0, X_test.shape[0] - 1, n, dtype=int)
    all_ig = []
    for idx in indices:
        sample = X_test[idx:idx+1]
        ig_map = integrated_gradients(model, sample, steps=steps)
        # Mean across time → per-sensor importance
        all_ig.append(ig_map.mean(axis=0))
    importance = np.mean(all_ig, axis=0)
    return importance


def plot_sensor_importance(importance: np.ndarray, feature_names: List[str],
                           out_path: str, title: str):
    """Horizontal bar chart of sensor importance."""
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.3)))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance[order], color=COLORS[0], edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel("Mean |Integrated Gradients|")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved sensor importance: {out_path}")


# =========================================================================
# 4. Prediction Error Distribution by RUL Range
# =========================================================================

def plot_error_distribution(Y_test: np.ndarray, Y_pred: np.ndarray,
                            out_path: str, title: str):
    """Box plots of prediction error binned by true RUL range."""
    errors = (Y_pred.flatten() - Y_test.flatten())
    true_rul = Y_test.flatten()

    # Define RUL bins
    bins = [(0, 20, "0-20"), (20, 50, "20-50"), (50, 80, "50-80"), (80, 115, "80-115")]
    box_data = []
    labels = []
    for lo, hi, label in bins:
        mask = (true_rul >= lo) & (true_rul < hi)
        if mask.sum() > 0:
            box_data.append(errors[mask])
            labels.append(f"{label}\n(n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Zero error")
    ax.set_xlabel("True RUL Range")
    ax.set_ylabel("Prediction Error (Pred - True)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved error distribution: {out_path}")


# =========================================================================
# 5. Physics Consistency Analysis
# =========================================================================

def plot_physics_consistency(Y_test: np.ndarray, Y_pred: np.ndarray,
                             out_path: str, title: str):
    """Analyse monotonicity and slope consistency of predictions.

    Shows how well the model preserves the physics constraint that RUL
    should decrease monotonically over time.
    """
    y_true = Y_test.flatten()
    y_pred = Y_pred.flatten()

    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    mask = true_diffs < 0  # same-engine transitions

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Monotonicity: histogram of pred_diffs where true decreases
    ax = axes[0]
    if mask.sum() > 0:
        mono_diffs = pred_diffs[mask]
        violations = (mono_diffs > 0).sum()
        total = mask.sum()
        ax.hist(mono_diffs, bins=50, color=COLORS[0], edgecolor="black",
                linewidth=0.5, alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted RUL difference (should be < 0)")
        ax.set_ylabel("Count")
        ax.set_title(f"Monotonicity\n({violations}/{total} violations = {100*violations/total:.1f}%)")

    # (b) Slope tracking: scatter of pred slope vs true slope
    ax = axes[1]
    if mask.sum() > 0:
        ax.scatter(true_diffs[mask], pred_diffs[mask], alpha=0.3, s=8, c=COLORS[2])
        lims = [min(true_diffs[mask].min(), pred_diffs[mask].min()),
                max(true_diffs[mask].max(), pred_diffs[mask].max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect tracking")
        ax.set_xlabel("True RUL slope")
        ax.set_ylabel("Predicted RUL slope")
        ax.set_title("Slope Tracking")
        ax.legend()

    # (c) Cumulative monotonicity violation rate by position
    ax = axes[2]
    if mask.sum() > 0:
        # Bin by true RUL value
        rul_vals = y_true[1:][mask]
        viol = (mono_diffs > 0).astype(float)
        bins_rul = np.linspace(0, 115, 12)
        bin_indices = np.digitize(rul_vals, bins_rul) - 1
        bin_rates = []
        bin_labels = []
        for b in range(len(bins_rul) - 1):
            bmask = bin_indices == b
            if bmask.sum() > 0:
                bin_rates.append(viol[bmask].mean() * 100)
                bin_labels.append(f"{bins_rul[b]:.0f}-{bins_rul[b+1]:.0f}")
        if bin_rates:
            ax.bar(range(len(bin_rates)), bin_rates, color=COLORS[3],
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(bin_rates)))
            ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("True RUL Range")
            ax.set_ylabel("Violation Rate (%)")
            ax.set_title("Monotonicity Violations by RUL")

    fig.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved physics consistency: {out_path}")


# =========================================================================
# 6. SE Attention Weight Visualisation
# =========================================================================

def extract_se_weights(model: keras.Model) -> List[Tuple[str, np.ndarray]]:
    """Extract learned SE squeeze-excitation weights from the model.

    Returns list of (block_name, channel_importance) tuples.
    """
    se_info = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense) and "sigmoid" in str(getattr(layer, "activation", "")):
            # Check if this is the SE excitation Dense layer
            w = layer.get_weights()
            if len(w) >= 1:
                # The bias gives the per-channel excitation offset
                # But we want the actual SE output for a representative input
                pass

    # Alternative: find Multiply layers (SE output) and get their effective weights
    # by passing a unit input through the model
    # Let's instead extract from the sigmoid Dense layers directly
    se_blocks = []
    block_idx = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Dense):
            # Check if activation is sigmoid (SE excitation layer)
            act_config = layer.get_config().get("activation", "")
            if act_config == "sigmoid" or (isinstance(act_config, dict) and act_config.get("class_name") == "sigmoid"):
                weights, biases = layer.get_weights()
                # Channel importance ≈ sigmoid(bias) — when input ≈ 0
                importance = 1.0 / (1.0 + np.exp(-biases))
                se_blocks.append((f"SE Block {block_idx + 1}", importance))
                block_idx += 1

    return se_blocks


def plot_se_attention(se_blocks: List[Tuple[str, np.ndarray]],
                      out_path: str, title: str):
    """Visualise SE block channel importance weights."""
    if not se_blocks:
        print("  No SE blocks found in model, skipping SE visualisation.")
        return

    n_blocks = len(se_blocks)
    fig, axes = plt.subplots(1, n_blocks, figsize=(4 * n_blocks, 4), squeeze=False)
    axes = axes.flatten()

    for i, (name, importance) in enumerate(se_blocks):
        ax = axes[i]
        n_channels = len(importance)
        ax.barh(range(n_channels), importance, color=COLORS[i % len(COLORS)],
                edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Channel weight (sigmoid)")
        ax.set_ylabel("Channel index")
        ax.set_title(f"{name} ({n_channels}ch)")
        ax.set_xlim(0, 1)

    fig.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved SE attention: {out_path}")


# =========================================================================
# 7. Prediction Scatter Plot
# =========================================================================

def plot_prediction_scatter(Y_test: np.ndarray, Y_pred: np.ndarray,
                            out_path: str, title: str):
    """True vs Predicted RUL scatter with ideal line and metrics."""
    y_true = Y_test.flatten()
    y_pred = Y_pred.flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # NASA score
    diff = y_pred - y_true
    over = np.exp(diff / 10.0) - 1.0
    under = np.exp(-diff / 13.0) - 1.0
    score = float(np.where(diff > 0, over, under).sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, c=COLORS[0], edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y=x)")
    ax.fill_between(
        np.linspace(lims[0], lims[1], 100),
        np.linspace(lims[0], lims[1], 100) - 10,
        np.linspace(lims[0], lims[1], 100) + 10,
        alpha=0.1, color="green", label="$\\pm$10 band"
    )
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper left")

    # Metrics text box
    textstr = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nNASA Score = {score:.1f}"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved scatter: {out_path}")


# =========================================================================
# 8. Combined Summary Dashboard
# =========================================================================

def generate_summary_dashboard(
    model: keras.Model, X_test: np.ndarray, Y_test: np.ndarray,
    Y_pred: np.ndarray, feature_names: List[str],
    out_path: str, fd: str
):
    """Generate a combined 2x3 dashboard figure for a single FD."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- (0,0) Prediction Scatter ---
    ax = fig.add_subplot(gs[0, 0])
    y_true = Y_test.flatten()
    y_pred = Y_pred.flatten()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    diff = y_pred - y_true
    over = np.exp(diff / 10.0) - 1.0
    under = np.exp(-diff / 13.0) - 1.0
    score = float(np.where(diff > 0, over, under).sum())
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, c=COLORS[0], edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predictions (RMSE={rmse:.2f}, Score={score:.0f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # --- (0,1) Error Distribution ---
    ax = fig.add_subplot(gs[0, 1])
    errors = y_pred - y_true
    bins_def = [(0, 20, "0-20"), (20, 50, "20-50"), (50, 80, "50-80"), (80, 115, "80-115")]
    box_data = []
    labels = []
    for lo, hi, label in bins_def:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            box_data.append(errors[mask])
            labels.append(f"{label}")
    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("True RUL Range")
    ax.set_ylabel("Error (Pred - True)")
    ax.set_title("Error Distribution by RUL")

    # --- (0,2) Sensor Importance ---
    ax = fig.add_subplot(gs[0, 2])
    print(f"  Computing sensor importance (this may take a moment)...")
    importance = compute_sensor_importance(model, X_test, n_samples=30, steps=20)
    order = np.argsort(importance)
    ax.barh(range(len(feature_names)), importance[order],
            color=COLORS[0], edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Mean |IG|")
    ax.set_title("Sensor Importance")

    # --- (1,0) Monotonicity Analysis ---
    ax = fig.add_subplot(gs[1, 0])
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    mono_mask = true_diffs < 0
    if mono_mask.sum() > 0:
        mono_diffs = pred_diffs[mono_mask]
        violations = (mono_diffs > 0).sum()
        total = mono_mask.sum()
        ax.hist(mono_diffs, bins=40, color=COLORS[2], edgecolor="black",
                linewidth=0.3, alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Pred RUL diff")
        ax.set_title(f"Monotonicity ({100*violations/total:.1f}% violations)")

    # --- (1,1) IG Heatmap for a representative sample ---
    ax = fig.add_subplot(gs[1, 1])
    mid_idx = len(X_test) // 2
    ig_map = integrated_gradients(model, X_test[mid_idx:mid_idx+1], steps=30)
    im = ax.imshow(ig_map, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Time step")
    ax.set_title(f"IG Attribution (sample {mid_idx})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- (1,2) SE Channel Weights ---
    ax = fig.add_subplot(gs[1, 2])
    se_blocks = extract_se_weights(model)
    if se_blocks:
        # Show last SE block's weights
        name, weights = se_blocks[-1]
        n_ch = len(weights)
        ax.bar(range(n_ch), weights, color=COLORS[4], edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Channel index")
        ax.set_ylabel("SE weight (sigmoid)")
        ax.set_title(f"{name} Channel Attention")
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No SE blocks found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("SE Attention (N/A)")

    fig.suptitle(f"Explainability Dashboard - FD00{fd}", fontweight="bold", fontsize=16, y=1.01)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved dashboard: {out_path}")


# =========================================================================
# Auto-discover model paths
# =========================================================================

def find_model_path(fd: str, project_root: str) -> Optional[str]:
    """Try to find a trained model file for the given FD."""
    search_patterns = [
        os.path.join(project_root, "saved_models", f"FD00{fd}_model.h5"),
        os.path.join(project_root, "model", "Physics", f"FD{fd}", "*.h5"),
        os.path.join(project_root, "experiments_result", "ablation_models", "full_proposed", f"*fd{fd}.h5"),
    ]
    for pattern in search_patterns:
        if "*" in pattern:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        elif os.path.exists(pattern):
            return pattern
    return None


# =========================================================================
# Main
# =========================================================================

def run_explainability(fd: str, model_path: str, out_dir: str,
                       n_ig_samples: int = 5):
    """Run complete explainability study for one FD dataset."""
    print(f"\n{'='*60}")
    print(f"  EXPLAINABILITY STUDY - FD00{fd}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    fd_dir = os.path.join(out_dir, f"FD{fd}")
    os.makedirs(fd_dir, exist_ok=True)

    # Load data and model
    X_test, Y_test, feature_names = load_data(fd)
    model = load_model_safe(model_path)
    model.summary()

    # Predictions
    Y_pred = model.predict(X_test, verbose=0)

    # 1. Prediction scatter
    plot_prediction_scatter(
        Y_test, Y_pred,
        os.path.join(fd_dir, f"scatter_FD{fd}.png"),
        f"FD00{fd}: True vs Predicted RUL"
    )

    # 2. Error distribution
    plot_error_distribution(
        Y_test, Y_pred,
        os.path.join(fd_dir, f"error_dist_FD{fd}.png"),
        f"FD00{fd}: Prediction Error by RUL Range"
    )

    # 3. Physics consistency
    plot_physics_consistency(
        Y_test, Y_pred,
        os.path.join(fd_dir, f"physics_consistency_FD{fd}.png"),
        f"FD00{fd}: Physics Consistency Analysis"
    )

    # 4. Integrated Gradients heatmaps (multiple samples)
    n_samples = min(n_ig_samples, X_test.shape[0])
    sample_indices = np.linspace(0, X_test.shape[0] - 1, n_samples, dtype=int)
    for idx in sample_indices:
        ig_map = integrated_gradients(model, X_test[idx:idx+1], steps=50)
        plot_ig_heatmap(
            ig_map, feature_names,
            os.path.join(fd_dir, f"ig_FD{fd}_sample{idx}.png"),
            f"FD00{fd} sample {idx} (true RUL={Y_test[idx][0]:.0f})"
        )

    # 5. Grad-CAM
    for idx in sample_indices[:3]:
        cam = grad_cam(model, X_test[idx:idx+1])
        plot_grad_cam_heatmap(
            cam, feature_names,
            os.path.join(fd_dir, f"gradcam_FD{fd}_sample{idx}.png"),
            f"FD00{fd} Grad-CAM sample {idx}"
        )

    # 6. Aggregated sensor importance
    importance = compute_sensor_importance(model, X_test, n_samples=50, steps=30)
    plot_sensor_importance(
        importance, feature_names,
        os.path.join(fd_dir, f"sensor_importance_FD{fd}.png"),
        f"FD00{fd}: Sensor Importance (Integrated Gradients)"
    )

    # Save raw importance values to CSV
    imp_df = {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}
    import pandas as pd
    pd.DataFrame([imp_df]).T.rename(columns={0: "importance"}).sort_values(
        "importance", ascending=False
    ).to_csv(os.path.join(fd_dir, f"sensor_importance_FD{fd}.csv"))

    # 7. SE attention weights
    se_blocks = extract_se_weights(model)
    plot_se_attention(
        se_blocks,
        os.path.join(fd_dir, f"se_attention_FD{fd}.png"),
        f"FD00{fd}: SE Channel Attention Weights"
    )

    # 8. Combined dashboard
    generate_summary_dashboard(
        model, X_test, Y_test, Y_pred, feature_names,
        os.path.join(fd_dir, f"dashboard_FD{fd}.png"), fd
    )

    # Print summary metrics
    rmse = np.sqrt(np.mean((Y_test.flatten() - Y_pred.flatten()) ** 2))
    diff = Y_pred.flatten() - Y_test.flatten()
    nasa_score = float(np.where(diff > 0,
                                np.exp(diff/10.0) - 1.0,
                                np.exp(-diff/13.0) - 1.0).sum())
    print(f"\n  FD00{fd} Results: RMSE={rmse:.4f}, NASA Score={nasa_score:.2f}")
    print(f"  All figures saved to: {fd_dir}")

    return {"fd": fd, "rmse": rmse, "score": nasa_score}


def main():
    parser = argparse.ArgumentParser(
        description="Explainability Study for PI-DP-SE-FCN RUL Model"
    )
    parser.add_argument("--fd", default="all",
                        help="FD sub-dataset: 1, 2, 3, 4, or 'all'")
    parser.add_argument("--model_path", default=None,
                        help="Path to trained .h5 model (auto-detected if omitted)")
    parser.add_argument("--n_ig_samples", type=int, default=5,
                        help="Number of samples for IG heatmaps")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: ../figure/explainability)")
    args = parser.parse_args()

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(_script_dir, ".."))

    if args.out_dir is None:
        args.out_dir = os.path.join(project_root, "figure", "explainability")
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine which FDs to run
    if args.fd.lower() == "all":
        fd_list = ["1", "2", "3", "4"]
    else:
        fd_list = [args.fd]

    all_results = []
    for fd in fd_list:
        if args.model_path and len(fd_list) == 1:
            model_path = args.model_path
        else:
            model_path = find_model_path(fd, project_root)
            if model_path is None:
                print(f"\n  WARNING: No model found for FD00{fd}, skipping.")
                print(f"  Searched in: saved_models/, model/Physics/FD{fd}/, ablation_models/")
                continue

        result = run_explainability(fd, model_path, args.out_dir, args.n_ig_samples)
        all_results.append(result)

    # Cross-FD summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            print(f"  FD00{r['fd']}: RMSE={r['rmse']:.4f}  Score={r['score']:.2f}")

        import pandas as pd
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.out_dir, "explainability_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary CSV: {summary_path}")

    print("\nExplainability study complete.")


if __name__ == "__main__":
    main()
