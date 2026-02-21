"""
Generalizability Study for Physics-Informed Dual-Pooling FCN (PI-DP-FCN).

Tests the methodology on arbitrary time-series degradation / RUL datasets
beyond C-MAPSS, to evaluate whether the physics-informed loss components
(monotonicity, slope matching, asymmetric penalty, score proxy) provide
consistent benefits across different domains.

Usage examples:
  # 1) Quick test with built-in synthetic data
  python generalizability_study.py --mode synthetic

  # 2) Your own CSV dataset
  python generalizability_study.py --mode csv \
      --train_path data/train.csv --test_path data/test.csv \
      --unit_col turbine_id --cycle_col timestamp --target_col RUL

  # 3) NASA Bearing IMS dataset
  python generalizability_study.py --mode nasa_bearing --data_dir data/IMS

  # 4) Battery capacity fade
  python generalizability_study.py --mode battery --csv_path data/battery.csv

  # 5) PHM 2012 PRONOSTIA bearings
  python generalizability_study.py --mode phm2012 --data_dir data/PHM2012

  # 6) NumPy arrays
  python generalizability_study.py --mode numpy \
      --train_X data/X_train.npy --train_Y data/Y_train.npy \
      --test_X data/X_test.npy --test_Y data/Y_test.npy

  # 7) JSON config file (multiple datasets at once)
  python generalizability_study.py --config datasets_config.json

Generates the same publication-quality ablation plots as ablation_physics_fcn.py,
plus cross-dataset comparison charts.
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
from scipy.signal import lfilter, lfilter_zi

# Import the generic dataset loader (same directory)
import GenericTimeSeriesDataset as GDS

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
HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]


# =========================================================================
# Metrics
# =========================================================================

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse_np(predictions, targets):
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def mae_np(predictions, targets):
    return float(np.abs(predictions - targets).mean())


def nasa_score_np(Y_test, Y_pred):
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


def r_squared_np(y_true, y_pred):
    """Coefficient of determination (R^2)."""
    ss_res = np.sum((y_true.flatten() - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_true.flatten() - np.mean(y_true.flatten())) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)


# =========================================================================
# EMA smoothing
# =========================================================================

def apply_ema(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Vectorised EMA along time axis."""
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = lfilter(b, a, [1.0])  # initial condition
    from scipy.signal import lfilter_zi
    zi = lfilter_zi(b, a)
    smoothed = np.empty_like(data)
    for j in range(data.shape[2]):
        x_2d = data[:, :, j].astype(np.float64)
        zi_2d = zi * x_2d[:, 0:1]
        smoothed[:, :, j], _ = lfilter(b, a, x_2d, axis=1, zi=zi_2d)
    return smoothed.astype(np.float32)


# =========================================================================
# Physics-informed loss factory
# =========================================================================

def make_physics_loss(
    alpha: float = 0.0,
    gamma: float = 0.0,
    asym_weight: float = 0.0,
    score_weight: float = 0.0,
    smooth_weight: float = 0.0,
):
    def loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        mse = K.mean(K.square(y_true_flat - y_pred_flat))
        total = mse

        if asym_weight > 0:
            over_est = K.relu(y_pred_flat - y_true_flat)
            total = total + asym_weight * K.mean(K.square(over_est))

        if score_weight > 0:
            error = y_pred_flat - y_true_flat
            pos = K.exp(error / 10.0) - 1.0
            neg = K.exp(-error / 13.0) - 1.0
            score_loss = K.mean(K.switch(error > 0, pos, neg))
            total = total + score_weight * score_loss

        if alpha > 0 or gamma > 0 or smooth_weight > 0:
            true_diffs = y_true_flat[1:] - y_true_flat[:-1]
            pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]
            mask = K.cast(true_diffs < 0.0, K.floatx())
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
# Squeeze-and-Excitation block
# =========================================================================

def se_block(x, ratio=8):
    filters = int(x.shape[-1])
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Dense(max(filters // ratio, 4), activation='relu',
                            kernel_initializer='he_normal')(se)
    se = keras.layers.Dense(filters, activation='sigmoid',
                            kernel_initializer='he_normal')(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.Multiply()([x, se])


# =========================================================================
# Model builders (adaptive to input shape)
# =========================================================================

def build_fcn_3block(input_shape, dropout_rate=0.5):
    """3-block FCN with SE attention + Dual Pooling (for smaller datasets)."""
    in0 = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (11, 1), padding="same")(in0)
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

    avg_pool = keras.layers.GlobalAveragePooling2D()(x)
    max_pool = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Concatenate()([avg_pool, max_pool])

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(1, activation="relu")(x)
    return keras.models.Model(inputs=in0, outputs=[out])


def build_fcn_4block(input_shape, dropout_rate=0.3):
    """4-block deeper FCN with SE attention + Dual Pooling (for larger/complex datasets)."""
    in0 = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (11, 1), padding="same")(in0)
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
    x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(1, activation="relu")(x)
    return keras.models.Model(inputs=in0, outputs=[out])


def auto_select_model(input_shape, n_train_samples: int, dropout_rate: float = 0.4):
    """Automatically choose 3-block vs 4-block based on dataset size."""
    if n_train_samples > 15000 or input_shape[-1] > 16:
        print(f"  Auto-selected 4-block FCN (n_train={n_train_samples}, n_feat={input_shape[-1]})")
        return build_fcn_4block(input_shape, dropout_rate)
    else:
        print(f"  Auto-selected 3-block FCN (n_train={n_train_samples}, n_feat={input_shape[-1]})")
        return build_fcn_3block(input_shape, dropout_rate)


# =========================================================================
# Ablation configs
# =========================================================================

def get_ablation_configs(include_score: bool = True) -> List[Dict]:
    """Standard ablation configurations."""
    configs = [
        {
            "name": "mse_only",
            "display_name": "MSE Only",
            "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.0,
            "score_weight": 0.0, "smooth_weight": 0.0,
        },
        {
            "name": "mse_asym",
            "display_name": "MSE+Asym",
            "alpha": 0.0, "gamma": 0.0, "asym_weight": 0.05,
            "score_weight": 0.0, "smooth_weight": 0.0,
        },
        {
            "name": "mse_mono",
            "display_name": "MSE+Mono",
            "alpha": 0.001, "gamma": 0.0, "asym_weight": 0.0,
            "score_weight": 0.0, "smooth_weight": 0.0,
        },
        {
            "name": "mse_mono_slope",
            "display_name": "MSE+Mono+Slope",
            "alpha": 0.001, "gamma": 0.0003, "asym_weight": 0.0,
            "score_weight": 0.0, "smooth_weight": 0.0,
        },
        {
            "name": "mse_asym_mono_slope",
            "display_name": "MSE+Asym+Mono+Slope",
            "alpha": 0.001, "gamma": 0.0003, "asym_weight": 0.05,
            "score_weight": 0.0, "smooth_weight": 0.0,
        },
    ]
    if include_score:
        configs.append({
            "name": "full_proposed",
            "display_name": "Full Proposed",
            "alpha": 0.001, "gamma": 0.0003, "asym_weight": 0.05,
            "score_weight": 0.02, "smooth_weight": 0.0,
        })
    else:
        # Rename last as Full Proposed (without score)
        configs[-1]["name"] = "full_proposed"
        configs[-1]["display_name"] = "Full Proposed"

    return configs


# =========================================================================
# Training + evaluation
# =========================================================================

def run_single_config(
    X_train, Y_train, X_test, Y_test,
    loss_fn, cfg: Dict, args,
    dataset_name: str,
    model_builder=None,
) -> Dict:
    """Train and evaluate one ablation configuration."""
    tf.random.set_seed(42)
    np.random.seed(42)

    # Paths
    model_dir = os.path.join(args.out_models, dataset_name, cfg["name"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_{cfg['name']}.h5")
    weights_path = os.path.join(model_dir, f"weights_{cfg['name']}.weights.h5")
    history_path = os.path.join(model_dir, f"history_{cfg['name']}.json")
    preds_path = os.path.join(model_dir, f"preds_{cfg['name']}.npz")

    # Build model
    input_shape = X_train.shape[1:]  # (seq_len, 1, n_features)
    if model_builder is not None:
        model = model_builder(input_shape)
    else:
        model = auto_select_model(input_shape, len(X_train), args.dropout)

    lr = float(args.lr)
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[root_mean_squared_error])

    total_epochs = int(args.epochs)
    min_lr = 1e-5
    def cosine_lr(epoch, _lr):
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
    cosine_cb = keras.callbacks.LearningRateScheduler(cosine_lr, verbose=0)

    # Check cache
    cached = (
        not args.force_retrain
        and os.path.exists(model_path)
        and os.path.exists(history_path)
        and os.path.exists(preds_path)
    )

    if cached:
        print(f"  [CACHE HIT] {cfg['display_name']} — loading from {model_dir}")
        try:
            model = keras.models.load_model(model_path, compile=False)
            model.compile(loss=loss_fn, optimizer=optimizer, metrics=[root_mean_squared_error])
        except Exception:
            model.load_weights(weights_path)
        with open(history_path) as f:
            history_dict = json.load(f)
        saved = np.load(preds_path)
        Y_pred = saved["Y_pred"]
        Y_test_saved = saved["Y_test"]
    else:
        print(f"  [TRAINING] {cfg['display_name']}...")
        cb = [
            cosine_cb,
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                verbose=1, restore_best_weights=True
            ),
        ]
        hist = model.fit(
            X_train, Y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=cb,
            shuffle=True,
        )

        model.save(model_path)
        model.save_weights(weights_path)

        history_dict = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        Y_pred = model.predict(X_test)
        np.savez_compressed(preds_path, Y_pred=Y_pred, Y_test=Y_test)
        Y_test_saved = Y_test

    # Metrics
    rmse_val = rmse_np(Y_pred, Y_test_saved)
    mae_val = mae_np(Y_pred, Y_test_saved)
    score_val = nasa_score_np(Y_test_saved, Y_pred)
    r2_val = r_squared_np(Y_test_saved, Y_pred)
    mono_mean, slope_rmse_val = physics_metrics_np(Y_test_saved, Y_pred)

    print(f"    RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  Score={score_val:.2f}  "
          f"R²={r2_val:.4f}  Mono={mono_mean:.4f}  Slope={slope_rmse_val:.4f}")

    return {
        "name": cfg["name"],
        "display_name": cfg.get("display_name", cfg["name"]),
        "rmse": rmse_val,
        "mae": mae_val,
        "score": score_val,
        "r2": r2_val,
        "mono": mono_mean,
        "slope": slope_rmse_val,
        "history": history_dict,
        "model_path": model_path,
        "Y_pred": Y_pred,
        "Y_test": Y_test_saved,
    }


# =========================================================================
# Plotting functions
# =========================================================================

def plot_grouped_bars(results, metrics, out_path, title):
    n_configs = len(results)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    axes = axes.flatten()
    x = np.arange(n_configs)
    for m_idx, (mkey, mlabel) in enumerate(metrics):
        ax = axes[m_idx]
        vals = [r[mkey] for r in results]
        labels = [r["display_name"] for r in results]
        bars = ax.bar(x, vals, width=0.6,
                      color=[COLORS[i % len(COLORS)] for i in range(n_configs)],
                      edgecolor="black", linewidth=0.5,
                      hatch=[HATCHES[i % len(HATCHES)] for i in range(n_configs)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(mlabel, fontweight="bold")
        ax.set_title(mlabel, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar(results, metrics, out_path, title):
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    raw_vals = {mk: [r[mk] for r in results] for mk, _ in metrics}
    for i, r in enumerate(results):
        values = []
        for mk, _ in metrics:
            v = r[mk]
            vmin, vmax = min(raw_vals[mk]), max(raw_vals[mk])
            rng = vmax - vmin if vmax != vmin else 1.0
            values.append(1.0 - (v - vmin) / rng)  # inverted: lower=better
        values += values[:1]
        ax.plot(angles, values, "o-", label=r["display_name"],
                color=COLORS[i % len(COLORS)], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ml for _, ml in metrics], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curves(results, out_path, title, log_scale=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        ep = range(1, len(r["history"]["loss"]) + 1)
        ax1.plot(ep, r["history"]["loss"], color=c, linewidth=1.5, label=r["display_name"])
        if "val_loss" in r["history"]:
            ax2.plot(ep, r["history"]["val_loss"], color=c, linewidth=1.5, label=r["display_name"])
    for ax, lbl in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(lbl, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        if log_scale:
            ax.set_yscale("log"); ax.set_ylabel("Loss (log)")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(results, out_path, title):
    n = len(results)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    for idx, r in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        yt, yp = r["Y_test"].flatten(), r["Y_pred"].flatten()
        ax.scatter(yt, yp, s=18, alpha=0.6, c=COLORS[idx % len(COLORS)], edgecolor="none")
        lims = [0, max(yt.max(), yp.max()) + 5]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Ideal")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL")
        ax.set_title(f"{r['display_name']}\nRMSE={r['rmse']:.2f} R²={r['r2']:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_improvement_delta(results, metric, metric_label, out_path, title):
    """Bar chart showing % improvement over MSE-only baseline."""
    baseline = results[0][metric]
    names = [r["display_name"] for r in results]
    pct_improvements = [0.0] + [
        100.0 * (baseline - r[metric]) / (abs(baseline) + 1e-10) for r in results[1:]
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ["#029E73" if p > 0 else "#D55E00" for p in pct_improvements]
    ax.bar(names, pct_improvements, color=colors_bar, edgecolor="black", linewidth=0.5)
    for i, (name, pct) in enumerate(zip(names, pct_improvements)):
        ax.text(i, pct, f"{pct:+.1f}%", ha="center",
                va="bottom" if pct >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"% Improvement in {metric_label}", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_latex_table(results, out_path, dataset_name):
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Generalizability study results on {dataset_name}.}}",
        rf"\label{{tab:gen_{dataset_name}}}",
        r"\small",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Config} & \textbf{RMSE}$\downarrow$ & \textbf{MAE}$\downarrow$ & "
        r"\textbf{R$^2$}$\uparrow$ & \textbf{Score}$\downarrow$ & "
        r"\textbf{Mono}$\downarrow$ & \textbf{Slope}$\downarrow$ \\",
        r"\midrule",
    ]
    best = {
        "rmse": min(r["rmse"] for r in results),
        "mae": min(r["mae"] for r in results),
        "r2": max(r["r2"] for r in results),
        "score": min(r["score"] for r in results),
        "mono": min(r["mono"] for r in results),
        "slope": min(r["slope"] for r in results),
    }

    def _fmt(val, best_val, higher_better=False):
        s = f"{val:.2f}"
        if higher_better:
            is_best = abs(val - best_val) < 1e-6
        else:
            is_best = abs(val - best_val) < 1e-6
        return rf"\textbf{{{s}}}" if is_best else s

    for r in results:
        line = (
            f"{r['display_name']} & "
            f"{_fmt(r['rmse'], best['rmse'])} & "
            f"{_fmt(r['mae'], best['mae'])} & "
            f"{_fmt(r['r2'], best['r2'], True)} & "
            f"{_fmt(r['score'], best['score'])} & "
            f"{_fmt(r['mono'], best['mono'])} & "
            f"{_fmt(r['slope'], best['slope'])} \\\\"
        )
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX table to {out_path}")


# =========================================================================
# Cross-dataset comparison
# =========================================================================

def plot_cross_dataset_comparison(all_results: Dict[str, List[Dict]], metric_key, metric_label, out_path):
    """Grouped bar chart: datasets × ablation configs for one metric."""
    ds_names = list(all_results.keys())
    config_names = [r["display_name"] for r in all_results[ds_names[0]]]
    n_ds = len(ds_names)
    n_configs = len(config_names)
    x = np.arange(n_configs)
    width = 0.8 / n_ds

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_configs), 5))
    for di, ds_name in enumerate(ds_names):
        vals = [r[metric_key] for r in all_results[ds_name]]
        offset = (di - (n_ds - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=ds_name,
                      color=COLORS[di % len(COLORS)], edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"Cross-Dataset: {metric_label}", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_generalizability_heatmap(all_results: Dict[str, List[Dict]], metric_key, metric_label, out_path):
    """Heatmap: datasets (rows) × configs (columns) for one metric."""
    ds_names = list(all_results.keys())
    config_names = [r["display_name"] for r in all_results[ds_names[0]]]

    data = np.zeros((len(ds_names), len(config_names)))
    for i, ds in enumerate(ds_names):
        for j, r in enumerate(all_results[ds]):
            data[i, j] = r[metric_key]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(config_names)), max(4, 0.8 * len(ds_names))))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names, fontsize=10)
    # Annotate cells
    for i in range(len(ds_names)):
        for j in range(len(config_names)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, label=metric_label)
    ax.set_title(f"Generalizability: {metric_label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_cross_dataset_latex(all_results: Dict[str, List[Dict]], out_path):
    """LaTeX table: Full Proposed results across all datasets."""
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Cross-dataset generalizability of Full Proposed PI-DP-FCN.}",
        r"\label{tab:generalizability}",
        r"\small",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{RMSE}$\downarrow$ & \textbf{MAE}$\downarrow$ & "
        r"\textbf{R$^2$}$\uparrow$ & \textbf{Score}$\downarrow$ & "
        r"\textbf{Mono}$\downarrow$ & \textbf{Slope}$\downarrow$ \\",
        r"\midrule",
    ]
    for ds_name in sorted(all_results.keys()):
        full = all_results[ds_name][-1]  # Full Proposed is last
        lines.append(
            f"{ds_name} & {full['rmse']:.2f} & {full['mae']:.2f} & "
            f"{full['r2']:.3f} & {full['score']:.2f} & "
            f"{full['mono']:.4f} & {full['slope']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved cross-dataset LaTeX table to {out_path}")


# =========================================================================
# Per-dataset runner
# =========================================================================

def run_ablation_for_dataset(
    dataset: GDS.GenericTimeSeriesDataset,
    args,
) -> List[Dict]:
    """Run the full ablation study for one dataset."""
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # Optional EMA smoothing (on the 4-D array, dim=2 is the dummy 1-dim)
    if args.smoothing_alpha > 0:
        # Squeeze the dummy dim, smooth, re-expand
        Xtr_3d = X_train[:, :, 0, :]
        Xte_3d = X_test[:, :, 0, :]
        Xtr_3d = apply_ema(Xtr_3d, args.smoothing_alpha)
        Xte_3d = apply_ema(Xte_3d, args.smoothing_alpha)
        X_train = Xtr_3d[:, :, np.newaxis, :]
        X_test = Xte_3d[:, :, np.newaxis, :]

    print(f"\n{'='*60}")
    print(f"  GENERALIZABILITY STUDY — {dataset.name}")
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")
    print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}  "
          f"Dropout={args.dropout}  Smoothing={args.smoothing_alpha}")
    print(f"{'='*60}\n")

    configs = get_ablation_configs(include_score=args.include_score)
    results = []

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"  Config: {cfg['display_name']}")
        print(f"  alpha={cfg['alpha']}, gamma={cfg['gamma']}, asym={cfg['asym_weight']}, "
              f"score={cfg['score_weight']}")
        print(f"{'─'*50}")

        loss_fn = make_physics_loss(
            alpha=cfg["alpha"], gamma=cfg["gamma"],
            asym_weight=cfg["asym_weight"],
            score_weight=cfg["score_weight"],
            smooth_weight=cfg["smooth_weight"],
        )

        res = run_single_config(
            X_train, Y_train, X_test, Y_test,
            loss_fn, cfg, args, dataset.name,
        )
        results.append(res)

    # ---- Save CSV ----
    df = pd.DataFrame([{
        "config": r["display_name"],
        "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
        "nasa_score": r["score"],
        "mono_violation": r["mono"], "slope_rmse": r["slope"],
        "model_path": r["model_path"],
    } for r in results])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_csv, f"generalizability_{dataset.name}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV to {csv_path}")

    # ---- Per-dataset plots ----
    prefix = os.path.join(args.out_plots, dataset.name)
    metrics_list = [
        ("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²"),
        ("score", "NASA Score"), ("mono", "Mono Violation"), ("slope", "Slope RMSE"),
    ]

    plot_grouped_bars(results, metrics_list[:4], f"{prefix}_grouped_bars.png",
                      f"Ablation — {dataset.name}")
    plot_radar(results, [("rmse", "RMSE"), ("mae", "MAE"), ("score", "NASA Score"),
                         ("mono", "Mono"), ("slope", "Slope")],
               f"{prefix}_radar.png", f"Radar — {dataset.name}")
    plot_loss_curves(results, f"{prefix}_loss_curves.png",
                     f"Loss Curves — {dataset.name}")
    plot_loss_curves(results, f"{prefix}_loss_curves_log.png",
                     f"Loss Curves (Log) — {dataset.name}", log_scale=True)
    plot_scatter(results, f"{prefix}_scatter.png",
                 f"True vs Predicted — {dataset.name}")
    plot_improvement_delta(results, "rmse", "RMSE",
                           f"{prefix}_improvement_rmse.png",
                           f"RMSE Improvement — {dataset.name}")
    plot_improvement_delta(results, "score", "NASA Score",
                           f"{prefix}_improvement_score.png",
                           f"Score Improvement — {dataset.name}")
    generate_latex_table(results, f"{prefix}_table.tex", dataset.name)

    # ---- Print summary ----
    print(f"\n{'─'*80}")
    print(f"{'Config':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Score':>10} {'Mono':>8} {'Slope':>8}")
    print(f"{'─'*80}")
    for r in results:
        print(f"{r['display_name']:<25} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.4f} "
              f"{r['score']:>10.2f} {r['mono']:>8.4f} {r['slope']:>8.4f}")
    print(f"{'─'*80}")

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generalizability study for PI-DP-FCN on arbitrary time-series data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic degradation data (quick sanity check)
  python generalizability_study.py --mode synthetic

  # Your own CSV
  python generalizability_study.py --mode csv --train_path train.csv --unit_col unit_id

  # Multiple datasets via JSON config
  python generalizability_study.py --config datasets_config.json
        """,
    )

    # Dataset source
    parser.add_argument("--mode", default="synthetic",
                        choices=["synthetic", "csv", "nasa_bearing", "battery", "phm2012", "numpy"],
                        help="Dataset type (ignored if --config is provided)")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file defining one or more datasets")
    parser.add_argument("--name", default=None,
                        help="Dataset name for labeling outputs")

    # CSV mode
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--test_rul_path", default=None)
    parser.add_argument("--unit_col", default="unit_id")
    parser.add_argument("--cycle_col", default="cycle")
    parser.add_argument("--target_col", default="RUL")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # NASA Bearing / PHM2012 / Battery
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--condition", default="1", help="PHM2012 operating condition")
    parser.add_argument("--eol_threshold", type=float, default=0.7, help="Battery EOL threshold")

    # NumPy mode
    parser.add_argument("--train_X", default=None)
    parser.add_argument("--train_Y", default=None)
    parser.add_argument("--test_X", default=None)
    parser.add_argument("--test_Y", default=None)

    # Synthetic params
    parser.add_argument("--n_units_train", type=int, default=80)
    parser.add_argument("--n_units_test", type=int, default=20)
    parser.add_argument("--max_life", type=int, default=200)
    parser.add_argument("--n_features", type=int, default=14)

    # Model / training
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--rul_cap", type=float, default=125.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--smoothing_alpha", type=float, default=0.0,
                        help="EMA smoothing alpha (0 = disabled)")
    parser.add_argument("--include_score", action="store_true",
                        help="Include score-proxy loss in Full Proposed config")
    parser.add_argument("--force_retrain", action="store_true")

    # Output
    parser.add_argument("--out_root", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "experiments_result", "generalizability")))

    args = parser.parse_args()

    # Output dirs
    args.out_plots = os.path.join(args.out_root, "plots")
    args.out_csv = os.path.join(args.out_root, "csv")
    args.out_models = os.path.join(args.out_root, "models")
    for d in [args.out_plots, args.out_csv, args.out_models]:
        os.makedirs(d, exist_ok=True)

    # ---- Build dataset list ----
    datasets = []

    if args.config:
        # JSON config with one or more datasets
        with open(args.config) as f:
            cfg_data = json.load(f)
        if isinstance(cfg_data, list):
            for c in cfg_data:
                c.setdefault("sequence_length", args.sequence_length)
                c.setdefault("rul_cap", args.rul_cap)
                datasets.append(GDS.load_dataset_from_config(c))
        elif isinstance(cfg_data, dict):
            cfg_data.setdefault("sequence_length", args.sequence_length)
            cfg_data.setdefault("rul_cap", args.rul_cap)
            datasets.append(GDS.load_dataset_from_config(cfg_data))
    else:
        # Single dataset from CLI args
        ds_name = args.name or args.mode
        ds = GDS.GenericTimeSeriesDataset(
            name=ds_name,
            sequence_length=args.sequence_length,
            rul_cap=args.rul_cap,
        )

        if args.mode == "synthetic":
            ds.load_synthetic(
                n_units_train=args.n_units_train,
                n_units_test=args.n_units_test,
                max_life=args.max_life,
                n_features=args.n_features,
            )
        elif args.mode == "csv":
            if not args.train_path:
                parser.error("--train_path is required for csv mode")
            ds.load_from_csv(
                train_path=args.train_path,
                test_path=args.test_path,
                test_rul_path=args.test_rul_path,
                unit_col=args.unit_col,
                cycle_col=args.cycle_col,
                target_col=args.target_col,
                delimiter=args.delimiter,
                train_ratio=args.train_ratio,
            )
        elif args.mode == "nasa_bearing":
            if not args.data_dir:
                parser.error("--data_dir is required for nasa_bearing mode")
            ds.load_nasa_bearing(data_dir=args.data_dir)
        elif args.mode == "battery":
            if not args.csv_path:
                parser.error("--csv_path is required for battery mode")
            ds.load_battery(
                csv_path=args.csv_path,
                eol_threshold=args.eol_threshold,
            )
        elif args.mode == "phm2012":
            if not args.data_dir:
                parser.error("--data_dir is required for phm2012 mode")
            ds.load_phm2012(
                data_dir=args.data_dir,
                condition=args.condition,
            )
        elif args.mode == "numpy":
            for p in [args.train_X, args.train_Y, args.test_X, args.test_Y]:
                if not p:
                    parser.error("All of --train_X, --train_Y, --test_X, --test_Y required")
            X_train = np.load(args.train_X)
            Y_train = np.load(args.train_Y)
            X_test = np.load(args.test_X)
            Y_test = np.load(args.test_Y)
            ds.load_from_numpy(X_train, Y_train, X_test, Y_test)

        datasets.append(ds)

    # ---- Run ablation for each dataset ----
    all_results: Dict[str, List[Dict]] = {}
    for ds in datasets:
        print(f"\n{'#'*60}")
        print(ds.summary())
        print(f"{'#'*60}")
        results = run_ablation_for_dataset(ds, args)
        all_results[ds.name] = results

    # ---- Cross-dataset comparison (if multiple datasets) ----
    if len(datasets) > 1:
        print(f"\n{'#'*60}")
        print(f"  CROSS-DATASET GENERALIZABILITY SUMMARY")
        print(f"{'#'*60}")

        # Use common configs across datasets
        common_names = None
        for ds_name in all_results:
            names = {r["display_name"] for r in all_results[ds_name]}
            common_names = names if common_names is None else common_names & names

        filtered = {}
        for ds_name in all_results:
            filtered[ds_name] = [r for r in all_results[ds_name] if r["display_name"] in common_names]

        cross_prefix = os.path.join(args.out_plots, "cross_dataset")
        for mk, ml in [("rmse", "RMSE"), ("mae", "MAE"), ("score", "NASA Score"),
                        ("mono", "Mono Violation"), ("slope", "Slope RMSE")]:
            plot_cross_dataset_comparison(filtered, mk, ml, f"{cross_prefix}_{mk}.png")
            plot_generalizability_heatmap(filtered, mk, ml, f"{cross_prefix}_heatmap_{mk}.png")

        generate_cross_dataset_latex(all_results, f"{cross_prefix}_table.tex")

        # Summary CSV
        rows = []
        for ds_name in sorted(all_results.keys()):
            for r in all_results[ds_name]:
                rows.append({
                    "dataset": ds_name, "config": r["display_name"],
                    "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
                    "nasa_score": r["score"], "mono": r["mono"], "slope": r["slope"],
                })
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cross_csv = os.path.join(args.out_csv, f"generalizability_all_{ts}.csv")
        pd.DataFrame(rows).to_csv(cross_csv, index=False)
        print(f"  Saved cross-dataset CSV to {cross_csv}")

        # Grand summary
        print(f"\n{'─'*90}")
        print(f"{'Dataset':<20} {'Config':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Score':>10}")
        print(f"{'─'*90}")
        for ds_name in sorted(all_results.keys()):
            full = all_results[ds_name][-1]
            mse_only = all_results[ds_name][0]
            delta = 100 * (mse_only["rmse"] - full["rmse"]) / (mse_only["rmse"] + 1e-10)
            print(f"{ds_name:<20} {full['display_name']:<25} {full['rmse']:>8.2f} "
                  f"{full['mae']:>8.2f} {full['r2']:>8.4f} {full['score']:>10.2f}  "
                  f"(RMSE impr: {delta:+.1f}%)")
        print(f"{'─'*90}")

    print(f"\n{'='*60}")
    print(f"  GENERALIZABILITY STUDY COMPLETE")
    print(f"  Outputs: {args.out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
