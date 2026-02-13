
import argparse
import datetime
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
import CMAPSSDataset

# -------------------------
# Metrics and helpers
# -------------------------

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse_np(predictions, targets):
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


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


def unbalanced_penalty_score_1out(Y_test, Y_pred):
    s = 0.0
    for i in range(len(Y_pred)):
        if Y_pred[i] > Y_test[i]:
            s += math.exp((Y_pred[i] - Y_test[i]) / 10) - 1
        else:
            s += math.exp((Y_test[i] - Y_pred[i]) / 13) - 1
    return float(s)


def error_range_1out(Y_test, Y_pred):
    return float((Y_test - Y_pred).min()), float((Y_test - Y_pred).max())


# -------------------------
# Loss factory
# -------------------------

def make_physics_loss(alpha: float, beta: float, gamma: float):
    def _diff_mask(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        true_diffs = y_true[1:] - y_true[:-1]
        pred_diffs = y_pred[1:] - y_pred[:-1]
        same_engine_mask = K.cast(true_diffs < 0.0, K.floatx())
        return true_diffs, pred_diffs, same_engine_mask

    def loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        mse = K.mean(K.square(y_true_flat - y_pred_flat))
        true_diffs, pred_diffs, same_engine_mask = _diff_mask(y_true_flat, y_pred_flat)
        masked_pred_diffs = pred_diffs * same_engine_mask
        mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(same_engine_mask) + K.epsilon())
        slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * same_engine_mask)) / (
            K.sum(same_engine_mask) + K.epsilon()
        )
        if beta > 0:
            ddiffs = pred_diffs[1:] - pred_diffs[:-1]
            smooth_penalty = K.mean(K.square(ddiffs))
        else:
            smooth_penalty = 0.0
        return mse + alpha * mono_penalty + gamma * slope_penalty + beta * smooth_penalty

    return loss


# -------------------------
# Model builder
# -------------------------

def build_fcn(input_shape, num_filter1=64, num_filter2=128, num_filter3=64, k1=16, k2=10, k3=6):
    in0 = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(num_filter1, k1, strides=1, padding="same")(in0)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filter2, k2, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filter3, k3, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(1, activation="relu")(x)
    return keras.models.Model(inputs=in0, outputs=[out])


# -------------------------
# Data loader
# -------------------------

def load_data(fd: str, num_test: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    train_label_slice[train_label_slice > 115] = 115
    test_label_slice[test_label_slice > 115] = 115

    X_train = np.reshape(train_feature_slice, (-1, train_feature_slice.shape[1], 1, train_feature_slice.shape[2]))
    Y_train = train_label_slice
    X_test = np.reshape(test_feature_slice, (-1, test_feature_slice.shape[1], 1, test_feature_slice.shape[2]))
    Y_test = test_label_slice
    return X_train, Y_train, X_test, Y_test


# -------------------------
# Training + evaluation for one config
# -------------------------

def run_config(X_train, Y_train, X_test, Y_test, loss_fn, cfg: Dict, args) -> Dict:
    model = build_fcn(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
        num_filter1=args.num_filter1,
        num_filter2=args.num_filter2,
        num_filter3=args.num_filter3,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
    )
    optimizer = keras.optimizers.Adam()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[root_mean_squared_error])

    cb = [
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=args.patience_reduce_lr, min_lr=0.0001),
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
        shuffle=False,
    )

    # Save best model
    model_dir = os.path.join(args.out_models, cfg["name"])
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_{cfg['name']}.h5")
    model.save(model_path)

    Y_pred = model.predict(X_test)
    rmse_val = rmse_np(Y_pred, Y_test)
    upe_val = unbalanced_penalty_score_1out(Y_test, Y_pred)
    er_left, er_right = error_range_1out(Y_test, Y_pred)
    mono_mean, slope_rmse = physics_metrics_np(Y_test, Y_pred)

    return {
        "name": cfg["name"],
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "gamma": cfg["gamma"],
        "rmse": rmse_val,
        "upe": upe_val,
        "er_left": er_left,
        "er_right": er_right,
        "mono": mono_mean,
        "slope": slope_rmse,
        "history": hist.history,
        "model_path": model_path,
    }


# -------------------------
# Plotting
# -------------------------

def plot_metric_bar(results: List[Dict], metric: str, out_path: str, title: str):
    labels = [r["name"] for r in results]
    vals = [r[metric] for r in results]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="#4c72b0")
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_losses(results: List[Dict], out_path: str):
    plt.figure(figsize=(7, 4))
    for r in results:
        plt.plot(r["history"]["loss"], label=f"{r['name']}-train")
        if "val_loss" in r["history"]:
            plt.plot(r["history"]["val_loss"], linestyle="--", label=f"{r['name']}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study for physics-informed FCN RUL")
    parser.add_argument("--fd", default="4", help="FD set: 1,2,3,4")
    parser.add_argument("--num_test", type=int, default=100, choices=[100, 10000], help="Test slicing mode")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--patience_reduce_lr", type=int, default=15)
    parser.add_argument("--num_filter1", type=int, default=64)
    parser.add_argument("--num_filter2", type=int, default=128)
    parser.add_argument("--num_filter3", type=int, default=64)
    parser.add_argument("--k1", type=int, default=16)
    parser.add_argument("--k2", type=int, default=10)
    parser.add_argument("--k3", type=int, default=6)
    parser.add_argument("--out_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments_result")))
    args = parser.parse_args()

    args.out_ablation = os.path.join(args.out_root, "ablation")
    args.out_ablation_csv = os.path.join(args.out_root, "ablation_csv")
    args.out_models = os.path.join(args.out_root, "ablation_models")
    os.makedirs(args.out_ablation, exist_ok=True)
    os.makedirs(args.out_ablation_csv, exist_ok=True)
    os.makedirs(args.out_models, exist_ok=True)

    X_train, Y_train, X_test, Y_test = load_data(args.fd, args.num_test, args.batch_size)

    configs = [
        {"name": "mse_only", "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        {"name": "mono", "alpha": 0.1, "beta": 0.0, "gamma": 0.0},
        {"name": "mono_slope", "alpha": 0.1, "beta": 0.0, "gamma": 0.05},
        {"name": "mono_slope_smooth", "alpha": 0.1, "beta": 1e-3, "gamma": 0.05},
    ]

    results = []
    for cfg in configs:
        print(f"=== Running config {cfg['name']} (alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']}) ===")
        loss_fn = make_physics_loss(cfg["alpha"], cfg["beta"], cfg["gamma"])
        res = run_config(X_train, Y_train, X_test, Y_test, loss_fn, cfg, args)
        results.append(res)

    # Save table
    df = pd.DataFrame(
        [
            {
                "name": r["name"],
                "alpha": r["alpha"],
                "beta": r["beta"],
                "gamma": r["gamma"],
                "rmse": r["rmse"],
                "upe": r["upe"],
                "er_left": r["er_left"],
                "er_right": r["er_right"],
                "mono": r["mono"],
                "slope": r["slope"],
                "model_path": r["model_path"],
            }
            for r in results
        ]
    )
    csv_path = os.path.join(
        args.out_ablation_csv,
        f"ablation_fd{args.fd}_numtest{args.num_test}_time{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics table to {csv_path}")

    # Plots
    plot_metric_bar(results, "rmse", os.path.join(args.out_ablation, f"rmse_fd{args.fd}.png"), "RMSE comparison")
    plot_metric_bar(results, "upe", os.path.join(args.out_ablation, f"upe_fd{args.fd}.png"), "UPE comparison")
    plot_metric_bar(results, "mono", os.path.join(args.out_ablation, f"mono_fd{args.fd}.png"), "Monotonic violation")
    plot_metric_bar(results, "slope", os.path.join(args.out_ablation, f"slope_fd{args.fd}.png"), "Slope RMSE")
    plot_losses(results, os.path.join(args.out_ablation, f"loss_curves_fd{args.fd}.png"))
    print(f"Saved plots to {args.out_ablation}")


if __name__ == "__main__":
    main()
