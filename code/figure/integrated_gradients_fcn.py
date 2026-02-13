# -*- coding: utf-8 -*-
"""
Integrated Gradients explainability for the FCN RUL model.
Generates time × sensor saliency heatmaps on CMAPSS slices.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
import CMAPSSDataset


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def physics_loss(y_true, y_pred):
    # Matching the training script for loading checkpoints
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    mse = K.mean(K.square(y_true_flat - y_pred_flat))
    true_diffs = y_true_flat[1:] - y_true_flat[:-1]
    pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]
    mask = K.cast(true_diffs < 0.0, K.floatx())
    mono_penalty = K.sum(K.relu(pred_diffs * mask)) / (K.sum(mask) + K.epsilon())
    slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * mask)) / (K.sum(mask) + K.epsilon())
    return mse + 0.1 * mono_penalty + 0.05 * slope_penalty


def integrated_gradients(model, x, baseline=None, steps=32):
    """Compute Integrated Gradients for a single sample (x shape: 1, T, 1, F)."""
    if baseline is None:
        baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    alphas_x = tf.reshape(alphas, (-1, 1, 1, 1, 1))
    interpolated = baseline + alphas_x * (x - baseline)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
    grads = tape.gradient(preds, interpolated)
    avg_grads = tf.reduce_mean(grads, axis=0)  # average across alphas
    ig = (x - baseline) * avg_grads
    ig_map = tf.reduce_sum(tf.abs(ig), axis=2)[0]  # remove channel dim -> (T, F)
    return ig_map.numpy()


def plot_heatmap(ig_map, feature_names, out_path, title):
    plt.figure(figsize=(8, 4))
    plt.imshow(ig_map, aspect="auto", cmap="magma")
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.yticks(np.arange(ig_map.shape[0]), np.arange(ig_map.shape[0]))
    plt.xlabel("Sensor / setting")
    plt.ylabel("Time step in window")
    plt.title(title)
    plt.colorbar(label="|Integrated Gradients|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_feature_config(fd):
    if fd == "1":
        return 31, ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    if fd == "2":
        return 21, ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
    if fd == "3":
        return 38, ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    if fd == "4":
        return 19, ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s20", "s21"]
    raise ValueError("Unsupported FD number")


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients for FCN RUL model")
    parser.add_argument("--fd", default="4", help="FD set: 1,2,3,4")
    parser.add_argument("--model_path", required=True, help="Path to trained .h5 model")
    parser.add_argument("--num_test", type=int, default=100, choices=[100, 10000], help="Test slicing mode")
    parser.add_argument("--sample_index", type=int, default=0, help="Test sample index to explain")
    parser.add_argument("--steps", type=int, default=32, help="IG integration steps")
    parser.add_argument("--out_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figure")), help="Directory to save heatmap")
    args = parser.parse_args()

    sequence_length, feature_cols = build_feature_config(args.fd)

    datasets = CMAPSSDataset.CMAPSSDataset(
        fd_number=args.fd,
        batch_size=1024,
        sequence_length=sequence_length,
        deleted_engine=[1000],
        feature_columns=feature_cols,
    )

    test_data = datasets.get_test_data()
    if args.num_test == 100:
        feature_slice, label_slice = datasets.get_last_data_slice(test_data)
    else:
        feature_slice = datasets.get_feature_slice(test_data)
        label_slice = datasets.get_label_slice(test_data)

    label_slice[label_slice > 115] = 115
    x_test = np.reshape(feature_slice, (-1, feature_slice.shape[1], 1, feature_slice.shape[2]))

    sample_idx = max(0, min(args.sample_index, x_test.shape[0] - 1))
    sample = x_test[sample_idx : sample_idx + 1]

    model = keras.models.load_model(
        args.model_path,
        custom_objects={"root_mean_squared_error": root_mean_squared_error, "physics_loss": physics_loss},
    )

    ig_map = integrated_gradients(model, tf.constant(sample, dtype=tf.float32), steps=args.steps)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model_path))[0]
    out_path = os.path.join(args.out_dir, f"ig_fd{args.fd}_{base}_idx{sample_idx}.png")
    title = f"Integrated Gradients - FD{args.fd} sample {sample_idx}"
    plot_heatmap(ig_map, feature_cols, out_path, title)

    print(f"Saved IG heatmap to: {out_path}")
    print(f"Sample true RUL: {label_slice[sample_idx][0]:.1f}")


if __name__ == "__main__":
    main()
