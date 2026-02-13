# -*- coding: utf-8 -*-
"""
Physics-informed CNN-BiLSTM for CMAPSS RUL (monotonic degradation penalty)
"""
import os
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import (
    Lambda, Conv1D, BatchNormalization, Activation, Dropout,
    Bidirectional, LSTM, Dense, GlobalAveragePooling1D,
    Concatenate, Reshape, Flatten, Softmax, Dot,
)
import CMAPSSDataset

# Paths (use script location, not cwd, so saved models are found correctly)
_script_dir = os.path.dirname(os.path.abspath(__file__))
last_last_path = os.path.abspath(os.path.join(_script_dir, "../.."))
last_path = os.path.abspath(os.path.join(_script_dir, ".."))
print(f"last_path: {last_path}")

# Metrics

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def physics_metrics_np(y_true, y_pred):
    """Compute physics diagnostics on numpy arrays (monotonic violations and slope error)."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    true_diffs = y_true_flat[1:] - y_true_flat[:-1]
    pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]
    mask = (true_diffs < 0).astype(np.float32)
    denom = np.sum(mask) + 1e-8
    mono_violation = np.maximum(pred_diffs, 0) * mask
    mono_mean = mono_violation.sum() / denom
    slope_sqerr = ((pred_diffs - true_diffs) ** 2) * mask
    slope_rmse = math.sqrt(slope_sqerr.sum() / denom)
    return mono_mean, slope_rmse

# Physics-inspired regularization to discourage RUL increases over time
# and to keep predicted slopes close to observed slopes (data-driven linear prior).
physics_alpha = 0.02  # weight for monotonicity penalty (reduced for better fitting)
physics_beta = 0.0    # optional curvature smoothing across windows
physics_gamma = 0.02  # weight for slope tracking vs. observed RUL deltas (reduced)


def _diff_mask(y_true, y_pred):
    """Return diffs and a mask that is 1 only when consecutive samples are from the same engine.

    Consecutive labels inside the same engine drop (negative delta). When a new engine begins,
    the delta is positive; we mask those boundaries to avoid penalizing cross-engine jumps.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    same_engine_mask = K.cast(true_diffs < 0.0, K.floatx())
    return true_diffs, pred_diffs, same_engine_mask


def physics_loss(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    mse = K.mean(K.square(y_true_flat - y_pred_flat))

    true_diffs, pred_diffs, same_engine_mask = _diff_mask(y_true_flat, y_pred_flat)

    # Monotonicity: penalize upward jumps (positive deltas) only within engines.
    masked_pred_diffs = pred_diffs * same_engine_mask
    mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(same_engine_mask) + K.epsilon())

    # Linear decay prior: keep predicted slopes close to observed slopes inside each engine.
    slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * same_engine_mask)) / (
        K.sum(same_engine_mask) + K.epsilon()
    )

    # Optional curvature smoothing on predicted slopes (rarely used, kept for completeness).
    if physics_beta > 0:
        ddiffs = pred_diffs[1:] - pred_diffs[:-1]
        smooth_penalty = K.mean(K.square(ddiffs))
    else:
        smooth_penalty = 0.0

    return mse + physics_alpha * mono_penalty + physics_gamma * slope_penalty + physics_beta * smooth_penalty

# Hyperparameters
num_test = 100
run_times = 1
nb_epochs = 200
batch_size = 512       # smaller batch for better generalization
patience = 50
patience_reduce_lr = 15  # reduce LR sooner

# CNN hyperparameters
num_filter1 = 64
num_filter2 = 128
num_filter3 = 64
kernel1_size = 7       # smaller kernels work better with BiLSTM
kernel2_size = 5
kernel3_size = 3

# BiLSTM hyperparameters
lstm_units = 64
dropout_rate = 0.3

# Storage for cross-FD summary
fd_summary = {}

for FD in ['1','2']:#,'2','3','4']:  # []
    # Feature selection per FD
    if FD == "1":
        sequence_length = 31
        FD_feature_columns = [
            "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12",
            "s13", "s14", "s15", "s17", "s20", "s21",
        ]
    if FD == "2":
        sequence_length = 21
        FD_feature_columns = [
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
            "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
        ]
    if FD == "3":
        sequence_length = 38
        FD_feature_columns = [
            "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
            "s13", "s14", "s15", "s17", "s20", "s21",
        ]
    if FD == "4":
        sequence_length = 19
        FD_feature_columns = [
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
            "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s20", "s21",
        ]

    method_name = "grid_FD{}_without_num_test{}".format(FD, num_test)
    dataset = "cmapssd"

    def unbalanced_penalty_score_1out(Y_test, Y_pred):
        s = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] > Y_test[i]:
                s = s + math.exp((Y_pred[i] - Y_test[i]) / 10) - 1
            else:
                s = s + math.exp((Y_test[i] - Y_pred[i]) / 13) - 1
        print("unbalanced_penalty_score{}".format(s))
        return s

    def error_range_1out(Y_test, Y_pred):
        error_range = (Y_test - Y_pred).min(), (Y_test - Y_pred).max()
        print("error range{}".format(error_range))
        return error_range

    datasets = CMAPSSDataset.CMAPSSDataset(
        fd_number=FD,
        batch_size=batch_size,
        sequence_length=sequence_length,
        deleted_engine=[1000],
        feature_columns=FD_feature_columns,
    )

    train_data = datasets.get_train_data()
    train_feature_slice = datasets.get_feature_slice(train_data)
    train_label_slice = datasets.get_label_slice(train_data)

    print("train_data.shape: {}".format(train_data.shape))
    print("train_feature_slice.shape: {}".format(train_feature_slice.shape))
    print("train_label_slice.shape: {}".format(train_label_slice.shape))


    test_data = datasets.get_test_data()
    if num_test == 100:
        test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
    if num_test == 10000:
        test_feature_slice = datasets.get_feature_slice(test_data)
        test_label_slice = datasets.get_label_slice(test_data)

    print("test_data.shape: {}".format(test_data.shape))
    print("test_feature_slice.shape: {}".format(test_feature_slice.shape))
    print("test_label_slice.shape: {}".format(test_label_slice.shape))

    # 3D input (samples, timesteps, features) for Conv1D + BiLSTM
    X_train = train_feature_slice  # shape: (samples, sequence_length, num_features)
    train_label_slice[train_label_slice > 115] = 115
    Y_train = train_label_slice

    X_test = test_feature_slice    # shape: (samples, sequence_length, num_features)
    test_label_slice[test_label_slice > 115] = 115
    Y_test = test_label_slice

    print("X_train.shape: {}".format(X_train.shape))
    print("Y_train.shape: {}".format(Y_train.shape))
    print("X_test.shape: {}".format(X_test.shape))
    print("Y_test.shape: {}".format(Y_test.shape))

    def attention_layer(inputs):
        """Simple temporal attention mechanism."""
        # inputs shape: (batch, timesteps, features)
        score = Dense(1, activation='tanh')(inputs)          # (batch, timesteps, 1)
        score = Flatten()(score)                             # (batch, timesteps)
        attention_weights = Softmax()(score)                 # (batch, timesteps)
        # Weighted sum
        context = Dot(axes=[1, 1])([inputs, Reshape((-1, 1))(attention_weights)])
        context = Flatten()(context)
        return context

    def CNN_BiLSTM_model():
        """CNN-BiLSTM with Attention for RUL prediction."""
        in0 = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))  # (timesteps, features)

        # === CNN Block: Extract local degradation patterns ===
        x = Conv1D(num_filter1, kernel1_size, strides=1, padding="same")(in0)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout_rate * 0.5)(x)

        x = Conv1D(num_filter2, kernel2_size, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout_rate * 0.5)(x)

        x = Conv1D(num_filter3, kernel3_size, strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # x shape: (batch, timesteps, num_filter3)

        # === BiLSTM Block: Capture temporal dependencies in both directions ===
        lstm_out = Bidirectional(
            LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1),
            merge_mode='concat'
        )(x)
        # lstm_out shape: (batch, timesteps, lstm_units*2)

        lstm_out2 = Bidirectional(
            LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1),
            merge_mode='concat'
        )(lstm_out)
        # lstm_out2 shape: (batch, timesteps, lstm_units)

        # === Attention: Focus on the most informative timesteps ===
        attn_out = attention_layer(lstm_out2)  # (batch, lstm_units)

        # === Also use global average pooling from CNN as a parallel path ===
        gap = GlobalAveragePooling1D()(x)  # (batch, num_filter3)

        # === Merge both representations ===
        merged = Concatenate()([attn_out, gap])

        # === Fully connected head ===
        merged = Dense(128, activation="relu")(merged)
        merged = Dropout(dropout_rate)(merged)
        merged = Dense(64, activation="relu")(merged)
        merged = Dropout(dropout_rate * 0.5)(merged)
        out = Dense(1, activation="relu")(merged)

        return keras.models.Model(inputs=in0, outputs=[out])

    if __name__ == "__main__":
        error_record = []
        unbalanced_penalty_score_record = []
        error_range_left_record = []
        error_range_right_record = []
        mono_violation_record = []
        slope_rmse_record = []
        index_min_val_loss_record, min_val_loss_record = [], []

        log_dir = os.path.join(last_path, "experiments_result", "log")
        err_dir = os.path.join(last_path, "experiments_result", "method_error_txt")
        model_dir = os.path.join(last_path, "model", "Physics_BiLSTM", f"FD{FD}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        method_error_path = os.path.join(err_dir, f"{method_name}_bilstm.txt")
        if os.path.exists(method_error_path):
            os.remove(method_error_path)

        for i in range(run_times):
            print(f"=== FD{FD} Run {i} ===")
            model = CNN_BiLSTM_model()
            model.summary()
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(loss=physics_loss, optimizer=optimizer, metrics=[root_mean_squared_error])

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=patience_reduce_lr, min_lr=1e-5
            )
            model_name = "{}_dataset_{}_bilstm_run{}".format(method_name, dataset, i)
            model_path_full = os.path.join(model_dir, f"{model_name}.h5")
            print(f"Checking for existing model at: {model_path_full}")
            already_trained = False
            if os.path.exists(model_path_full):
                print(f"Found existing checkpoint {model_path_full}, loading weights and skipping training.")
                model = keras.models.load_model(
                    model_path_full,
                    custom_objects={"root_mean_squared_error": root_mean_squared_error, "physics_loss": physics_loss},
                )
                already_trained = True
            else:
                earlystopping = keras.callbacks.EarlyStopping(
                    monitor="loss", patience=patience, verbose=1, restore_best_weights=True
                )
                modelcheckpoint = keras.callbacks.ModelCheckpoint(
                    monitor="loss",
                    filepath=model_path_full,
                    save_best_only=True,
                    verbose=1,
                )
                hist = model.fit(
                    X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[reduce_lr, earlystopping, modelcheckpoint],
                    shuffle=True,  # shuffle for better training
                )

                log = pd.DataFrame(hist.history)
                log.to_excel(
                    os.path.join(
                        log_dir,
                        "{}_dataset_{}_log{}_time{}.xlsx".format(
                            method_name, dataset, i, datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        ),
                    )
                )

                epochs = range(len(hist.history["loss"]))
                plt.figure()
                plt.plot(epochs, hist.history["loss"], "b", label="Training loss")
                plt.plot(epochs, hist.history["val_loss"], "r", label="Validation val_loss")
                plt.title(f"FD{FD} CNN-BiLSTM Training and Validation loss")
                plt.legend()
                plt.show()

            model = keras.models.load_model(
                model_path_full,
                custom_objects={"root_mean_squared_error": root_mean_squared_error, "physics_loss": physics_loss},
            )
            for layer in model.layers:
                layer.trainable = False

            Y_pred = model.predict(X_test)
            rmse_value = rmse(Y_test, Y_pred)
            print("rmse:{}".format(rmse_value))

            mono_violation_mean, slope_rmse = physics_metrics_np(Y_test, Y_pred)
            print("physics_mono_violation_mean:{}".format(mono_violation_mean))
            print("physics_slope_rmse:{}".format(slope_rmse))

            unbalanced_penalty_score = unbalanced_penalty_score_1out(Y_test, Y_pred)
            error_range = error_range_1out(Y_test, Y_pred)

            if not already_trained: # type: ignore
                index_min_val_loss = log["loss"].idxmin()
                min_val_loss = log["val_loss"].iloc[index_min_val_loss]
                index_min_val_loss_record.append(index_min_val_loss)
                min_val_loss_record.append(min_val_loss)
            else:
                index_min_val_loss, min_val_loss = -1, -1
            error_record.append(rmse_value)
            unbalanced_penalty_score_record.append(unbalanced_penalty_score)
            error_range_left_record.append(error_range[0])
            error_range_right_record.append(error_range[1])
            mono_violation_record.append(mono_violation_mean)
            slope_rmse_record.append(slope_rmse)

            with open(method_error_path, "a") as file:
                file.write(
                    "       (" + str(i) + ")   "
                    + "index_min_loss:" + str(index_min_val_loss)
                    + "        min_val_loss:" + str(min_val_loss)
                    + "     RMSE:    " + str("%.6f" % (rmse_value))
                    + "     UPE:    " + str("%.6f" % (unbalanced_penalty_score))
                    + "    ER:(" + str("%.6f" % (error_range[0])) + "," + str("%.6f" % (error_range[1])) + ")"
                    + "    MonoViolation: " + str("%.6f" % mono_violation_mean)
                    + "    SlopeRMSE: " + str("%.6f" % slope_rmse)
                    + "\n"
                )

        r0, r1, r2, r3 = [], [], [], []
        for i in range(len(error_record)):
            if error_record[i] < 40:
                r0.append(unbalanced_penalty_score_record[i])
                r1.append(error_range_left_record[i])
                r2.append(error_range_right_record[i])
                r3.append(error_record[i])
        unbalanced_penalty_score_record = r0
        error_range_left_record = r1
        error_range_right_record = r2
        error_record = r3
        with open(method_error_path, "a") as file:
            file.write(
                "    mean_score:" + "     (" + str(np.mean(error_record)) + ")     "
                + "       "
                + "     mean_RMSE:   " + str("%.6f" % (np.mean(error_record)))
                + "     UPE:    " + str("%.6f" % (np.mean(unbalanced_penalty_score_record)))
                + "    (" + str("%.6f" % (np.mean(error_range_left_record)))
                + "," + str("%.6f" % (np.mean(error_range_right_record)))
                + ")"
                + "    MonoViolation: " + str("%.6f" % (np.mean(mono_violation_record)))
                + "    SlopeRMSE: " + str("%.6f" % (np.mean(slope_rmse_record)))
                + "        " + "\n"
            )

        # Store summary for this FD
        fd_summary[FD] = {
            "rmse": np.mean(error_record) if error_record else float('nan'),
            "upe": np.mean(unbalanced_penalty_score_record) if unbalanced_penalty_score_record else float('nan'),
            "mono": np.mean(mono_violation_record) if mono_violation_record else float('nan'),
            "slope": np.mean(slope_rmse_record) if slope_rmse_record else float('nan'),
        }

        error_record = []
        unbalanced_penalty_score_record = []
        error_range_left_record = []
        error_range_right_record = []
        mono_violation_record = []
        slope_rmse_record = []

# After all FD datasets, generate summary plots and table
if __name__ == "__main__" and fd_summary:
    print("\n=== Cross-FD Summary (CNN-BiLSTM) ===")
    for fd_key in sorted(fd_summary.keys()):
        s = fd_summary[fd_key]
        print(f"FD{fd_key}: RMSE={s['rmse']:.4f}, UPE={s['upe']:.2f}, Mono={s['mono']:.6f}, Slope={s['slope']:.4f}")

    # Save summary CSV
    summary_csv_path = os.path.join(last_path, "experiments_result", "physics_bilstm_fd_summary.csv")
    summary_df = pd.DataFrame([
        {"FD": f"FD{k}", "RMSE": v["rmse"], "UPE": v["upe"], "MonoViolation": v["mono"], "SlopeRMSE": v["slope"]}
        for k, v in fd_summary.items()
    ])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV to {summary_csv_path}")

    # Comparison bar plots
    fig_dir = os.path.join(last_path, "figure", "physics_bilstm_summary")
    os.makedirs(fig_dir, exist_ok=True)
    fds = [f"FD{k}" for k in sorted(fd_summary.keys())]
    metrics = {"RMSE": "rmse", "UPE": "upe", "MonoViolation": "mono", "SlopeRMSE": "slope"}
    for metric_label, metric_key in metrics.items():
        vals = [fd_summary[k][metric_key] for k in sorted(fd_summary.keys())]
        plt.figure(figsize=(5, 4))
        plt.bar(fds, vals, color="#4c72b0")
        plt.title(f"{metric_label} across FD datasets (CNN-BiLSTM)")
        plt.ylabel(metric_label)
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{metric_label}_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")
