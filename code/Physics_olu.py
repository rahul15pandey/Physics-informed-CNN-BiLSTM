# -*- coding: utf-8 -*-
"""
Physics-informed FCN for CMAPSS RUL (Dual Pooling + Asymmetric Loss)
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

# Paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
last_last_path = os.path.abspath(os.path.join(_script_dir, "../.."))
last_path = os.path.abspath(os.path.join(_script_dir, ".."))
print(f"last_path: {last_path}")

# --- 1. Metrics ---

def root_mean_squared_error_np(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def physics_metrics_np(y_true, y_pred):
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

# --- 2. Physics Loss (Enhanced for Score Reduction) ---

# Weights are reduced to prevent fighting against MSE
physics_alpha = 0.001   # Monotonicity weight (Low)
physics_gamma = 0.001   # Slope weight (Low)
# New: Penalty for over-estimating RUL (Late prediction = High Score Penalty)
late_prediction_weight = 0.2 

def _diff_mask(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    same_engine_mask = K.cast(true_diffs < 0.0, K.floatx())
    return true_diffs, pred_diffs, same_engine_mask

@register_keras_serializable()
def physics_loss(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    # 1. Standard MSE
    mse = K.mean(K.square(y_true_flat - y_pred_flat))

    # 2. Asymmetric Penalty (Targets the CMAPSS Score)
    # Penalize (Pred - True) ONLY if Pred > True
    # This teaches the model to be conservative (underrate RUL slightly)
    # which drastically reduces the Scoring metric.
    over_estimation = K.relu(y_pred_flat - y_true_flat)
    asymmetric_loss = K.mean(K.square(over_estimation))

    # 3. Physics Constraints
    true_diffs, pred_diffs, same_engine_mask = _diff_mask(y_true_flat, y_pred_flat)
    masked_pred_diffs = pred_diffs * same_engine_mask
    
    # Monotonicity
    mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(same_engine_mask) + K.epsilon())
    
    # Slope
    slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * same_engine_mask)) / (
        K.sum(same_engine_mask) + K.epsilon()
    )

    return mse + (late_prediction_weight * asymmetric_loss) + (physics_alpha * mono_penalty) + (physics_gamma * slope_penalty)

# --- 3. Hyperparameters ---
num_test = 100
run_times = 1
nb_epochs = 200
batch_size = 1024  # Kept Large as requested
patience = 40     
patience_reduce_lr = 15

# Architecture Params
num_filter1 = 32
num_filter2 = 64
num_filter3 = 128 # Increased depth
kernel1_size = 11 # Wider field of view
kernel2_size = 9
kernel3_size = 5

# --- Helper: Exponential Smoothing ---
def apply_exponential_smoothing(data, alpha=0.1):
    """Smooths the features along the time axis."""
    smoothed_data = np.copy(data)
    # Manual loop to ensure compatibility
    for i in range(smoothed_data.shape[0]): # Samples
        for j in range(smoothed_data.shape[2]): # Features
            # Simple vectorization for EMA
            # s[t] = alpha*x[t] + (1-alpha)*s[t-1]
            x = smoothed_data[i, :, j]
            s = np.zeros_like(x)
            s[0] = x[0]
            for t in range(1, len(x)):
                s[t] = alpha * x[t] + (1-alpha) * s[t-1]
            smoothed_data[i, :, j] = s
    return smoothed_data

fd_summary = {}

for FD in ['1', '2']: #, '3', '4']: 
    if FD == "1":
        sequence_length = 31
        FD_feature_columns = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    if FD == "2":
        sequence_length = 21
        FD_feature_columns = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
    if FD == "3":
        sequence_length = 38
        FD_feature_columns = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    if FD == "4":
        sequence_length = 19
        FD_feature_columns = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s20", "s21"]

    method_name = "DualPooling_AsymLoss_FD{}_without_num_test{}".format(FD, num_test)
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

    test_data = datasets.get_test_data()
    if num_test == 100:
        test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
    if num_test == 10000:
        test_feature_slice = datasets.get_feature_slice(test_data)
        test_label_slice = datasets.get_label_slice(test_data)

    # --- APPLY SMOOTHING ---
    # This reduces noise significantly, helping the large batch training converge
    print("Applying Exponential Smoothing...")
    train_feature_slice = apply_exponential_smoothing(train_feature_slice, alpha=0.3)
    test_feature_slice = apply_exponential_smoothing(test_feature_slice, alpha=0.3)

    X_train = np.reshape(train_feature_slice, (-1, train_feature_slice.shape[1], 1, train_feature_slice.shape[2]))
    train_label_slice[train_label_slice > 115] = 115
    Y_train = train_label_slice

    X_test = np.reshape(test_feature_slice, (-1, test_feature_slice.shape[1], 1, test_feature_slice.shape[2]))
    test_label_slice[test_label_slice > 115] = 115
    Y_test = test_label_slice

    print("X_train.shape: {}".format(X_train.shape))
    print("Y_train.shape: {}".format(Y_train.shape))

    # --- Dual Pooling Model Architecture ---
    def DualPooling_Model():
        in0 = keras.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        
        # Block 1 - Wide Kernel
        x = keras.layers.Conv2D(num_filter1, (kernel1_size, 1), strides=1, padding="same")(in0)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Block 2
        x = keras.layers.Conv2D(num_filter2, (kernel2_size, 1), strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Block 3
        x = keras.layers.Conv2D(num_filter3, (kernel3_size, 1), strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # --- Dual Pooling Strategy ---
        # Average captures the general degradation trend.
        # Max captures the sharpest fault signatures/noise spikes.
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        max_pool = keras.layers.GlobalMaxPooling2D()(x)
        
        # Concatenate features
        x = keras.layers.Concatenate()([avg_pool, max_pool])
        
        # --- Regularized Head ---
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x) # Dropout is critical for large batch size
        
        out = keras.layers.Dense(1, activation="relu")(x)
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
        model_dir = os.path.join(last_path, "model", "Physics", f"FD{FD}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        method_error_path = os.path.join(err_dir, f"{method_name}.txt")
        if os.path.exists(method_error_path):
            os.remove(method_error_path)

        for i in range(run_times):
            print(f"--- Run {i+1}/{run_times} ---")
            model = DualPooling_Model()
            
            # Use Adam with a slightly higher learning rate for Batch 1024
            optimizer = keras.optimizers.Adam(learning_rate=0.003)
            
            model.compile(loss=physics_loss, optimizer=optimizer, 
                          metrics=[keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")])

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=patience_reduce_lr, min_lr=0.0001
            )
            model_name = "{}_dataset_{}_run{}".format(method_name, dataset, i)
            model_path_full = os.path.join(model_dir, f"{model_name}.h5")
            already_trained = False
            
            if os.path.exists(model_path_full):
                print(f"Found existing checkpoint {model_path_full}, loading weights.")
                model = keras.models.load_model(
                    model_path_full,
                    custom_objects={"physics_loss": physics_loss},
                )
                already_trained = True
            else:
                earlystopping = keras.callbacks.EarlyStopping(monitor="loss", patience=patience, verbose=1)
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
                    shuffle=False,
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
                plt.title("Training and Validation loss")
                plt.legend()
                plt.show()

                model = keras.models.load_model(
                    model_path_full,
                    custom_objects={"physics_loss": physics_loss},
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

            if not already_trained:
                try:
                    index_min_val_loss = log["loss"].idxmin()
                    min_val_loss = log["val_loss"].iloc[index_min_val_loss]
                except:
                     index_min_val_loss, min_val_loss = -1, -1
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
        
        if error_record:
            mean_rmse = np.mean(error_record)
            mean_upe = np.mean(unbalanced_penalty_score_record)
            mean_er_l = np.mean(error_range_left_record)
            mean_er_r = np.mean(error_range_right_record)
            mean_mono = np.mean(mono_violation_record)
            mean_slope = np.mean(slope_rmse_record)
        else:
            mean_rmse = mean_upe = mean_er_l = mean_er_r = mean_mono = mean_slope = 0.0

        with open(method_error_path, "a") as file:
            file.write(
                "    mean_score:" + "     (" + str(mean_rmse) + ")     "
                + "       "
                + "     mean_RMSE:   " + str("%.6f" % (mean_rmse))
                + "     UPE:    " + str("%.6f" % (mean_upe))
                + "    (" + str("%.6f" % (mean_er_l))
                + "," + str("%.6f" % (mean_er_r))
                + ")"
                + "    MonoViolation: " + str("%.6f" % (mean_mono))
                + "    SlopeRMSE: " + str("%.6f" % (mean_slope))
                + "        " + "\n"
            )

        fd_summary[FD] = {
            "rmse": mean_rmse if error_record else float('nan'),
            "upe": mean_upe if unbalanced_penalty_score_record else float('nan'),
            "mono": mean_mono if mono_violation_record else float('nan'),
            "slope": mean_slope if slope_rmse_record else float('nan'),
        }

        error_record = []
        unbalanced_penalty_score_record = []
        error_range_left_record = []
        error_range_right_record = []
        mono_violation_record = []
        slope_rmse_record = []

if __name__ == "__main__" and fd_summary:
    print("\n=== Cross-FD Summary ===")
    for fd_key in sorted(fd_summary.keys()):
        s = fd_summary[fd_key]
        print(f"FD{fd_key}: RMSE={s['rmse']:.4f}, UPE={s['upe']:.2f}, Mono={s['mono']:.6f}, Slope={s['slope']:.4f}")

    summary_csv_path = os.path.join(last_path, "experiments_result", "physics_fd_summary.csv")
    summary_df = pd.DataFrame([
        {"FD": f"FD{k}", "RMSE": v["rmse"], "UPE": v["upe"], "MonoViolation": v["mono"], "SlopeRMSE": v["slope"]}
        for k, v in fd_summary.items()
    ])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV to {summary_csv_path}")

    fig_dir = os.path.join(last_path, "figure", "physics_summary")
    os.makedirs(fig_dir, exist_ok=True)
    fds = [f"FD{k}" for k in sorted(fd_summary.keys())]
    metrics = {"RMSE": "rmse", "UPE": "upe", "MonoViolation": "mono", "SlopeRMSE": "slope"}
    for metric_label, metric_key in metrics.items():
        vals = [fd_summary[k][metric_key] for k in sorted(fd_summary.keys())]
        plt.figure(figsize=(5, 4))
        plt.bar(fds, vals, color="#4c72b0")
        plt.title(f"{metric_label} across FD datasets")
        plt.ylabel(metric_label)
        plt.tight_layout()
        out_path = os.path.join(fig_dir, f"{metric_label}_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")