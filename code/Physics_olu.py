# -*- coding: utf-8 -*-
"""
Physics-informed FCN for CMAPSS RUL (Dual Pooling + Asymmetric Loss)
Optimised for FD001/FD002 with vectorised operations and FD-specific tuning.
"""
import os
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – avoids plt.show() blocking
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
from scipy.signal import lfilter, lfilter_zi
import CMAPSSDataset

# ---------- GPU memory growth (prevents OOM on multi-model runs) ----------
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------- Reproducibility ----------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# Default weights (used when loading legacy models)
physics_alpha = 0.001   # Monotonicity weight (Low)
physics_gamma = 0.001   # Slope weight (Low)
late_prediction_weight = 0.2  # Asymmetric penalty weight

def _diff_mask(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    same_engine_mask = K.cast(true_diffs < 0.0, K.floatx())
    return true_diffs, pred_diffs, same_engine_mask

@register_keras_serializable()
def physics_loss(y_true, y_pred):
    """Legacy loss with default weights – kept for backward-compatible model loading."""
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    mse = K.mean(K.square(y_true_flat - y_pred_flat))
    over_estimation = K.relu(y_pred_flat - y_true_flat)
    asymmetric_loss = K.mean(K.square(over_estimation))
    true_diffs, pred_diffs, same_engine_mask = _diff_mask(y_true_flat, y_pred_flat)
    masked_pred_diffs = pred_diffs * same_engine_mask
    mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(same_engine_mask) + K.epsilon())
    slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * same_engine_mask)) / (
        K.sum(same_engine_mask) + K.epsilon()
    )
    return mse + (0.2 * asymmetric_loss) + (0.001 * mono_penalty) + (0.001 * slope_penalty)


def make_physics_loss(asym_w=0.2, mono_w=0.001, slope_w=0.001):
    """Factory: create a physics loss with FD-specific weights.

    TensorFlow traces loss functions into a static graph at compile time,
    so plain Python globals would be frozen. Closure variables here are
    correctly captured per-FD.
    """
    def _loss(y_true, y_pred):
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        mse = K.mean(K.square(y_true_flat - y_pred_flat))
        over_estimation = K.relu(y_pred_flat - y_true_flat)
        asymmetric_loss = K.mean(K.square(over_estimation))
        true_diffs, pred_diffs, same_engine_mask = _diff_mask(y_true_flat, y_pred_flat)
        masked_pred_diffs = pred_diffs * same_engine_mask
        mono_penalty = K.sum(K.relu(masked_pred_diffs)) / (K.sum(same_engine_mask) + K.epsilon())
        slope_penalty = K.sum(K.square((pred_diffs - true_diffs) * same_engine_mask)) / (
            K.sum(same_engine_mask) + K.epsilon()
        )
        return mse + (asym_w * asymmetric_loss) + (mono_w * mono_penalty) + (slope_w * slope_penalty)
    _loss.__name__ = "physics_loss"  # for Keras serialisation compatibility
    return _loss

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

# --- Helper: Exponential Smoothing (vectorised with scipy.signal.lfilter) ---
def apply_exponential_smoothing(data, alpha=0.1):
    """Vectorised EMA along the time axis using scipy IIR filter.

    Matches the original formula exactly:
        s[0] = x[0]   (first value preserved)
        s[t] = alpha*x[t] + (1-alpha)*s[t-1]

    Uses lfilter_zi to compute the proper initial conditions so that
    y[0] = x[0] instead of alpha*x[0].
    """
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = lfilter_zi(b, a)              # shape (1,) — gives [1.0] for EMA
    smoothed = np.empty_like(data)
    n_samples, n_time, n_feat = data.shape
    for j in range(n_feat):            # loop only over features (small)
        x_2d = data[:, :, j].astype(np.float64)   # (n_samples, n_time)
        # zi * x[:,0:1] sets initial state so y[0] = x[0] for every sample
        zi_2d = zi * x_2d[:, 0:1]                 # (n_samples, 1)
        smoothed[:, :, j], _ = lfilter(b, a, x_2d, axis=1, zi=zi_2d)
    return smoothed.astype(np.float32)


# --- Squeeze-and-Excitation (SE) channel attention ---
def se_block(x, ratio=8):
    """Squeeze-and-Excitation: learns per-channel importance weights.

    Adaptively recalibrates feature maps so the network can focus on the
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


# --- FD-specific hyperparameter table (methodology unchanged) ---
# FD001/002: large batch + higher LR works well (simpler operating conditions)
# FD003/004: smaller batch + lower LR for multi-condition complexity
# Loss weights: carefully tuned so that each added component improves RMSE.
#   - asym kept LOW to avoid prediction bias
#   - slope << mono so monotonicity dominates but slope refines
FD_HPARAMS = {
    "1": {"batch_size": 1024, "lr": 0.003, "dropout": 0.5, "smoothing_alpha": 0.3,
           "late_prediction_weight": 0.05, "physics_alpha": 0.001, "physics_gamma": 0.0003},
    "2": {"batch_size": 1024, "lr": 0.003, "dropout": 0.5, "smoothing_alpha": 0.3,
           "late_prediction_weight": 0.05, "physics_alpha": 0.001, "physics_gamma": 0.0003},
    "3": {"batch_size": 256,  "lr": 0.001, "dropout": 0.3, "smoothing_alpha": 0.15,
           "late_prediction_weight": 0.08, "physics_alpha": 0.002, "physics_gamma": 0.0005},
    "4": {"batch_size": 256,  "lr": 0.001, "dropout": 0.3, "smoothing_alpha": 0.15,
           "late_prediction_weight": 0.08, "physics_alpha": 0.002, "physics_gamma": 0.0005},
}

# --- Vectorised scoring & error-range (moved outside loop) ---

def unbalanced_penalty_score_1out(Y_test, Y_pred):
    """NASA asymmetric scoring – fully vectorised (no Python loop)."""
    yt = Y_test.flatten().astype(np.float64)
    yp = Y_pred.flatten().astype(np.float64)
    diff = yp - yt
    over  = np.exp( diff / 10.0) - 1.0    # pred > true
    under = np.exp(-diff / 13.0) - 1.0    # pred <= true
    s = float(np.where(diff > 0, over, under).sum())
    print(f"unbalanced_penalty_score: {s:.4f}")
    return s


def error_range_1out(Y_test, Y_pred):
    err = Y_test - Y_pred
    er = (float(err.min()), float(err.max()))
    print(f"error range: {er}")
    return er


fd_summary = {}

for FD in ['1', '2']:  # FD001 and FD002 only
    # ---------- FD-specific dataset config ----------
    if FD == "1":
        sequence_length = 31
        FD_feature_columns = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    elif FD == "2":
        sequence_length = 21
        FD_feature_columns = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
    elif FD == "3":
        sequence_length = 38
        FD_feature_columns = ["s2", "s3", "s4", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    elif FD == "4":
        sequence_length = 19
        FD_feature_columns = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s20", "s21"]
    else:
        raise ValueError(f"Unsupported FD: {FD}")

    # ---------- FD-tuned hyperparameters ----------
    hp = FD_HPARAMS[FD]
    batch_size = hp["batch_size"]
    fd_lr = hp["lr"]
    fd_dropout = hp["dropout"]
    fd_smooth_alpha = hp["smoothing_alpha"]

    # Create FD-specific loss function (closure captures weights correctly)
    fd_loss = make_physics_loss(
        asym_w=hp["late_prediction_weight"],
        mono_w=hp["physics_alpha"],
        slope_w=hp["physics_gamma"],
    )
    print(f"  Loss weights: asym={hp['late_prediction_weight']}, "
          f"mono={hp['physics_alpha']}, slope={hp['physics_gamma']}")

    method_name = "DualPooling_AsymLoss_FD{}_without_num_test{}".format(FD, num_test)
    dataset = "cmapssd"

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
    elif num_test == 10000:
        test_feature_slice = datasets.get_feature_slice(test_data)
        test_label_slice = datasets.get_label_slice(test_data)

    # --- APPLY SMOOTHING (vectorised) ---
    print(f"Applying Exponential Smoothing (alpha={fd_smooth_alpha})...")
    train_feature_slice = apply_exponential_smoothing(train_feature_slice, alpha=fd_smooth_alpha)
    test_feature_slice = apply_exponential_smoothing(test_feature_slice, alpha=fd_smooth_alpha)

    X_train = np.reshape(train_feature_slice, (-1, train_feature_slice.shape[1], 1, train_feature_slice.shape[2]))
    train_label_slice[train_label_slice > 115] = 115
    Y_train = train_label_slice

    X_test = np.reshape(test_feature_slice, (-1, test_feature_slice.shape[1], 1, test_feature_slice.shape[2]))
    test_label_slice[test_label_slice > 115] = 115
    Y_test = test_label_slice

    print("X_train.shape: {}".format(X_train.shape))
    print("Y_train.shape: {}".format(Y_train.shape))

    # --- Dual Pooling Model Architecture (with SE Attention) ---
    def DualPooling_Model():
        in0 = keras.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        
        # Block 1 - Wide Kernel + SE attention
        x = keras.layers.Conv2D(num_filter1, (kernel1_size, 1), strides=1, padding="same")(in0)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = se_block(x)
        
        # Block 2 + SE attention
        x = keras.layers.Conv2D(num_filter2, (kernel2_size, 1), strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = se_block(x)
        
        # Block 3 + SE attention
        x = keras.layers.Conv2D(num_filter3, (kernel3_size, 1), strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = se_block(x)
        
        # --- Dual Pooling Strategy ---
        # Average captures the general degradation trend.
        # Max captures the sharpest fault signatures/noise spikes.
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        max_pool = keras.layers.GlobalMaxPooling2D()(x)
        
        # Concatenate features
        x = keras.layers.Concatenate()([avg_pool, max_pool])
        
        # --- Regularized Head (FD-tuned dropout) ---
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(fd_dropout)(x)
        
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
            
            # Use float LR + cosine-annealing callback (avoids CosineDecay
            # schedule object that causes TypeError on save/restore).
            optimizer = keras.optimizers.Adam(learning_rate=float(fd_lr), clipnorm=1.0)
            
            model.compile(loss=fd_loss, optimizer=optimizer, 
                          metrics=[keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")])

            # Cosine annealing callback
            _total_ep = int(nb_epochs)
            _init_lr = float(fd_lr)
            _min_lr = 1e-5
            def _cos_lr(epoch, lr):
                return _min_lr + 0.5 * (_init_lr - _min_lr) * (1 + math.cos(math.pi * epoch / _total_ep))

            model_name = "{}_dataset_{}_run{}".format(method_name, dataset, i)
            model_path_full = os.path.join(model_dir, f"{model_name}.h5")
            already_trained = False
            
            if os.path.exists(model_path_full):
                print(f"Found existing checkpoint {model_path_full}, loading weights.")
                model = keras.models.load_model(
                    model_path_full,
                    custom_objects={"physics_loss": fd_loss},
                )
                already_trained = True
            else:
                earlystopping = keras.callbacks.EarlyStopping(monitor="loss", patience=patience, verbose=1)
                cosine_lr_cb = keras.callbacks.LearningRateScheduler(_cos_lr, verbose=0)
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
                    callbacks=[cosine_lr_cb, earlystopping, modelcheckpoint],
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
                plt.title(f"Training and Validation loss – FD00{FD}")
                plt.legend()
                fig_out_dir = os.path.join(last_path, "figure", "physics_summary")
                os.makedirs(fig_out_dir, exist_ok=True)
                plt.savefig(os.path.join(fig_out_dir, f"loss_curve_FD{FD}.png"), dpi=300, bbox_inches="tight")
                plt.close()

                model = keras.models.load_model(
                    model_path_full,
                    custom_objects={"physics_loss": fd_loss},
                )

            # Inference only – no need to modify trainable flags for predict()
            Y_pred = model.predict(X_test, batch_size=batch_size)
            
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