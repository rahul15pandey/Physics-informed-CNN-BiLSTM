"""
Generic Time-Series Dataset Loader for Physics-Informed RUL / Degradation Prediction.

Supports:
  1. CSV files with configurable column names (unit_id, cycle, target, features)
  2. Direct NumPy arrays (X_train, Y_train, X_test, Y_test)
  3. Built-in adapters for popular prognostics benchmarks:
     - NASA Bearing Dataset (IMS / FEMTO)
     - PHM 2012 Prognostics Challenge (PRONOSTIA bearings)
     - Battery (capacity fade as RUL proxy)
     - Turbofan Degradation Simulation (any CMAPSS-like format)
  4. Synthetic degradation data for quick sanity checks

The loader mirrors the CMAPSSDataset API (windowed sequences + labels) so it
plugs directly into the PI-DP-FCN training pipeline.
"""

import os
import glob
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================================================================
# Helper: sliding-window segmentation
# =========================================================================

def _sliding_window(data: np.ndarray, labels: np.ndarray,
                    seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows of length `seq_len`.

    Args:
        data:   (total_timesteps, n_features)
        labels: (total_timesteps,)  – target at each timestep
        seq_len: window length

    Returns:
        X: (n_windows, seq_len, n_features)
        Y: (n_windows, 1)  – label at the last timestep of each window
    """
    n = data.shape[0]
    if n < seq_len:
        raise ValueError(f"Series length {n} < seq_len {seq_len}")
    X, Y = [], []
    for i in range(n - seq_len + 1):
        X.append(data[i:i + seq_len])
        Y.append(labels[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)


# =========================================================================
# Main class
# =========================================================================

class GenericTimeSeriesDataset:
    """Flexible loader that produces windowed (samples, timesteps, 1, features)
    arrays matching the PI-DP-FCN input convention."""

    def __init__(
        self,
        name: str = "custom",
        sequence_length: int = 30,
        rul_cap: float = 125.0,
        test_last_only: bool = True,
    ):
        self.name = name
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap
        self.test_last_only = test_last_only

        self.scaler = StandardScaler()

        # Populated by load_* methods
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.Y_test: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    # -----------------------------------------------------------------
    # Public API: get ready-to-use 4-D arrays
    # -----------------------------------------------------------------

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, Y_train, X_test, Y_test) in PI-DP-FCN shape:
        X: (samples, seq_len, 1, n_features)
        Y: (samples, 1)
        """
        assert self.X_train is not None, "Data not loaded. Call a load_* method first."
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    # -----------------------------------------------------------------
    # 1. From CSV with unit_id column
    # -----------------------------------------------------------------

    def load_from_csv(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        test_rul_path: Optional[str] = None,
        unit_col: str = "unit_id",
        cycle_col: str = "cycle",
        target_col: str = "RUL",
        feature_cols: Optional[List[str]] = None,
        delimiter: str = ",",
        train_ratio: float = 0.8,
    ):
        """Load from CSV files.

        If `test_path` is None, the training CSV is split into train/test
        by unit_id using `train_ratio`.

        If the CSV has no pre-computed RUL column, provide `target_col=None`
        and the loader will compute piece-wise linear RUL from max cycle.

        If `test_rul_path` is given (like C-MAPSS RUL_FDxxx.txt), it is used
        to compute the true RUL for test units.
        """
        train_df = pd.read_csv(train_path, delimiter=delimiter)

        # Auto-detect feature columns
        exclude = {unit_col, cycle_col, target_col} if target_col else {unit_col, cycle_col}
        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c not in exclude]
        self.feature_names = list(feature_cols)

        # Compute RUL if not present
        if target_col is None or target_col not in train_df.columns:
            target_col = "RUL"
            max_cycles = train_df.groupby(unit_col)[cycle_col].transform("max")
            train_df[target_col] = max_cycles - train_df[cycle_col]

        # Split into train / test if no separate test file
        if test_path is None:
            unit_ids = train_df[unit_col].unique()
            np.random.seed(42)
            np.random.shuffle(unit_ids)
            split = int(len(unit_ids) * train_ratio)
            train_ids, test_ids = unit_ids[:split], unit_ids[split:]
            test_df = train_df[train_df[unit_col].isin(test_ids)].copy()
            train_df = train_df[train_df[unit_col].isin(train_ids)].copy()
        else:
            test_df = pd.read_csv(test_path, delimiter=delimiter)
            if target_col not in test_df.columns:
                if test_rul_path is not None:
                    truth = pd.read_csv(test_rul_path, delimiter=r"\s+", header=None)
                    truth.columns = ["truth"]
                    truth[unit_col] = truth.index + 1
                    test_rul_info = test_df.groupby(unit_col)[cycle_col].max().reset_index()
                    test_rul_info.columns = [unit_col, "elapsed"]
                    test_rul_info = test_rul_info.merge(truth, on=unit_col, how="left")
                    test_rul_info["max"] = test_rul_info["elapsed"] + test_rul_info["truth"]
                    test_df = test_df.merge(test_rul_info[[unit_col, "max"]], on=unit_col, how="left")
                    test_df[target_col] = test_df["max"] - test_df[cycle_col]
                    test_df.drop("max", axis=1, inplace=True)
                else:
                    max_cycles = test_df.groupby(unit_col)[cycle_col].transform("max")
                    test_df[target_col] = max_cycles - test_df[cycle_col]

        # Normalize features (fit on train)
        self.scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

        # Cap RUL
        if self.rul_cap is not None:
            train_df.loc[train_df[target_col] > self.rul_cap, target_col] = self.rul_cap
            test_df.loc[test_df[target_col] > self.rul_cap, target_col] = self.rul_cap

        # Window per unit
        self._window_per_unit(train_df, test_df, unit_col, feature_cols, target_col)

    # -----------------------------------------------------------------
    # 2. From NumPy arrays (already windowed or not)
    # -----------------------------------------------------------------

    def load_from_numpy(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        already_windowed: bool = False,
        feature_names: Optional[List[str]] = None,
    ):
        """Load directly from NumPy arrays.

        If `already_windowed=True`, expects X shape (samples, seq_len, features).
        Otherwise expects X shape (total_timesteps, features) and will apply
        sliding-window segmentation.
        """
        if feature_names:
            self.feature_names = feature_names

        if already_windowed:
            # Reshape to 4-D: (samples, seq_len, 1, features)
            if X_train.ndim == 3:
                X_train = X_train[:, :, np.newaxis, :]
            if X_test.ndim == 3:
                X_test = X_test[:, :, np.newaxis, :]
            self.X_train = X_train.astype(np.float32)
            self.Y_train = Y_train.astype(np.float32).reshape(-1, 1)
            self.X_test = X_test.astype(np.float32)
            self.Y_test = Y_test.astype(np.float32).reshape(-1, 1)
        else:
            # Apply sliding window
            Xtr, Ytr = _sliding_window(X_train, Y_train.flatten(), self.sequence_length)
            Xte, Yte = _sliding_window(X_test, Y_test.flatten(), self.sequence_length)
            self.X_train = Xtr[:, :, np.newaxis, :]
            self.Y_train = Ytr
            self.X_test = Xte[:, :, np.newaxis, :]
            self.Y_test = Yte

        # Normalise
        n_feat = self.X_train.shape[-1]
        flat_train = self.X_train.reshape(-1, n_feat)
        self.scaler.fit(flat_train)
        self.X_train = self.scaler.transform(
            self.X_train.reshape(-1, n_feat)
        ).reshape(self.X_train.shape).astype(np.float32)
        self.X_test = self.scaler.transform(
            self.X_test.reshape(-1, n_feat)
        ).reshape(self.X_test.shape).astype(np.float32)

        # Cap RUL
        if self.rul_cap is not None:
            self.Y_train[self.Y_train > self.rul_cap] = self.rul_cap
            self.Y_test[self.Y_test > self.rul_cap] = self.rul_cap

    # -----------------------------------------------------------------
    # 3. Synthetic degradation data (for quick experiments)
    # -----------------------------------------------------------------

    def load_synthetic(
        self,
        n_units_train: int = 80,
        n_units_test: int = 20,
        max_life: int = 200,
        n_features: int = 14,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        """Generate synthetic multi-sensor degradation data.

        Each unit has a random lifetime ∈ [max_life//2, max_life].
        Features follow exponential degradation curves with Gaussian noise.
        """
        rng = np.random.RandomState(seed)
        self.feature_names = [f"sensor_{i+1}" for i in range(n_features)]

        def _gen_units(n_units):
            all_X, all_Y = [], []
            for _ in range(n_units):
                life = rng.randint(max_life // 2, max_life + 1)
                t = np.linspace(0, 1, life)
                features = np.zeros((life, n_features))
                for f in range(n_features):
                    rate = rng.uniform(1.0, 5.0)
                    amp = rng.uniform(0.5, 2.0)
                    features[:, f] = amp * (np.exp(rate * t) - 1) + rng.randn(life) * noise_std
                rul = np.arange(life - 1, -1, -1, dtype=np.float32)
                all_X.append(features)
                all_Y.append(rul)
            return all_X, all_Y

        train_X_list, train_Y_list = _gen_units(n_units_train)
        test_X_list, test_Y_list = _gen_units(n_units_test)

        # Create windowed samples
        X_tr, Y_tr = [], []
        for feats, rul in zip(train_X_list, train_Y_list):
            if len(rul) >= self.sequence_length:
                xw, yw = _sliding_window(feats, rul, self.sequence_length)
                X_tr.append(xw)
                Y_tr.append(yw)

        X_te, Y_te = [], []
        for feats, rul in zip(test_X_list, test_Y_list):
            if len(rul) >= self.sequence_length:
                if self.test_last_only:
                    # Only the last window per unit
                    xw, yw = _sliding_window(feats, rul, self.sequence_length)
                    X_te.append(xw[-1:])
                    Y_te.append(yw[-1:])
                else:
                    xw, yw = _sliding_window(feats, rul, self.sequence_length)
                    X_te.append(xw)
                    Y_te.append(yw)

        X_train = np.concatenate(X_tr, axis=0)
        Y_train = np.concatenate(Y_tr, axis=0)
        X_test = np.concatenate(X_te, axis=0)
        Y_test = np.concatenate(Y_te, axis=0)

        # Normalise
        n_feat = X_train.shape[-1]
        flat = X_train.reshape(-1, n_feat)
        self.scaler.fit(flat)
        X_train = self.scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

        # Cap RUL
        if self.rul_cap:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        # 4-D
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 4. NASA Bearing (IMS) adapter
    # -----------------------------------------------------------------

    def load_nasa_bearing(self, data_dir: str, test_ratio: float = 0.2):
        """Load NASA IMS Bearing dataset.

        Expected structure:  data_dir/<set_name>/<channel_files>
        Each file has one vibration reading per row (sampled at 20 kHz, 1-sec records).
        We extract statistical features per record and compute RUL from remaining records.
        """
        self.feature_names = ["rms", "kurtosis", "crest_factor", "skewness",
                              "peak", "std", "peak_to_peak", "shape_factor"]

        def _extract_features(filepath: str) -> np.ndarray:
            """Extract 8 time-domain features from a raw vibration file."""
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            feats = []
            for col in range(data.shape[1]):
                sig = data[:, col]
                rms = np.sqrt(np.mean(sig ** 2))
                kurt = float(pd.Series(sig).kurtosis())
                peak = np.max(np.abs(sig))
                crest = peak / (rms + 1e-10)
                skew = float(pd.Series(sig).skew())
                std = np.std(sig)
                p2p = np.max(sig) - np.min(sig)
                shape = rms / (np.mean(np.abs(sig)) + 1e-10)
                feats.extend([rms, kurt, crest, skew, peak, std, p2p, shape])
            return np.array(feats, dtype=np.float32)

        # Discover sets
        sets = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
        if not sets:
            raise FileNotFoundError(f"No sub-directories found in {data_dir}")

        all_features, all_rul = [], []
        for set_name in sets:
            set_dir = os.path.join(data_dir, set_name)
            files = sorted(glob.glob(os.path.join(set_dir, "*")))
            if not files:
                continue
            feat_seq = np.array([_extract_features(f) for f in files])
            n = len(feat_seq)
            rul = np.arange(n - 1, -1, -1, dtype=np.float32)
            all_features.append(feat_seq)
            all_rul.append(rul)

        # Split by bearing sets
        n_sets = len(all_features)
        n_test = max(1, int(n_sets * test_ratio))
        train_feats = all_features[:n_sets - n_test]
        train_ruls = all_rul[:n_sets - n_test]
        test_feats = all_features[n_sets - n_test:]
        test_ruls = all_rul[n_sets - n_test:]

        # Window & concatenate
        def _window_concat(feat_list, rul_list, last_only=False):
            Xs, Ys = [], []
            for f, r in zip(feat_list, rul_list):
                if len(r) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(f, r, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _window_concat(train_feats, train_ruls)
        X_test, Y_test = _window_concat(test_feats, test_ruls, last_only=self.test_last_only)

        self.load_from_numpy(X_train, Y_train, X_test, Y_test,
                             already_windowed=True, feature_names=self.feature_names)

    # -----------------------------------------------------------------
    # 5. Battery capacity fade
    # -----------------------------------------------------------------

    def load_battery(
        self,
        csv_path: str,
        cycle_col: str = "cycle",
        capacity_col: str = "capacity",
        cell_col: str = "cell_id",
        eol_threshold: float = 0.7,
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ):
        """Load battery degradation data.

        RUL is defined as remaining cycles until capacity drops below
        `eol_threshold` fraction of initial capacity.

        Args:
            csv_path: CSV with cycle-level battery data
            eol_threshold: fraction of initial capacity defining end-of-life
            feature_cols: columns to use as features (default: all except id/cycle/capacity)
        """
        df = pd.read_csv(csv_path)

        exclude = {cell_col, cycle_col, capacity_col}
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in exclude]
        # Always include capacity as a feature
        if capacity_col not in feature_cols:
            feature_cols = [capacity_col] + feature_cols
        self.feature_names = list(feature_cols)

        # Compute RUL per cell
        cells = df[cell_col].unique()
        for cell in cells:
            mask = df[cell_col] == cell
            cell_data = df.loc[mask].sort_values(cycle_col)
            init_cap = cell_data[capacity_col].iloc[0]
            threshold = init_cap * eol_threshold

            # Find EOL cycle
            eol_mask = cell_data[capacity_col] < threshold
            if eol_mask.any():
                eol_cycle = cell_data.loc[eol_mask, cycle_col].iloc[0]
            else:
                eol_cycle = cell_data[cycle_col].max()

            df.loc[mask, "RUL"] = eol_cycle - df.loc[mask, cycle_col]
            df.loc[mask & (df["RUL"] < 0), "RUL"] = 0

        self.load_from_csv(
            train_path=csv_path,
            unit_col=cell_col,
            cycle_col=cycle_col,
            target_col="RUL",
            feature_cols=feature_cols,
            train_ratio=train_ratio,
        )

    # -----------------------------------------------------------------
    # 6. PHM 2012 (PRONOSTIA) Bearing adapter
    # -----------------------------------------------------------------

    def load_phm2012(
        self,
        data_dir: str,
        condition: str = "1",
        train_bearings: Optional[List[str]] = None,
        test_bearings: Optional[List[str]] = None,
    ):
        """Load PHM 2012 PRONOSTIA bearing dataset.

        Expected structure:
          data_dir/Learning_set/Bearing<cond>_<id>/acc_<timestamp>.csv
          data_dir/Test_set/Bearing<cond>_<id>/acc_<timestamp>.csv

        Each acc CSV has columns: hour, minute, second, microsecond, h_acc, v_acc.
        We extract statistical features from h_acc and v_acc per file (one snapshot).
        """
        self.feature_names = [
            "h_rms", "h_kurtosis", "h_crest", "h_skew", "h_peak", "h_std",
            "v_rms", "v_kurtosis", "v_crest", "v_skew", "v_peak", "v_std",
        ]

        def _extract_phm_features(filepath: str) -> np.ndarray:
            df = pd.read_csv(filepath, header=None)
            feats = []
            for col_idx in [4, 5]:  # h_acc, v_acc
                sig = df.iloc[:, col_idx].values.astype(np.float64)
                rms = np.sqrt(np.mean(sig ** 2))
                kurt = float(pd.Series(sig).kurtosis())
                peak = np.max(np.abs(sig))
                crest = peak / (rms + 1e-10)
                skew = float(pd.Series(sig).skew())
                std = np.std(sig)
                feats.extend([rms, kurt, crest, skew, peak, std])
            return np.array(feats, dtype=np.float32)

        def _load_bearing_set(base_dir: str, bearing_ids: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            feat_list, rul_list = [], []
            for bid in bearing_ids:
                bearing_dir = os.path.join(base_dir, f"Bearing{condition}_{bid}")
                if not os.path.isdir(bearing_dir):
                    print(f"  Warning: {bearing_dir} not found, skipping.")
                    continue
                files = sorted(glob.glob(os.path.join(bearing_dir, "acc_*.csv")))
                if not files:
                    continue
                feats = np.array([_extract_phm_features(f) for f in files])
                rul = np.arange(len(feats) - 1, -1, -1, dtype=np.float32)
                feat_list.append(feats)
                rul_list.append(rul)
            return feat_list, rul_list

        # Default bearing splits
        if train_bearings is None:
            train_bearings = ["1", "2"]
        if test_bearings is None:
            test_bearings = ["3"]

        learn_dir = os.path.join(data_dir, "Learning_set")
        test_dir = os.path.join(data_dir, "Test_set") if os.path.isdir(
            os.path.join(data_dir, "Test_set")) else learn_dir

        train_feats, train_ruls = _load_bearing_set(learn_dir, train_bearings)
        test_feats, test_ruls = _load_bearing_set(test_dir, test_bearings)

        # Window & concatenate
        def _wc(flist, rlist, last_only=False):
            Xs, Ys = [], []
            for f, r in zip(flist, rlist):
                if len(r) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(f, r, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            if not Xs:
                raise ValueError("No windows could be created. Check data or reduce seq_len.")
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _wc(train_feats, train_ruls)
        X_test, Y_test = _wc(test_feats, test_ruls, last_only=self.test_last_only)

        self.load_from_numpy(X_train, Y_train, X_test, Y_test,
                             already_windowed=True, feature_names=self.feature_names)

    # -----------------------------------------------------------------
    # Internal: window per unit from DataFrames
    # -----------------------------------------------------------------

    def _window_per_unit(self, train_df, test_df, unit_col, feature_cols, target_col):
        """Create windowed arrays from train/test DataFrames with a unit column."""
        def _make_windows(df, last_only=False):
            Xs, Ys = [], []
            for uid in sorted(df[unit_col].unique()):
                unit_data = df[df[unit_col] == uid].sort_values(
                    by=[c for c in df.columns if "cycle" in c.lower()] or df.columns[:2]
                )
                feats = unit_data[feature_cols].values
                labels = unit_data[target_col].values
                if len(labels) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(feats, labels, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            if not Xs:
                raise ValueError("No windows created. Lower sequence_length or check data.")
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _make_windows(train_df, last_only=False)
        X_test, Y_test = _make_windows(test_df, last_only=self.test_last_only)

        # Reshape to 4-D: (samples, seq_len, 1, features)
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  Sequence length: {self.sequence_length}",
            f"  RUL cap: {self.rul_cap}",
            f"  Features ({len(self.feature_names)}): {self.feature_names[:10]}{'...' if len(self.feature_names) > 10 else ''}",
        ]
        if self.X_train is not None:
            lines.append(f"  X_train: {self.X_train.shape}  Y_train: {self.Y_train.shape}")
            lines.append(f"  X_test:  {self.X_test.shape}   Y_test:  {self.Y_test.shape}")
            lines.append(f"  Y_train range: [{self.Y_train.min():.1f}, {self.Y_train.max():.1f}]")
            lines.append(f"  Y_test  range: [{self.Y_test.min():.1f}, {self.Y_test.max():.1f}]")
        else:
            lines.append("  [Data not yet loaded]")
        return "\n".join(lines)


# =========================================================================
# Config-driven loader
# =========================================================================

def load_dataset_from_config(config: Dict) -> GenericTimeSeriesDataset:
    """Create and populate a GenericTimeSeriesDataset from a config dict.

    Example configs:

    Synthetic:
        {"type": "synthetic", "name": "synth", "sequence_length": 30,
         "n_units_train": 80, "n_units_test": 20, "max_life": 200}

    CSV:
        {"type": "csv", "name": "my_turbines", "sequence_length": 30,
         "train_path": "data/train.csv", "unit_col": "turbine_id",
         "cycle_col": "timestamp", "target_col": "RUL"}

    NASA Bearing:
        {"type": "nasa_bearing", "data_dir": "data/IMS", "sequence_length": 50}

    Battery:
        {"type": "battery", "csv_path": "data/battery.csv",
         "sequence_length": 20, "eol_threshold": 0.7}

    PHM 2012:
        {"type": "phm2012", "data_dir": "data/PHM2012", "condition": "1",
         "sequence_length": 40}

    NumPy:
        {"type": "numpy", "train_X_path": "data/X_train.npy",
         "train_Y_path": "data/Y_train.npy", ...}
    """
    ds_type = config.get("type", "synthetic")
    seq_len = config.get("sequence_length", 30)
    rul_cap = config.get("rul_cap", 125.0)
    test_last = config.get("test_last_only", True)
    name = config.get("name", ds_type)

    ds = GenericTimeSeriesDataset(
        name=name, sequence_length=seq_len,
        rul_cap=rul_cap, test_last_only=test_last,
    )

    if ds_type == "synthetic":
        ds.load_synthetic(
            n_units_train=config.get("n_units_train", 80),
            n_units_test=config.get("n_units_test", 20),
            max_life=config.get("max_life", 200),
            n_features=config.get("n_features", 14),
            noise_std=config.get("noise_std", 0.3),
            seed=config.get("seed", 42),
        )

    elif ds_type == "csv":
        ds.load_from_csv(
            train_path=config["train_path"],
            test_path=config.get("test_path"),
            test_rul_path=config.get("test_rul_path"),
            unit_col=config.get("unit_col", "unit_id"),
            cycle_col=config.get("cycle_col", "cycle"),
            target_col=config.get("target_col", "RUL"),
            feature_cols=config.get("feature_cols"),
            delimiter=config.get("delimiter", ","),
            train_ratio=config.get("train_ratio", 0.8),
        )

    elif ds_type == "nasa_bearing":
        ds.load_nasa_bearing(
            data_dir=config["data_dir"],
            test_ratio=config.get("test_ratio", 0.2),
        )

    elif ds_type == "battery":
        ds.load_battery(
            csv_path=config["csv_path"],
            cycle_col=config.get("cycle_col", "cycle"),
            capacity_col=config.get("capacity_col", "capacity"),
            cell_col=config.get("cell_col", "cell_id"),
            eol_threshold=config.get("eol_threshold", 0.7),
            feature_cols=config.get("feature_cols"),
            train_ratio=config.get("train_ratio", 0.8),
        )

    elif ds_type == "phm2012":
        ds.load_phm2012(
            data_dir=config["data_dir"],
            condition=config.get("condition", "1"),
            train_bearings=config.get("train_bearings"),
            test_bearings=config.get("test_bearings"),
        )

    elif ds_type == "numpy":
        X_train = np.load(config["train_X_path"])
        Y_train = np.load(config["train_Y_path"])
        X_test = np.load(config["test_X_path"])
        Y_test = np.load(config["test_Y_path"])
        ds.load_from_numpy(
            X_train, Y_train, X_test, Y_test,
            already_windowed=config.get("already_windowed", False),
            feature_names=config.get("feature_names"),
        )

    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    return ds


if __name__ == "__main__":
    # Quick test with synthetic data
    ds = GenericTimeSeriesDataset(name="synth_test", sequence_length=30, rul_cap=125)
    ds.load_synthetic(n_units_train=50, n_units_test=10, max_life=200, n_features=14)
    print(ds.summary())
    X_train, Y_train, X_test, Y_test = ds.get_data()
    print(f"\nReady for PI-DP-FCN training:")
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")
