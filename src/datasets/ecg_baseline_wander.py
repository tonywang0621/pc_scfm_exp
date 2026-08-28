from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

try:
    from .factory import register_dataset
except ImportError:
    from factory import register_dataset


def _as_single_lead_tensor(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    elif array.ndim == 3 and array.shape[1] != 1 and array.shape[2] == 1:
        array = np.transpose(array, (0, 2, 1))
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError(f"Expected ECG array shaped [N, T], [N, 1, T], or [N, T, 1], got {array.shape}.")
    return torch.from_numpy(array)


def _get_array(mapping, names):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Missing one of expected keys: {names}.")


@register_dataset("ecg_baseline_wander")
class ECGBaselineWanderDataset(Dataset):
    """
    Dataset adapter for the ECG baseline-wander-removal benchmark.

    It expects preprocessed NPZ files under `processed_data_dir` or `data_dir/processed`:
      train.npz, val.npz, test.npz

    External test datasets are optional and use the same NPZ schema:
      mit_bih.npz, chapman.npz, cpsc.npz, qtdb.npz

    Each NPZ must contain:
      noisy_ecg or input: model input with synthetic/real baseline wander
      clean_reference or target: ECG clean reference

    For MECG-E Table 1 reproduction, `dataset.pkl_file` can point to an official
    `dataset_bw_nv*.pkl` file shaped as [X_train, y_train, X_test, y_test].
    The training portion is split into train/val with the official random_state=1.

    Arrays may be [N, T], [N, 1, T], or [N, T, 1]. The loader returns
    (noisy_ecg, clean_reference).
    If `teacher_prediction_key` is configured and present in the NPZ, it additionally
    returns an all-valid mask and the teacher prediction for distillation models.
    """

    split_file_map = {
        "train": "train.npz",
        "val": "val.npz",
        "test": "test.npz",
        "mit_bih": "mit_bih.npz",
        "mit-bih": "mit_bih.npz",
        "chapman": "chapman.npz",
        "cpsc": "cpsc.npz",
        "qtdb": "qtdb.npz",
    }

    def __init__(self, data_mode="train", **kwargs):
        if data_mode not in self.split_file_map:
            raise ValueError(
                f"Unsupported data_mode={data_mode!r}. Expected one of {sorted(self.split_file_map)}."
            )

        data_dir = Path(kwargs["data_dir"])
        processed_dir = Path(kwargs.get("processed_data_dir", data_dir / "processed"))
        split_path = processed_dir / self.split_file_map[data_mode]

        explicit_pkl_file = kwargs.get("pkl_file") or kwargs.get("mecge_pkl_file")
        if split_path.exists() and explicit_pkl_file is None:
            self.noisy, self.clean, loaded = self._load_npz(split_path)
        else:
            pkl_path = self._resolve_pkl_path(data_dir, processed_dir, data_mode, kwargs)
            if pkl_path is None:
                raise FileNotFoundError(
                    f"Missing preprocessed ECG split: {split_path}. Create NPZ files with "
                    "`noisy_ecg`/`clean_reference` arrays before training, or set "
                    "`dataset.pkl_file` to a MECG-E dataset_bw_nv*.pkl file."
                )
            self.noisy, self.clean = self._load_pkl_split(pkl_path, data_mode, kwargs)
            loaded = {}

        if len(self.noisy) != len(self.clean):
            raise ValueError(
                f"noisy and clean arrays must have the same length, got {len(self.noisy)} and {len(self.clean)}."
            )

        expected_window_size = kwargs.get("window_size")
        if expected_window_size is not None:
            expected_window_size = int(expected_window_size)
            observed_window_size = int(self.noisy.shape[-1])
            if observed_window_size != expected_window_size:
                raise ValueError(
                    f"{self.source_path} has ECG windows of length {observed_window_size}, "
                    f"but dataset.window_size={expected_window_size}. Re-run preprocess_ecg.py "
                    "for this model-specific config, or point dataset.processed_data_dir/pkl_file to the correct files."
                )
        if self.clean.shape[-1] != self.noisy.shape[-1]:
            raise ValueError(
                f"noisy and clean window lengths must match, got {self.noisy.shape[-1]} and {self.clean.shape[-1]}."
            )
        teacher_key = kwargs.get("teacher_prediction_key")
        self.teacher = None
        if teacher_key and teacher_key in loaded:
            self.teacher = _as_single_lead_tensor(loaded[teacher_key])
            if len(self.teacher) != len(self.noisy):
                raise ValueError(
                    f"{teacher_key} length {len(self.teacher)} does not match ECG length {len(self.noisy)}."
                )

        self._validate_ptbxl_fold_metadata(loaded, data_mode, kwargs.get("split"))

    def _load_npz(self, split_path):
        self.source_path = split_path
        loaded = np.load(split_path)
        noisy_key = "noisy_ecg" if "noisy_ecg" in loaded else "input"
        clean_key = "clean_reference" if "clean_reference" in loaded else "target"
        if noisy_key not in loaded or clean_key not in loaded:
            raise KeyError(
                f"{split_path} must contain noisy_ecg/input and clean_reference/target arrays."
            )
        return _as_single_lead_tensor(loaded[noisy_key]), _as_single_lead_tensor(loaded[clean_key]), loaded

    def _resolve_pkl_path(self, data_dir, processed_dir, data_mode, kwargs):
        explicit = kwargs.get("pkl_file") or kwargs.get("mecge_pkl_file")
        if explicit:
            path = Path(explicit)
            return path if path.exists() else None

        candidates = [
            processed_dir / self.split_file_map[data_mode].replace(".npz", ".pkl"),
            processed_dir / "dataset_bw_nv1.pkl",
            data_dir / "raw" / "dataset_bw_nv1.pkl",
            data_dir / "dataset_bw_nv1.pkl",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_pkl_split(self, pkl_path, data_mode, kwargs):
        self.source_path = pkl_path
        with open(pkl_path, "rb") as handle:
            dataset = pickle.load(handle)

        if isinstance(dataset, dict):
            if data_mode == "train":
                noisy = _get_array(dataset, ("X_train", "noisy_train", "train_noisy", "input_train"))
                clean = _get_array(dataset, ("y_train", "clean_train", "train_clean", "target_train"))
                if "X_val" in dataset and "y_val" in dataset:
                    return _as_single_lead_tensor(noisy), _as_single_lead_tensor(clean)
                noisy, _, clean, _ = self._split_train_validation(noisy, clean, kwargs)
            elif data_mode == "val":
                if "X_val" in dataset and "y_val" in dataset:
                    noisy = dataset["X_val"]
                    clean = dataset["y_val"]
                else:
                    train_noisy = _get_array(dataset, ("X_train", "noisy_train", "train_noisy", "input_train"))
                    train_clean = _get_array(dataset, ("y_train", "clean_train", "train_clean", "target_train"))
                    _, noisy, _, clean = self._split_train_validation(train_noisy, train_clean, kwargs)
            elif data_mode == "test":
                noisy = _get_array(dataset, ("X_test", "noisy_test", "test_noisy", "input_test"))
                clean = _get_array(dataset, ("y_test", "clean_test", "test_clean", "target_test"))
            else:
                raise ValueError(f"PKL loading only supports train/val/test modes, got {data_mode!r}.")
            return _as_single_lead_tensor(noisy), _as_single_lead_tensor(clean)

        if len(dataset) < 4:
            raise ValueError(f"{pkl_path} must contain at least [X_train, y_train, X_test, y_test].")

        x_train, y_train, x_test, y_test = dataset[:4]
        if data_mode == "train":
            noisy, _, clean, _ = self._split_train_validation(x_train, y_train, kwargs)
        elif data_mode == "val":
            _, noisy, _, clean = self._split_train_validation(x_train, y_train, kwargs)
        elif data_mode == "test":
            noisy, clean = x_test, y_test
        else:
            raise ValueError(f"PKL loading only supports train/val/test modes, got {data_mode!r}.")
        return _as_single_lead_tensor(noisy), _as_single_lead_tensor(clean)

    def _split_train_validation(self, noisy, clean, kwargs):
        val_ratio = float(kwargs.get("pkl_validation_ratio", kwargs.get("validation_ratio", 0.3)))
        random_state = int(kwargs.get("pkl_validation_random_state", 1))
        return train_test_split(
            noisy,
            clean,
            test_size=val_ratio,
            shuffle=True,
            random_state=random_state,
        )

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        if self.teacher is not None:
            valid_mask = torch.ones_like(self.noisy[index])
            return self.noisy[index], self.clean[index], valid_mask, self.teacher[index]
        return self.noisy[index], self.clean[index]

    def _validate_ptbxl_fold_metadata(self, loaded, data_mode, split_config):
        if data_mode not in {"train", "val", "test"}:
            return
        if not isinstance(split_config, dict) or split_config.get("strategy") != "ptbxl_official_folds":
            return

        fold_key = None
        for candidate in ("ptbxl_fold", "strat_fold", "fold"):
            if candidate in loaded:
                fold_key = candidate
                break
        if fold_key is None:
            raise ValueError(
                f"{data_mode}.npz is used with ptbxl_official_folds but does not contain "
                "ptbxl_fold/strat_fold/fold metadata."
            )

        folds = np.asarray(loaded[fold_key], dtype=np.int64)
        if folds.shape[0] != len(self.noisy):
            raise ValueError(
                f"{fold_key} metadata length {folds.shape[0]} does not match ECG length {len(self.noisy)}."
            )

        if data_mode == "train":
            expected = set(split_config.get("train_folds", []))
        elif data_mode == "val":
            expected = {int(split_config.get("validation_fold", 9))}
        else:
            expected = {int(split_config.get("test_fold", 10))}

        observed = set(np.unique(folds).tolist())
        if observed - expected:
            raise ValueError(
                f"{data_mode}.npz contains PTB-XL folds {sorted(observed)}, expected only {sorted(expected)}."
            )
