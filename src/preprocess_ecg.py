import argparse
import csv
import math
from pathlib import Path

import numpy as np
import yaml
from scipy import signal


LEAD_ALIASES = {
    "i": ["i", "lead_i", "lead i"],
    "ii": ["ii", "lead_ii", "lead ii", "mlii", "ml ii"],
    "iii": ["iii", "lead_iii", "lead iii"],
    "v1": ["v1"],
    "v2": ["v2"],
    "v3": ["v3"],
    "v4": ["v4"],
    "v5": ["v5"],
    "v6": ["v6"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ECG baseline-wander-removal NPZ splits from ECG files."
    )
    parser.add_argument("--config", default="configs/ecg_baseline_wander_mecg_e.yaml")
    parser.add_argument("--input-dir", required=True, help="Directory containing source ECG files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to dataset.processed_data_dir, then dataset.data_dir/processed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild requested NPZ splits even when output files already exist.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ptbxl",
        choices=["ptbxl", "mit_bih", "chapman", "cpsc", "qtdb"],
    )
    parser.add_argument("--metadata-csv", default=None, help="Optional CSV with file path and fold columns.")
    parser.add_argument("--noise-dir", default=None, help="Optional NSTDB/noise ECG directory.")
    parser.add_argument("--source-fs", type=float, default=None, help="Override source sampling rate.")
    parser.add_argument(
        "--baseline-kind",
        default=None,
        choices=["nstdb", "sinusoidal", "multi_sine", "random_low_frequency_drift"],
        help="Override dataset.baseline_wander.train_source for all generated windows.",
    )
    parser.add_argument(
        "--alpha-values",
        default=None,
        help="Comma-separated alpha values, e.g. 0.05,0.1,0.2.",
    )
    parser.add_argument(
        "--frequencies-hz",
        default=None,
        help="Comma-separated baseline frequencies, e.g. 0.05,0.1,0.2.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated output splits to build, e.g. test or train,val,test.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return resolve_config_refs(cfg)


def resolve_config_refs(cfg):
    values = dict(cfg)

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            for key, replacement in values.items():
                if not isinstance(replacement, (dict, list)):
                    value = value.replace("${" + key + "}", str(replacement))
        return value

    return resolve(cfg)


def read_metadata(path):
    if path is None:
        return {}
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = [
                row.get("filename"),
                row.get("file"),
                row.get("path"),
                row.get("record"),
                row.get("filename_lr"),
                row.get("filename_hr"),
                row.get("ecg_id"),
            ]
            for key in keys:
                if not key:
                    continue
                key_path = Path(str(key))
                rows[str(key)] = row
                rows[key_path.stem] = row
                rows[key_path.name] = row
    return rows


def metadata_row_for_record(record_path, metadata):
    candidates = [
        str(record_path),
        record_path.as_posix(),
        record_path.stem,
        record_path.name,
        str(record_path.with_suffix("")),
        record_path.with_suffix("").as_posix(),
    ]

    parts = record_path.with_suffix("").parts
    if len(parts) >= 2:
        candidates.append(Path(*parts[-2:]).as_posix())

    for candidate in candidates:
        row = metadata.get(candidate)
        if row:
            return row
    return None


def discover_records(input_dir):
    input_dir = Path(input_dir)
    suffixes = {".npz", ".npy", ".csv", ".txt", ".hea"}
    records = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        if path.suffix.lower() == ".hea":
            records.append(path.with_suffix(""))
        else:
            records.append(path)
    deduped = []
    seen = set()
    for path in records:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def load_record(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=True)
        data_key = next(
            (key for key in ["ecg", "signal", "data", "clean_reference", "target"] if key in loaded),
            None,
        )
        if data_key is None:
            raise KeyError(f"{path} has no ECG array key. Expected ecg/signal/data/clean_reference/target.")
        data = np.asarray(loaded[data_key], dtype=np.float32)
        fs = float(loaded["fs"]) if "fs" in loaded else None
        leads = [str(x) for x in loaded["leads"].tolist()] if "leads" in loaded else None
        return data, fs, leads
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32), None, None
    if path.suffix.lower() in {".csv", ".txt"}:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        try:
            data = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            data = np.genfromtxt(path, delimiter=delimiter, skip_header=1)
            data = data[:, ~np.all(np.isnan(data), axis=0)] if data.ndim == 2 else data
        return np.asarray(data, dtype=np.float32), None, None

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("WFDB input requires `wfdb`. Install it or convert records to NPZ/CSV.") from exc

    record = wfdb.rdrecord(str(path))
    return np.asarray(record.p_signal, dtype=np.float32), float(record.fs), list(record.sig_name)


def select_lead(ecg, lead="II", lead_names=None):
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim == 1:
        if not lead_names:
            raise ValueError(
                f"Single-lead ECG has no lead metadata, so requested lead {lead!r} cannot be verified."
            )
        aliases = LEAD_ALIASES.get(str(lead).lower(), [str(lead).lower()])
        normalized = [str(name).lower().replace("-", "_") for name in lead_names]
        if len(normalized) == 1 and normalized[0] in aliases:
            return ecg
        raise ValueError(
            f"Single-lead ECG metadata {list(lead_names)!r} does not verify requested lead {lead!r}."
        )
    if ecg.ndim != 2:
        raise ValueError(f"Expected ECG record shaped [T], [T, L], or [L, T], got {ecg.shape}.")

    # Prefer time-major layout for common WFDB/PTB-XL exports.
    if ecg.shape[0] < ecg.shape[1] and ecg.shape[0] <= 16:
        ecg = ecg.T

    if lead_names:
        aliases = LEAD_ALIASES.get(str(lead).lower(), [str(lead).lower()])
        normalized = [str(name).lower().replace("-", "_") for name in lead_names]
        for idx, name in enumerate(normalized):
            if name in aliases:
                return ecg[:, idx]
        raise ValueError(
            f"Requested lead {lead!r} was not found in available leads {list(lead_names)!r}."
        )
    elif not lead_names:
        raise ValueError(
            f"ECG has no lead metadata, so requested lead {lead!r} cannot be verified."
        )
    raise ValueError(
        f"Requested lead {lead!r} was not found and no usable lead names were provided."
    )


def filter_ecg(ecg, fs, cfg):
    clean_cfg = cfg["dataset"].get("clean_reference", {})
    band = clean_cfg.get("bandpass_hz")
    if not band:
        return ecg.astype(np.float32)

    low, high = float(band[0]), float(band[1])
    order = int(clean_cfg.get("filter_order", 4))
    nyquist = fs / 2.0
    if low <= 0 and high >= nyquist:
        return ecg.astype(np.float32)
    if low <= 0:
        sos = signal.butter(order, high / nyquist, btype="lowpass", output="sos")
    elif high >= nyquist:
        sos = signal.butter(order, low / nyquist, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    if clean_cfg.get("zero_phase", True):
        return signal.sosfiltfilt(sos, ecg).astype(np.float32)
    return signal.sosfilt(sos, ecg).astype(np.float32)


def resample_ecg(ecg, source_fs, target_fs):
    if int(round(source_fs)) == int(round(target_fs)):
        return ecg.astype(np.float32)
    gcd = math.gcd(int(round(source_fs)), int(round(target_fs)))
    up = int(round(target_fs)) // gcd
    down = int(round(source_fs)) // gcd
    return signal.resample_poly(ecg, up, down).astype(np.float32)


def normalize_window(window, method="z_score", eps=1e-8):
    if method == "none":
        return window.astype(np.float32)
    if method == "endpoint_center":
        return (window - (window[0] + window[-1]) / 2.0).astype(np.float32)
    if method == "min_max":
        lo, hi = np.min(window), np.max(window)
        return ((window - lo) / (hi - lo + eps) * 2.0 - 1.0).astype(np.float32)
    if method == "min_max_01":
        # Per-window min-max to [0, 1]. Chiang et al. 2019 (FCN-DAE) normalise
        # every clean signal to [0, 1] before adding noise ("so that the
        # amplitudes of the sampling points laid between 0 and 1"); baseline
        # wander is then added on top, so the model input can leave [0, 1].
        lo, hi = np.min(window), np.max(window)
        return ((window - lo) / (hi - lo + eps)).astype(np.float32)
    return ((window - np.mean(window)) / (np.std(window) + eps)).astype(np.float32)


def make_windows(ecg, window_size, overlap_ratio):
    step = max(1, int(round(window_size * (1.0 - overlap_ratio))))
    if len(ecg) < window_size:
        return []
    return [ecg[start : start + window_size] for start in range(0, len(ecg) - window_size + 1, step)]


def load_noise_pool(noise_dir, target_fs, window_size, seed, shuffle=True):
    if noise_dir is None:
        return []
    rng = np.random.default_rng(seed)
    pool = []
    failures = []
    for path in discover_records(noise_dir):
        try:
            ecg, fs, leads = load_record(path)
            if fs is None:
                raise ValueError(
                    f"{path} has no sampling-rate metadata. Convert it to WFDB/NPZ with fs, "
                    "or preprocess with an NSTDB source that carries fs metadata."
                )
            lead = select_lead(ecg, lead="II", lead_names=leads)
            lead = resample_ecg(lead, fs, target_fs)
            for window in make_windows(lead, window_size, overlap_ratio=0.0):
                centered = window - np.mean(window)
                if np.std(centered) > 1e-8:
                    pool.append(centered.astype(np.float32))
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if not pool and failures:
        detail = "\n".join(failures[:10])
        raise ValueError(f"No usable NSTDB noise windows were loaded. First failures:\n{detail}")
    if shuffle:
        rng.shuffle(pool)
    return pool


def split_noise_pool(noise_pool, split_cfg, seed):
    ratios = split_cfg.get("ratio", [0.8, 0.1, 0.1])
    if len(ratios) != 3:
        raise ValueError("dataset.baseline_wander.noise_split.ratio must contain train/val/test ratios.")
    train_ratio, val_ratio, _ = [float(value) for value in ratios]
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("dataset.baseline_wander.noise_split.ratio must be non-negative and sum to <= 1.")

    pool = list(noise_pool)
    if split_cfg.get("shuffle", True):
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)

    n_total = len(pool)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = min(n_train, n_total)
    n_val = min(n_val, n_total - n_train)
    pools = {
        "train": pool[:n_train],
        "val": pool[n_train : n_train + n_val],
        "test": pool[n_train + n_val :],
    }
    if not pools["train"] or not pools["val"] or not pools["test"]:
        raise ValueError(
            "NSTDB noise split produced an empty pool. Provide more noise windows or adjust "
            f"dataset.baseline_wander.noise_split.ratio. Counts: "
            f"train={len(pools['train'])}, val={len(pools['val'])}, test={len(pools['test'])}."
        )
    return pools


def noise_pool_name_for_split(split_name):
    return split_name if split_name in {"train", "val"} else "test"


def synthetic_baseline(kind, length, fs, rng, frequencies):
    t = np.arange(length, dtype=np.float32) / float(fs)
    if kind == "sinusoidal":
        freq = float(rng.choice(frequencies))
        phase = float(rng.uniform(0, 2 * np.pi))
        return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    if kind == "multi_sine":
        out = np.zeros(length, dtype=np.float32)
        for _ in range(3):
            freq = float(rng.choice(frequencies))
            phase = float(rng.uniform(0, 2 * np.pi))
            out += float(rng.uniform(0.2, 1.0)) * np.sin(2 * np.pi * freq * t + phase)
        return out.astype(np.float32)
    white = rng.normal(0, 1, size=length).astype(np.float32)
    sos = signal.butter(2, min(1.0, max(frequencies)) / (fs / 2.0), btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, white).astype(np.float32)


def _sample_alpha(alpha_values, alpha_sampling, rng):
    if len(alpha_values) <= 1:
        return float(alpha_values[0]) if alpha_values else 0.1
    if alpha_sampling in {"uniform_range", "range", "continuous"}:
        low, high = min(alpha_values), max(alpha_values)
        return float(rng.uniform(low, high))
    if alpha_sampling in {"integer_percent_uniform", "deepfilter", "discrete_percent"}:
        low, high = min(alpha_values), max(alpha_values)
        return float(rng.integers(int(round(low * 100.0)), int(round(high * 100.0))) / 100.0)
    return float(rng.choice(alpha_values))


def contaminate(clean, cfg, noise_pool, rng, test_kind=None, baseline_override=None):
    bw_cfg = cfg["dataset"].get("baseline_wander", {})
    alpha_values = [float(x) for x in bw_cfg.get("alpha_values", [0.1])]
    alpha_sampling = str(bw_cfg.get("alpha_sampling", "discrete")).lower()
    frequencies = [float(x) for x in bw_cfg.get("controlled_frequencies_hz", [0.1, 0.2, 0.3, 0.5])]
    kind = test_kind or bw_cfg.get("train_source", "nstdb")

    if baseline_override is not None:
        baseline = np.asarray(baseline_override, dtype=np.float32)
        if len(baseline) != len(clean):
            baseline = signal.resample(baseline, len(clean)).astype(np.float32)
    elif kind == "nstdb" and noise_pool:
        baseline = noise_pool[int(rng.integers(0, len(noise_pool)))]
        if len(baseline) != len(clean):
            baseline = signal.resample(baseline, len(clean)).astype(np.float32)
    elif kind == "nstdb":
        raise ValueError(
            "baseline_wander.train_source is nstdb, but no NSTDB noise windows are available. "
            "Provide --noise-dir with usable NSTDB records; do not fall back to synthetic noise "
            "for the main experiment."
        )
    else:
        baseline = synthetic_baseline(kind, len(clean), cfg["dataset"]["resample_hz"], rng, frequencies)

    baseline = baseline - np.mean(baseline)
    # MECG-E (Hung et al. 2024) / DeepFilter (Romero et al. 2021) noise
    # scaling: lambda = delta * ptp(clean) / ptp(noise), i.e. normalize the
    # noise by its own peak-to-peak range (not peak absolute value) before
    # scaling it to a fraction `delta` of the clean ECG's peak-to-peak range.
    baseline = baseline / (np.ptp(baseline) + 1e-8)

    alpha = _sample_alpha(alpha_values, alpha_sampling, rng)

    amplitude = np.ptp(clean) if bw_cfg.get("alpha_mode", "peak_to_peak_ratio") == "peak_to_peak_ratio" else 1.0
    return (clean + alpha * amplitude * baseline).astype(np.float32)


def split_name_for_record(record_path, metadata, dataset_name, split_cfg):
    if dataset_name != "ptbxl":
        return dataset_name
    row = metadata_row_for_record(record_path, metadata)
    fold = None
    if row:
        fold_value = row.get("strat_fold") or row.get("fold") or row.get("ptbxl_fold")
        fold = int(float(fold_value)) if fold_value not in {None, ""} else None
    if fold is None:
        raise ValueError(
            f"PTB-XL preprocessing requires fold metadata for {record_path}. "
            "Provide --metadata-csv with strat_fold/fold/ptbxl_fold."
        )
    if fold in set(split_cfg["train_folds"]):
        return "train"
    if fold == int(split_cfg["validation_fold"]):
        return "val"
    if fold == int(split_cfg["test_fold"]):
        return "test"
    raise ValueError(f"Unexpected PTB-XL fold {fold} for {record_path}.")


# MIT-BIH Arrhythmia Database's own documentation notes that records 201
# and 202 come from the same (male) subject -- the only known same-patient
# duplicate among this project's external test sets' standard record
# numbering, so it is corrected here rather than left to a filename-based
# record_id (which would otherwise treat 201/202 as two different patients).
KNOWN_SAME_PATIENT_ALIASES = {
    "mit_bih": {"202": "201"},
}


def record_id_for(record_path, dataset_name, row):
    """Best-effort patient/record identifier for record-level or
    patient-level statistical aggregation downstream (see result_analysis.py
    paired-stats --group-by record).

    - PTB-XL: one patient can have multiple ECG records, so prefer the real
      `patient_id` from metadata (ptbxl_database.csv) when available.
    - External sets (MIT-BIH/Chapman/CPSC/QTDB): these are one-record-per-
      patient in their standard distributions, so the record filename itself
      is used as the identifier, with a small known-alias table for MIT-BIH's
      documented 201/202 same-patient exception.
    """
    if row:
        for key in ("patient_id", "subject_id", "patient"):
            value = row.get(key)
            if value not in (None, ""):
                return f"patient_{value}"
    stem = record_path.stem
    alias = KNOWN_SAME_PATIENT_ALIASES.get(dataset_name, {}).get(stem)
    return alias or stem


def save_split(path, clean_windows, noisy_windows, folds=None, record_ids=None):
    if not clean_windows:
        return
    arrays = {
        "clean_reference": np.asarray(clean_windows, dtype=np.float32)[:, None, :],
        "noisy_ecg": np.asarray(noisy_windows, dtype=np.float32)[:, None, :],
    }
    if folds:
        arrays["ptbxl_fold"] = np.asarray(folds, dtype=np.int64)
    if record_ids:
        # Fixed-width unicode string array; plain np.savez handles this
        # natively (no allow_pickle needed) since dtype is '<U...', not object.
        arrays["record_id"] = np.asarray(record_ids)
    np.savez(path, **arrays)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.baseline_kind:
        cfg["dataset"].setdefault("baseline_wander", {})["train_source"] = args.baseline_kind
    if args.alpha_values:
        cfg["dataset"].setdefault("baseline_wander", {})["alpha_values"] = [
            float(value) for value in args.alpha_values.split(",") if value.strip()
        ]
    if args.frequencies_hz:
        cfg["dataset"].setdefault("baseline_wander", {})["controlled_frequencies_hz"] = [
            float(value) for value in args.frequencies_hz.split(",") if value.strip()
        ]
    dataset_cfg = cfg["dataset"]
    requested_splits = (
        {value.strip() for value in args.splits.split(",") if value.strip()}
        if args.splits
        else None
    )
    output_dir = Path(
        args.output_dir
        or dataset_cfg.get("processed_data_dir")
        or (Path(dataset_cfg["data_dir"]) / "processed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    file_map = {"train": "train.npz", "val": "val.npz", "test": "test.npz"}
    file_map.update({
        "mit_bih": "mit_bih.npz",
        "chapman": "chapman.npz",
        "cpsc": "cpsc.npz",
        "qtdb": "qtdb.npz",
    })

    output_splits = requested_splits or (
        {"train", "val", "test"} if args.dataset_name == "ptbxl" else {args.dataset_name}
    )
    existing_splits = {
        split_name
        for split_name in output_splits
        if split_name in file_map and (output_dir / file_map[split_name]).exists()
    }
    if existing_splits and not args.overwrite:
        existing_paths = ", ".join(str(output_dir / file_map[split]) for split in sorted(existing_splits))
        raise FileExistsError(
            "Preprocessed split files already exist, and reusing stale NPZ files can violate "
            f"the experiment design: {existing_paths}. Re-run with --overwrite after confirming "
            "the source data, NSTDB noise, split, and alpha settings."
        )

    rng = np.random.default_rng(args.seed)
    metadata = read_metadata(args.metadata_csv)
    target_fs = float(dataset_cfg.get("resample_hz", 250))
    window_size = int(dataset_cfg.get("window_size", 512))
    overlap_ratio = float(dataset_cfg.get("overlap_ratio", 0.5))
    split_cfg = dataset_cfg["split"]
    bw_cfg = dataset_cfg.get("baseline_wander", {})
    noise_sampling = str(bw_cfg.get("noise_sampling", "random")).lower()
    noise_pool = load_noise_pool(
        args.noise_dir,
        target_fs,
        window_size,
        args.seed,
        shuffle=False,
    )
    if str(bw_cfg.get("train_source", "nstdb")).lower() == "nstdb" and not noise_pool:
        raise FileNotFoundError(
            "baseline_wander.train_source is nstdb, but --noise-dir was not provided or produced "
            "no usable NSTDB windows. This experiment design requires NSTDB baseline wander; "
            "use --baseline-kind sinusoidal/multi_sine/random_low_frequency_drift only for "
            "controlled robustness experiments."
        )
    noise_pools = split_noise_pool(
        noise_pool,
        bw_cfg.get("noise_split", {"ratio": [0.8, 0.1, 0.1], "shuffle": True}),
        args.seed,
    ) if noise_pool else {}
    if noise_pools:
        print(
            "NSTDB noise split: "
            f"train={len(noise_pools['train'])}, "
            f"val={len(noise_pools['val'])}, "
            f"test={len(noise_pools['test'])}"
        )
    noise_indices = {name: 0 for name in noise_pools}

    clean_by_split = {}
    noisy_by_split = {}
    folds_by_split = {}
    record_ids_by_split = {}
    skipped_records = 0
    records = discover_records(args.input_dir)
    if args.limit:
        records = records[: args.limit]

    for record_path in records:
        split_name = split_name_for_record(record_path, metadata, args.dataset_name, split_cfg)
        if requested_splits and split_name not in requested_splits:
            continue
        try:
            ecg, fs, leads = load_record(record_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ECG record {record_path}") from exc
        fs = args.source_fs if args.source_fs is not None else fs
        if fs is None:
            raise ValueError(
                f"{record_path} has no sampling-rate metadata. Provide --source-fs or use WFDB/NPZ input with fs."
            )
        fs = float(fs)
        try:
            lead = select_lead(ecg, lead=dataset_cfg.get("lead", "II"), lead_names=leads)
        except ValueError as exc:
            skipped_records += 1
            print(f"skipping {record_path}: {exc}")
            continue
        clean = filter_ecg(lead, fs, cfg)
        clean = resample_ecg(clean, fs, target_fs)

        fold = None
        row = metadata_row_for_record(record_path, metadata)
        if row:
            fold_value = row.get("strat_fold") or row.get("fold") or row.get("ptbxl_fold")
            fold = int(float(fold_value)) if fold_value not in {None, ""} else None
        record_id = record_id_for(record_path, args.dataset_name, row)

        for window in make_windows(clean, window_size, overlap_ratio):
            normalized = normalize_window(window, dataset_cfg.get("normalization", "z_score"))
            baseline_override = None
            current_noise_pool = noise_pools.get(noise_pool_name_for_split(split_name), noise_pool)
            if noise_pool and noise_sampling in {"sequential", "cyclic", "deepfilter"}:
                pool_name = noise_pool_name_for_split(split_name)
                noise_index = noise_indices.get(pool_name, 0)
                baseline_override = current_noise_pool[noise_index]
                noise_indices[pool_name] = (noise_index + 1) % len(current_noise_pool)
            noisy = contaminate(normalized, cfg, current_noise_pool, rng, baseline_override=baseline_override)
            clean_by_split.setdefault(split_name, []).append(normalized)
            noisy_by_split.setdefault(split_name, []).append(noisy)
            if fold is not None:
                folds_by_split.setdefault(split_name, []).append(fold)
            record_ids_by_split.setdefault(split_name, []).append(record_id)

    for split_name, clean_windows in clean_by_split.items():
        save_split(
            output_dir / file_map[split_name],
            clean_windows,
            noisy_by_split[split_name],
            folds_by_split.get(split_name),
            record_ids_by_split.get(split_name),
        )
        print(f"saved {split_name}: {len(clean_windows)} windows -> {output_dir / file_map[split_name]}")
    if skipped_records:
        print(f"skipped {skipped_records} records because the requested lead could not be verified.")

if __name__ == "__main__":
    main()
