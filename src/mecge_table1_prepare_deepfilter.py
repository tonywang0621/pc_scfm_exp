import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


TEST_SET = {
    "sel123",
    "sel233",
    "sel302",
    "sel307",
    "sel820",
    "sel853",
    "sel16420",
    "sel16795",
    "sele0106",
    "sele0121",
    "sel32",
    "sel49",
    "sel14046",
    "sel15814",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate DeepFilter/MECG-E dataset_bw_nv*.pkl from raw QTDB and NSTDB."
    )
    parser.add_argument("--qtdb-root", required=True, help="Directory containing raw QTDB WFDB records.")
    parser.add_argument("--nstdb-root", required=True, help="Directory containing raw NSTDB WFDB records.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-version", type=int, choices=[1, 2], required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--init-padding", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wfdb_module():
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("DeepFilter/MECG-E raw preparation requires wfdb. Install requirements.txt.") from exc
    return wfdb


def discover_wfdb_records(root):
    root = Path(root)
    records = []
    for dat_path in sorted(root.glob("*.dat")):
        stem = dat_path.with_suffix("")
        if stem.with_suffix(".hea").exists():
            records.append(stem)
    if not records:
        raise FileNotFoundError(f"No WFDB .dat/.hea records found under {root}.")
    return records


def prepare_qtdb(qtdb_root, target_fs=360):
    wfdb = wfdb_module()
    qtdb = {}
    for record_path in discover_wfdb_records(qtdb_root):
        record_name = record_path.name
        signal, fields = wfdb.rdsamp(str(record_path))
        ann = wfdb.rdann(str(record_path), "pu1")
        ann_type = np.asarray(ann.symbol)
        ann_samples = np.asarray(ann.sample)

        pidx = ann_samples[ann_type == "p"]
        sidx = ann_samples[ann_type == "("]
        ridx = ann_samples[ann_type == "N"]
        if len(pidx) == 0 or len(sidx) == 0:
            qtdb[record_name] = []
            continue

        indices = np.zeros(len(pidx), dtype=np.int64)
        for idx, p_sample in enumerate(pidx):
            preceding = np.where(p_sample > sidx)[0]
            if len(preceding) == 0:
                indices[idx] = 0
            else:
                indices[idx] = preceding[-1]
        pstart = sidx[indices] - int(0.04 * fields["fs"])
        pstart = pstart[pstart >= 0]

        first_channel = np.asarray(signal[:, 0], dtype=np.float64)
        beats = []
        for idx in range(len(pstart) - 1):
            has_two_qrs = np.sum((ridx > pstart[idx]) & (ridx < pstart[idx + 1])) >= 2
            if not has_two_qrs:
                beats.append(first_channel[pstart[idx] : pstart[idx + 1]])

        resampled = []
        for beat in beats:
            length = int(np.ceil(len(beat) * target_fs / fields["fs"]))
            padded = list(reversed(beat)) + list(beat) + list(reversed(beat))
            res = resample_poly(padded, target_fs, fields["fs"])
            resampled.append(np.asarray(res[length - 1 : 2 * length - 1], dtype=np.float32))
        qtdb[record_name] = resampled
    return qtdb


def prepare_nstdb(nstdb_root):
    wfdb = wfdb_module()
    bw_signals, _ = wfdb.rdsamp(str(Path(nstdb_root) / "bw"))
    return np.asarray(bw_signals, dtype=np.float32)


def split_noise(bw_signals, noise_version):
    half = int(bw_signals.shape[0] / 2)
    channel1_a = bw_signals[0:half, 0]
    channel1_b = bw_signals[half:-1, 0]
    channel2_a = bw_signals[0:half, 1]
    channel2_b = bw_signals[half:-1, 1]
    if noise_version == 1:
        return channel1_a, channel2_b
    return channel2_a, channel1_b


def build_clean_splits(qtdb, samples, init_padding):
    beats_train = []
    beats_test = []
    skipped = 0
    for signal_name in list(qtdb.keys()):
        for beat in qtdb[signal_name]:
            beat = np.asarray(beat, dtype=np.float32)
            if beat.shape[0] > samples - init_padding:
                skipped += 1
                continue
            out = np.zeros(samples, dtype=np.float32)
            out[init_padding : beat.shape[0] + init_padding] = beat - (beat[0] + beat[-1]) / 2
            if signal_name in TEST_SET:
                beats_test.append(out)
            else:
                beats_train.append(out)
    if not beats_train or not beats_test:
        raise ValueError(f"Empty train/test split: train={len(beats_train)} test={len(beats_test)} skipped={skipped}.")
    return np.asarray(beats_train, dtype=np.float32), np.asarray(beats_test, dtype=np.float32), skipped


def add_noise(clean_beats, noise_source, random_levels, samples):
    noisy = []
    noise_index = 0
    for beat, level in zip(clean_beats, random_levels):
        noise = noise_source[noise_index : noise_index + samples]
        beat_ptp = np.max(beat) - np.min(beat)
        noise_ptp = np.max(noise) - np.min(noise)
        if beat_ptp <= 0 or noise_ptp <= 0:
            alpha = 0.0
        else:
            ase = noise_ptp / beat_ptp
            alpha = level / ase
        noisy.append(beat + alpha * noise)
        noise_index += samples
        if noise_index > len(noise_source) - samples:
            noise_index = 0
    return np.asarray(noisy, dtype=np.float32)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"dataset_bw_nv{args.noise_version}.pkl"
    rnd_test_path = output_dir / f"rnd_test_nv{args.noise_version}.npy"
    if (pkl_path.exists() or rnd_test_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{pkl_path} or {rnd_test_path} already exists; pass --overwrite.")

    np.random.seed(seed=args.seed)
    qtdb = prepare_qtdb(args.qtdb_root)
    bw_signals = prepare_nstdb(args.nstdb_root)
    noise_train, noise_test = split_noise(bw_signals, args.noise_version)
    y_train, y_test, skipped = build_clean_splits(qtdb, args.samples, args.init_padding)

    rnd_train = np.random.randint(low=20, high=200, size=len(y_train)) / 100
    x_train = add_noise(y_train, noise_train, rnd_train, args.samples)
    rnd_test = np.random.randint(low=20, high=200, size=len(y_test)) / 100
    x_test = add_noise(y_test, noise_test, rnd_test, args.samples)

    dataset = [
        np.expand_dims(x_train, axis=2),
        np.expand_dims(y_train, axis=2),
        np.expand_dims(x_test, axis=2),
        np.expand_dims(y_test, axis=2),
    ]
    with open(pkl_path, "wb") as handle:
        pickle.dump(dataset, handle)
    np.save(rnd_test_path, rnd_test)
    print(f"saved {pkl_path}")
    print(f"saved {rnd_test_path}")
    print(f"train={len(y_train)} test={len(y_test)} skipped={skipped} shape={dataset[0].shape}")


if __name__ == "__main__":
    main()
