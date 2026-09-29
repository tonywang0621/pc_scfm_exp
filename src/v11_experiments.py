"""Isolated v11 experiments using the unchanged Table 1 training protocol."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "src"
V7_CONFIG = "mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml"
V11_CONFIG = "mecge_table1_repro_lstm_v11.yaml"
VARIANTS = {
    "v7_max": (V7_CONFIG, {"model.lambda_max": 0.05, "model.mad_topk": 8}, "v7 + top-k MAD loss"),
    "v7_straight": (V7_CONFIG, {"model.cfm_bridge_noise_scale": 0.0}, "v7 + straight CFM path only"),
    "v11": (V11_CONFIG, {}, "Full single-residual v11"),
    "v11_no_max": (V11_CONFIG, {"model.lambda_max": 0.0}, "v11 without top-k MAD loss"),
    "v11_direct": (V11_CONFIG, {"model.v11_use_cfm": False}, "Same v11 refiner and reconstruction losses, without CFM supervision"),
    "v11_no_band_split": (V11_CONFIG, {"model.v11_use_band_split": False}, "One full-band correction instead of frequency-dependent correction"),
    "v11_no_feature_dapp": (V11_CONFIG, {"model.v11_feature_dapp": False}, "Remove feature-domain DAPP"),
    "v11_no_unet_dapp": (V11_CONFIG, {"model.v11_unet_dapp": False}, "Remove U-Net skip DAPP"),
    "v11_no_resconv": (V11_CONFIG, {"model.v11_resconv": False}, "Remove residual convolution context, keep both LSTM axes"),
    "v11_no_dual_head": (V11_CONFIG, {"model.v11_use_dual_head": False}, "Remove both coarse auxiliary heads and their supervision"),
    "v11_fixed_gate": (V11_CONFIG, {"model.v11_adaptive_gate": False}, "Fix gates at their initial amplitudes"),
}
CORE = ("v7_max", "v7_straight", "v11", "v11_no_max", "v11_direct", "v11_no_band_split")
MODULES = ("v11", "v11_no_feature_dapp", "v11_no_unet_dapp", "v11_no_resconv", "v11_no_dual_head", "v11_fixed_gate")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--models", nargs="+", choices=tuple(VARIANTS), default=None)
    choice.add_argument("--suite", choices=("core", "modules", "all"))
    parser.add_argument("--nv", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/mecge_table1_repro")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/v11_performance")
    parser.add_argument("--pkl-file", help="Dataset path; use {nv} when running both noise versions")
    parser.add_argument("--rnd-test", help="Noise-amplitude array path, optionally containing {nv}")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Shared model/training OmegaConf override")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="Use the legacy runner's epoch-level resume semantics")
    restart.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoints in this run root")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if args.nv == "all" and args.pkl_file and "{nv}" not in args.pkl_file:
        parser.error("--pkl-file must contain {nv} with --nv all")
    for override in args.set:
        if "=" not in override or not override.startswith(("model.", "training.")):
            parser.error("--set accepts model.* or training.* overrides only")
    return args


def select_variants(args):
    if args.models:
        return tuple(dict.fromkeys(args.models))
    return {"core": CORE, "modules": MODULES, "all": tuple(VARIANTS)}.get(args.suite, ("v11",))


def check_output_root(root):
    root = root.resolve()
    protected = [PROJECT / "runs/mecge_table1_repro", PROJECT / "data", PROJECT / "src", PROJECT / "scripts"]
    for path in protected:
        path = path.resolve()
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError(f"Output root overlaps a protected legacy/data/source path: {root}")
    return root


def dataset_path(args, nv):
    path = args.pkl_file.replace("{nv}", str(nv)) if args.pkl_file else args.data_root / "raw" / f"dataset_bw_nv{nv}.pkl"
    return Path(path).resolve()


def amplitude_paths(args):
    result = {}
    versions = (1, 2) if args.nv == "all" else (int(args.nv),)
    for nv in versions:
        if args.rnd_test:
            path = Path(args.rnd_test.replace("{nv}", str(nv))).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Requested noise-amplitude array is missing: {path}")
            result[nv] = path
            continue
        candidates = [
            args.data_root / "raw" / f"rnd_test_nv{nv}.npy",
            args.data_root / "raw/rnd_test.npy",
            PROJECT / "references/MECG-E" / f"rnd_test_nv{nv}.npy",
            PROJECT / "references/MECG-E/rnd_test.npy",
        ]
        result[nv] = next((path.resolve() for path in candidates if path.is_file()), None)
    return result


def build_config(variant, seed, nv, dataset, overrides=(), epochs=None):
    from omegaconf import OmegaConf

    filename, changes, _ = VARIANTS[variant]
    config = OmegaConf.load(APP / "configs" / filename)
    for key, value in changes.items():
        OmegaConf.update(config, key, value, merge=False)
    if variant == "v7_max":
        config.model.loss_fn += "+max"
    if variant == "v11_no_max":
        config.model.loss_fn = "+".join(token for token in config.model.loss_fn.split("+") if token != "max")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if epochs is not None:
        config.training.train_epochs = epochs
    if int(config.training.train_epochs) < 1 or int(config.training.batch_size) < 1:
        raise ValueError("Training epochs and batch size must be positive")
    if config.training.get("selection_metric", "val_loss") != "val_loss":
        raise ValueError("The official runner selects val_loss; other selection metrics are not supported here")
    config.seed = seed
    config.exp_name = f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}"
    config.dataset.pkl_file = str(dataset)
    return config


def source_digests():
    files = list((APP / "models").glob("*.py"))
    files += [APP / "mecge_table1_run_official_local_model.py", APP / "mecge_table1_run_v11_model.py"]
    return {str(path.relative_to(PROJECT)): file_digest(path) for path in sorted(files)}


def environment_info(device):
    import numpy
    import sklearn
    import torch
    import omegaconf

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the training machine or explicitly pass --device cpu")
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "omegaconf": omegaconf.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def job_command(config_path, dataset, checkpoint, log, result, seed, device, resume=False, eval_only=False):
    command = [
        sys.executable, "-B", str(APP / "mecge_table1_run_v11_model.py"),
        "--config", str(config_path), "--dataset-pkl", str(dataset),
        "--checkpoint-dir", str(checkpoint), "--log-dir", str(log),
        "--output-pkl", str(result), "--seed", str(seed), "--device", device,
    ]
    if resume:
        command.append("--resume")
    if eval_only:
        command.append("--skip-train")
    return command


def run_job(args, root, variant, seed, nv, dataset, data_hash, sources, environment):
    from omegaconf import OmegaConf

    job = root / variant / f"nv{nv}" / f"seed{seed}"
    checkpoint = job / "checkpoint"
    result = root / "official_results" / f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}.pkl"
    config = build_config(variant, seed, nv, dataset, args.set, args.epochs)
    config.root_dir = str(job)
    config.checkpoint_dir = str(checkpoint)
    config.log_dir = str(job / "log")
    config.results_dir = str(job / "results")
    config_path = job / "resolved_config.yaml"
    command = job_command(config_path, dataset, checkpoint, job / "log", result, seed, args.device, args.resume, args.eval_only)
    print(f"\n{variant} / nv{nv} / seed{seed}: {VARIANTS[variant][2]}", flush=True)
    if args.dry_run:
        print(shlex.join(command))
        return

    identity = {
        "schema": 1, "variant": variant, "seed": seed, "nv": nv,
        "config": OmegaConf.to_container(config, resolve=True),
        "dataset_sha256": data_hash, "sources": sources, "environment": environment,
    }
    manifest_path = job / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise RuntimeError(f"Settings, data, code, or environment changed for {job}. Choose a new --run-root.")
        if manifest.get("complete") and result.is_file() and not args.eval_only:
            if file_digest(result) != manifest.get("result_sha256"):
                raise RuntimeError(f"Completed result changed: {result}")
            print("Matching completed run found; skipping.", flush=True)
            return
    elif result.exists() or (job.exists() and any(job.iterdir())):
        raise RuntimeError(f"Unmanaged existing output found at {job}; choose a new --run-root.")

    state = checkpoint / "training_state.pt"
    if args.eval_only and not (checkpoint / "best_model.pt").is_file():
        raise FileNotFoundError(f"No checkpoint to evaluate: {checkpoint / 'best_model.pt'}")
    if state.exists() and not args.resume and not args.eval_only:
        raise RuntimeError(f"An interrupted run exists at {job}. Use --resume or a new --run-root.")
    if args.resume and not state.exists():
        command.remove("--resume")
    job.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_path, resolve=True)
    atomic_json(manifest_path, {"identity": identity, "command": command, "complete": False})
    subprocess.run(command, cwd=APP, check=True)
    if not result.is_file():
        raise RuntimeError(f"Training finished without producing {result}")
    atomic_json(manifest_path, {
        "identity": identity, "command": command, "complete": True,
        "result_sha256": file_digest(result),
        "checkpoint_sha256": file_digest(checkpoint / "best_model.pt"),
    })


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, (_, _, description) in VARIANTS.items():
            print(f"{name:26s} {description}")
        return
    root = check_output_root(args.run_root)
    args.data_root = args.data_root.resolve()
    amplitudes = {} if args.skip_robustness else amplitude_paths(args)
    if args.collect_only:
        from v11_results import collect_results
        collect_results(root, amplitudes)
        return

    variants = select_variants(args)
    versions = (1, 2) if args.nv == "all" else (int(args.nv),)
    seeds = tuple(dict.fromkeys(args.seeds))
    datasets = {nv: dataset_path(args, nv) for nv in versions}
    print(f"Plan: {len(variants)} variants x {len(versions)} noise versions x {len(seeds)} seeds")
    print(f"Output: {root}")
    environment, sources, hashes = {}, {}, {}
    if not args.dry_run:
        for path in datasets.values():
            if not path.is_file():
                raise FileNotFoundError(f"Dataset is missing: {path}. Point --data-root or --pkl-file at existing official data.")
        environment = environment_info(args.device)
        sources = source_digests()
        hashes = {nv: file_digest(path) for nv, path in datasets.items()}
    if args.resume:
        print("Resume follows the legacy runner: RNG states are not restored. Use uninterrupted runs for final reproducibility comparisons.")
    for variant in variants:
        for seed in seeds:
            for nv in versions:
                run_job(args, root, variant, seed, nv, datasets[nv], hashes.get(nv), sources, environment)
    if not args.dry_run:
        from v11_results import collect_results
        collect_results(root, amplitudes)


if __name__ == "__main__":
    main()
