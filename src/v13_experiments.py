"""Train v13 and its ablations; keep primary val_loss and secondary SSD reports separate."""

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from v11_experiments import atomic_json, environment_info, file_digest


PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "src"
V13_CONFIG = "mecge_table1_repro_lstm_v13.yaml"
VARIANTS = {
    "v13": ({}, "v12 gates + delayed top-k + extra MSE + raw cosine loss"),
    "v13_no_extra_mse": ({"model.v13_extra_mse_weight": 0.0}, "Remove only the extra MSE term"),
    "v13_no_raw_cos": ({"model.v13_raw_cos_weight": 0.0}, "Remove only the raw cosine term"),
    "v13_tail_only": ({"model.v13_extra_mse_weight": 0.0, "model.v13_raw_cos_weight": 0.0},
                      "v12_delayed_topk control, run under the same v13 pipeline"),
    "v13_no_topk": ({"model.v12_tail_weight": 0.0}, "Remove only the delayed top-k term"),
    "v13_no_local_gate": ({"model.v12_local_gate": False}, "Remove only local temporal gate offsets"),
}
CORE = tuple(VARIANTS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--models", nargs="+", choices=tuple(VARIANTS))
    choice.add_argument("--suite", choices=("core", "all"))
    parser.add_argument("--nv", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/mecge_table1_repro")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/v13_performance")
    parser.add_argument("--pkl-file", help="Existing official dataset path; use {nv} for both noise versions")
    parser.add_argument("--rnd-test", help="Noise-amplitude array, optionally containing {nv}")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Shared model.* or training.* OmegaConf override")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="Resume incomplete runs with saved RNG/optimizer state")
    restart.add_argument("--eval-only", action="store_true", help="Export both existing checkpoint selections again")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if args.nv == "all" and args.pkl_file and "{nv}" not in args.pkl_file:
        parser.error("--pkl-file must contain {nv} with --nv all")
    if args.collect_only and (args.dry_run or args.resume or args.eval_only):
        parser.error("--collect-only cannot be combined with --dry-run, --resume, or --eval-only")
    for override in args.set:
        if "=" not in override or not override.startswith(("model.", "training.")):
            parser.error("--set accepts model.* or training.* overrides only")
    return args


def select_variants(args):
    if args.models:
        return tuple(dict.fromkeys(args.models))
    return {"core": CORE, "all": CORE}[args.suite or "all"]


def check_output_root(root):
    root = root.resolve()
    protected = [PROJECT / item for item in
                 ("runs/mecge_table1_repro", "runs/v11_performance", "runs/v12_performance", "data", "src", "scripts", "docs", "tests")]
    for path in protected:
        path = path.resolve()
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError(f"Output root overlaps a protected legacy/data/source path: {root}")
    return root


def dataset_path(args, nv):
    path = args.pkl_file.replace("{nv}", str(nv)) if args.pkl_file else args.data_root / "raw" / f"dataset_bw_nv{nv}.pkl"
    return Path(path).resolve()


def amplitude_paths(args):
    if args.skip_robustness:
        return {}
    result = {}
    for nv in ((1, 2) if args.nv == "all" else (int(args.nv),)):
        if args.rnd_test:
            path = Path(args.rnd_test.replace("{nv}", str(nv))).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Requested noise-amplitude array is missing: {path}")
            result[nv] = path
        else:
            candidates = [args.data_root / "raw" / f"rnd_test_nv{nv}.npy", args.data_root / "raw/rnd_test.npy",
                          PROJECT / "references/MECG-E" / f"rnd_test_nv{nv}.npy",
                          PROJECT / "references/MECG-E/rnd_test.npy"]
            result[nv] = next((path.resolve() for path in candidates if path.is_file()), None)
    return result


def build_config(variant, seed, nv, dataset, overrides=(), epochs=None):
    from omegaconf import OmegaConf

    config = OmegaConf.load(APP / "configs" / V13_CONFIG)
    # Shared sweeps tune the full model. Named ablations must still remove their
    # declared factor, including when that same factor appears in --set.
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    for key, value in VARIANTS[variant][0].items():
        OmegaConf.update(config, key, value, merge=False)
    if epochs is not None:
        config.training.train_epochs = epochs
    if int(config.training.train_epochs) < 1 or int(config.training.batch_size) < 1:
        raise ValueError("Training epochs and batch size must be positive")
    if config.training.get("selection_metric", "val_loss") != "val_loss":
        raise ValueError("Primary selection remains val_loss; best_val_ssd is saved and reported separately")
    model = config.model
    if "max" in model.loss_fn.split("+"):
        raise ValueError("Use model.v12_tail_weight for the scheduled top-k experiment; do not also add +max")
    for key in ("v12_tail_weight", "v13_extra_mse_weight", "v13_raw_cos_weight"):
        value = float(model[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if float(model.v13_raw_cos_weight) > 0 and "cos" in model.loss_fn.split("+"):
        raise ValueError("Do not combine v13 raw cosine with the legacy centered 'cos' loss")
    for key in ("v12_tail_ramp_epochs", "v13_ramp_epochs", "mad_topk"):
        value = float(model[key])
        if not math.isfinite(value) or value < 1 or value != int(value):
            raise ValueError(f"{key} must be a positive integer")
    if not model.v12_local_gate and all(float(model[key]) == 0 for key in (
        "v12_tail_weight", "v13_extra_mse_weight", "v13_raw_cos_weight"
    )) and all(model[key] for key in (
        "v12_feature_dapp", "v12_unet_dapp", "v12_resconv", "v12_dual_head"
    )):
        raise ValueError("Original v7 is excluded from this suite; retain at least one v13 change")
    config.seed = seed
    config.exp_name = f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}"
    config.dataset.pkl_file = str(dataset)
    return config


def source_digests():
    files = list((APP / "models").glob("*.py"))
    # Opt-in packages preserve legacy launcher source inventories.
    files += list((APP / "models/lstm_v12").rglob("*.py"))
    files += list((APP / "models/lstm_v13").rglob("*.py"))
    files += [APP / "configs" / V13_CONFIG]
    files += [APP / name for name in ("mecge_table1_run_official_local_model.py", "mecge_table1_run_v13_model.py",
                                     "v11_experiments.py", "v13_experiments.py", "v12_results.py",
                                     "mecge_table1_collect_official_protocol.py", "utils_ecg.py",
                                     "mecge_table1_official_result_metrics.py", "mecge_table1_robustness_bins.py",
                                     "mecge_table1_collect_results.py")]
    return {str(path.relative_to(PROJECT)): file_digest(path) for path in sorted(files)}


def job_command(config_path, dataset, checkpoint, log, result, ssd_result, seed, device, resume=False, eval_only=False):
    command = [sys.executable, "-B", str(APP / "mecge_table1_run_v13_model.py"),
               "--config", str(config_path), "--dataset-pkl", str(dataset),
               "--checkpoint-dir", str(checkpoint), "--log-dir", str(log),
               "--output-pkl", str(result), "--ssd-output-pkl", str(ssd_result),
               "--seed", str(seed), "--device", device]
    if resume:
        command.append("--resume")
    if eval_only:
        command.append("--skip-train")
    return command


def run_job(args, root, variant, seed, nv, dataset, data_hash, sources, environment):
    from omegaconf import OmegaConf

    job = root / variant / f"nv{nv}" / f"seed{seed}"
    checkpoint = job / "checkpoint"
    name = f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}.pkl"
    result = root / "official_results" / name
    ssd_result = root / "validation_ssd/official_results" / name
    config = build_config(variant, seed, nv, dataset, args.set, args.epochs)
    config.root_dir, config.checkpoint_dir = str(job), str(checkpoint)
    config.log_dir, config.results_dir = str(job / "log"), str(job / "results")
    config_path = job / "resolved_config.yaml"
    command = job_command(config_path, dataset, checkpoint, job / "log", result, ssd_result,
                          seed, args.device, args.resume, args.eval_only)
    print(f"\n{variant} / nv{nv} / seed{seed}: {VARIANTS[variant][1]}", flush=True)
    if args.dry_run:
        print(shlex.join(command))
        return
    identity = {"schema": 1, "variant": variant, "seed": seed, "nv": nv,
                "config": OmegaConf.to_container(config, resolve=True),
                "dataset_sha256": data_hash, "sources": sources, "environment": environment}
    artifacts = (result, ssd_result, checkpoint / "best_model.pt", checkpoint / "best_val_ssd.pt")
    manifest_path = job / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise RuntimeError(f"Settings, data, code, or environment changed for {job}. Choose a new --run-root.")
        if manifest.get("complete"):
            # Evaluation may regenerate predictions, but may not silently bless
            # externally modified trained weights as the same experiment.
            for path in artifacts[2:]:
                if path.is_file() and file_digest(path) != manifest.get("artifacts_sha256", {}).get(str(path.relative_to(root))):
                    raise RuntimeError(f"Completed checkpoint changed: {path}; use a new --run-root")
        if manifest.get("complete") and all(path.is_file() for path in artifacts) and not args.eval_only:
            actual = {str(path.relative_to(root)): file_digest(path) for path in artifacts}
            if actual != manifest.get("artifacts_sha256"):
                raise RuntimeError(f"Completed results or checkpoints changed for {job}; use a new --run-root")
            print("Matching completed run found; skipping.", flush=True)
            return
    elif result.exists() or ssd_result.exists() or (job.exists() and any(job.iterdir())):
        raise RuntimeError(f"Unmanaged existing output found at {job}; choose a new --run-root.")
    state = checkpoint / "training_state.pt"
    if args.eval_only and not all(path.is_file() for path in artifacts[2:]):
        raise FileNotFoundError(f"Both best_model.pt and best_val_ssd.pt are required for evaluation: {checkpoint}")
    if state.exists() and not args.resume and not args.eval_only:
        raise RuntimeError(f"An interrupted run exists at {job}. Use --resume or a new --run-root.")
    if args.resume and not state.exists():
        command.remove("--resume")
    job.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_path, resolve=True)
    atomic_json(manifest_path, {"identity": identity, "command": command, "complete": False})
    subprocess.run(command, cwd=APP, check=True)
    if not all(path.is_file() for path in artifacts):
        raise RuntimeError(f"Training/evaluation did not produce both checkpoints and result PKLs: {job}")
    atomic_json(manifest_path, {"identity": identity, "command": command, "complete": True,
                               "artifacts_sha256": {str(path.relative_to(root)): file_digest(path) for path in artifacts}})


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, (_, description) in VARIANTS.items():
            print(f"{name:26s} {description}")
        return
    root = check_output_root(args.run_root)
    args.data_root = args.data_root.resolve()
    amplitudes = {} if args.dry_run else amplitude_paths(args)
    if args.collect_only:
        from v12_results import collect_results
        collect_results(root, amplitudes)
        return
    variants = select_variants(args)
    versions = (1, 2) if args.nv == "all" else (int(args.nv),)
    seeds = tuple(dict.fromkeys(args.seeds))
    datasets = {nv: dataset_path(args, nv) for nv in versions}
    # Resolve all requested configurations before starting any expensive work.
    for variant in variants:
        build_config(variant, seeds[0], versions[0], datasets[versions[0]], args.set, args.epochs)
    print(f"Plan: {len(variants)} variants x {len(versions)} noise versions x {len(seeds)} seeds")
    print("Each job exports best_loss and best_val_ssd separately; neither selector uses test metrics.")
    print("Only v13 and its ablations are run; compare combined NV1+NV2 with existing v7 results.")
    print(f"Output: {root}")
    environment, sources, hashes = {}, {}, {}
    if not args.dry_run:
        for path in datasets.values():
            if not path.is_file():
                raise FileNotFoundError(f"Dataset is missing: {path}. Point --data-root or --pkl-file at existing official data.")
        environment, sources = environment_info(args.device), source_digests()
        hashes = {nv: file_digest(path) for nv, path in datasets.items()}
    for variant in variants:
        for seed in seeds:
            for nv in versions:
                run_job(args, root, variant, seed, nv, datasets[nv], hashes.get(nv), sources, environment)
    if not args.dry_run:
        from v12_results import collect_results
        collect_results(root, amplitudes)


if __name__ == "__main__":
    main()
