"""Reproducible v17 controls, cooperation ablations and velocity backbones."""
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from v11_experiments import atomic_json, environment_info, file_digest
from v16_experiments import _template_fields, check_output_root as legacy_root_check

PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "src"
VARIANTS = {
    "v7_max_control": ({}, "Unmodified v7 + top-k; retrain with the matched v17 runner"),
    "v12_topk_control": ({}, "Unmodified v12 local gate + v7 top-k"),
    "v12_topk_compact": ({}, "Same old cooperation/losses; exact parameter compaction"),
    "v17": ({}, "Dual candidates + one independent flow field + two effective local gates"),
    "v17_no_baseline_fusion": ({"v17_independent_paths": False}, "Baseline head conditions flow only; remove coarse fusion gate"),
    "v17_condition_gate": ({"v17_correction_gate": False}, "Gate sees coarse conditions only; remove correction input columns"),
    "v17_no_local_gate": ({"v17_local_gate": False}, "Global control only"),
    "v17_no_topk": ({"lambda_max": 0.0}, "Remove top-k objective"),
    "v17_no_flow": ({"v17_flow": False}, "Retrain with two coarse candidates and blend gate; remove flow and amplitude gates"),
    "v17_no_dual_head": ({"v17_dual_head": False}, "Remove coarse dual head, its losses and condition channel"),
    "v17_fixed_gate": ({"v17_fixed_gate": True}, "Fixed initial gates; remove all learned gate parameters"),
    "v17_tcn": ({"v17_backbone": "tcn"}, "Conditional dilated residual TCN velocity field; retain FM"),
    "v17_half_flow": ({"cfm_unet_base_channels": 16}, "Half-width U-Net capacity control"),
    "v17_four_steps": ({"cfm_inference_steps": 4}, "Four Euler steps in training reconstruction and inference"),
    "v17_bridge": ({"v17_bridge_scale": 0.02}, "Stochastic bridge with its correct derivative target"),
    "v17_direct": ({"v17_fm_objective": False}, "Direct residual regression; fold constant time and remove state/FM-only parameters"),
}
SUITES = {
    "core": tuple(VARIANTS)[:7],
    "modules": ("v17", "v17_no_topk", "v17_no_flow", "v17_no_dual_head", "v17_fixed_gate", "v17_direct"),
    "backbones": ("v17", "v17_tcn", "v17_half_flow", "v17_four_steps", "v17_bridge"),
    "all": tuple(VARIANTS),
}
# Factor-defining flags cannot be changed behind an experiment's name.
TUNABLES = {"training.lr", "training.batch_size", "training.train_epochs",
            "model.dense_channel", "model.lstm_hidden", "model.num_tscblocks",
            "model.dual_noise_head_hidden", "model.cfm_unet_base_channels",
            "model.cfm_unet_channel_mults", "model.cfm_unet_time_dim",
            "model.v17_tcn_channels", "model.v17_gate_hidden"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--models", nargs="+", choices=tuple(VARIANTS))
    choice.add_argument("--suite", choices=tuple(SUITES), default=None)
    parser.add_argument("--nv", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/mecge_table1_repro")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/v17_performance")
    parser.add_argument("--pkl-file", help="Existing official dataset; use {nv} for both noise versions")
    parser.add_argument("--rnd-test", help="Matched amplitude array; optionally use {nv}")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    for flag in ("list", "dry-run", "collect-only", "skip-robustness"):
        parser.add_argument("--"+flag, action="store_true")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true")
    restart.add_argument("--eval-only", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0,2**32)")
    if args.collect_only and (args.resume or args.eval_only or args.dry_run):
        parser.error("--collect-only cannot be combined with resume/eval-only/dry-run")
    try:
        if args.pkl_file:
            fields = _template_fields(args.pkl_file, {"nv"})
            if args.nv == "all" and "nv" not in fields:
                raise ValueError("--pkl-file requires {nv} when --nv all")
        if args.rnd_test:
            _template_fields(args.rnd_test, {"nv"})
        for override in args.set:
            key, separator, _ = override.partition("=")
            if not separator or key not in TUNABLES:
                raise ValueError(f"Unsupported --set key: {key}")
    except ValueError as error:
        parser.error(str(error))
    return args


def select_variants(args):
    return tuple(dict.fromkeys(args.models or SUITES[args.suite or "core"]))


def check_output_root(root, inputs=()):
    return legacy_root_check(root, [PROJECT / "runs/v16_performance", *inputs])


def dataset_path(args, nv):
    return Path(args.pkl_file.format(nv=nv) if args.pkl_file else args.data_root / "raw" / f"dataset_bw_nv{nv}.pkl").resolve()


def amplitude_paths(args):
    if args.skip_robustness:
        return {}
    result = {}
    for nv in ((1, 2) if args.nv == "all" else (int(args.nv),)):
        if args.rnd_test:
            path = Path(args.rnd_test.format(nv=nv)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing amplitude array: {path}")
        else:
            # Never silently attach a reference amplitude file to arbitrary data.
            choices = [args.data_root / "raw" / f"rnd_test_nv{nv}.npy", args.data_root / "raw/rnd_test.npy"]
            path = next((p.resolve() for p in choices if p.is_file()), None)
        result[nv] = path
    return result


def build_config(variant, seed, nv, dataset, overrides=(), epochs=None):
    from omegaconf import OmegaConf
    base = OmegaConf.load(APP / "configs/mecge_table1_repro_lstm_v17.yaml")
    if variant in {"v7_max_control", "v12_topk_control", "v12_topk_compact"}:
        legacy = OmegaConf.load(APP / "configs/mecge_table1_repro_lstm_v12.yaml")
        base.model = legacy.model
        base.model.loss_fn = base.model.loss_fn + "+max"
        base.model.lambda_max, base.model.mad_topk = 0.05, 8
        base.model.v12_local_gate = variant != "v7_max_control"
        base.model.v12_tail_weight = 0.0
        base.model_name = "lstm_v17_compact_control_ecg" if variant == "v12_topk_compact" else "lstm_temporal_gate_v12_ecg"
    for override in overrides:
        key, separator, _ = override.partition("=")
        if not separator or key not in TUNABLES:
            raise ValueError(f"Unsupported --set key: {key}")
    base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    for key, value in VARIANTS[variant][0].items():
        if variant == "v17_half_flow" and key == "cfm_unet_base_channels":
            value = max(2, int(base.model.cfm_unet_base_channels)//2)
        OmegaConf.update(base, "model."+key, value)
    if epochs is not None:
        base.training.train_epochs = epochs
    for key in TUNABLES:
        value = OmegaConf.select(base, key)
        if value is None:
            continue
        values = list(value) if OmegaConf.is_list(value) else [value]
        for number in values:
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"{key} must contain finite positive numbers")
            if key != "training.lr" and type(number) is not int:
                raise ValueError(f"{key} requires integers")
    if base.model.cfm_unet_time_dim < 4 or base.model.cfm_unet_time_dim % 2:
        raise ValueError("The sinusoidal time dimension must be even and >=4")
    if len(base.model.cfm_unet_channel_mults) < 1:
        raise ValueError("U-Net must have at least one level")
    base.seed, base.exp_name = seed, f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}"
    base.dataset.pkl_file = str(dataset)
    return base


def source_digests():
    files = list(APP.rglob("*.py")) + [APP / "configs" / name for name in (
        "mecge_table1_repro_lstm_v17.yaml", "mecge_table1_repro_lstm_v12.yaml")]
    files += [PROJECT / "scripts/run_v17_ablation.sh"]
    return {str(path.relative_to(PROJECT)): file_digest(path) for path in sorted(files)}


def run_job(args, root, variant, seed, nv, dataset, identity_common):
    from omegaconf import OmegaConf
    job = root / variant / f"nv{nv}" / f"seed{seed}"
    checkpoint = job / "checkpoint"
    name = f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}.pkl"
    result, secondary = root / "official_results" / name, root / "validation_loss/official_results" / name
    config = build_config(variant, seed, nv, dataset, args.set, args.epochs)
    config.root_dir, config.checkpoint_dir = str(job), str(checkpoint)
    config.log_dir, config.results_dir = str(job / "log"), str(job / "results")
    path = job / "resolved_config.yaml"
    command = [sys.executable, "-B", str(APP / "mecge_table1_run_v17_model.py"),
               "--config", str(path), "--dataset-pkl", str(dataset), "--checkpoint-dir", str(checkpoint),
               "--log-dir", str(job / "log"), "--output-pkl", str(result), "--loss-output-pkl", str(secondary),
               "--seed", str(seed), "--device", args.device]
    if args.dry_run:
        print(shlex.join(command))
        return
    identity = dict(identity_common, schema=17, variant=variant, seed=seed, nv=nv,
                    config=OmegaConf.to_container(config, resolve=True), dataset_sha256=file_digest(dataset))
    manifest_path = job / "manifest.json"
    checkpoints = [checkpoint / name for name in ("best_model.pt", "best_loss.pt")]
    artifacts = [result, secondary, *checkpoints, checkpoint / "parameter_audit.json",
                 checkpoint / "loss_history.csv", checkpoint / "selection.json", checkpoint / "profile.json"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if identity != manifest.get("identity"):
            raise RuntimeError(f"Settings/data/source/environment changed for {job}; use a new --run-root")
        if manifest.get("complete"):
            check = checkpoints if args.eval_only else artifacts
            for artifact in check:
                expected = manifest.get("artifacts_sha256", {}).get(str(artifact.relative_to(root)))
                if not artifact.is_file() or file_digest(artifact) != expected:
                    raise RuntimeError(f"Completed artifact missing or changed: {artifact}")
            if not args.eval_only:
                print(f"Matching completed run found; skipping {variant}/nv{nv}/seed{seed}", flush=True)
                return
    elif result.exists() or secondary.exists() or (job.exists() and any(job.iterdir())):
        raise RuntimeError(f"Unmanaged existing output at {job}; use a new --run-root")
    state = checkpoint / "training_state.pt"
    if args.eval_only:
        if not all(p.is_file() for p in checkpoints):
            raise FileNotFoundError("Both selected checkpoints are required for --eval-only")
        command.append("--skip-train")
    elif state.is_file():
        if not args.resume:
            raise RuntimeError(f"Interrupted run found at {job}; use --resume")
        command.append("--resume")
    job.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path, resolve=True)
    atomic_json(manifest_path, {"identity": identity, "command": command, "complete": False})
    subprocess.run(command, cwd=APP, check=True)
    if not all(p.is_file() for p in artifacts):
        raise RuntimeError(f"Missing expected v17 artifacts: {job}")
    atomic_json(manifest_path, {"identity": identity, "command": command, "complete": True,
                               "artifacts_sha256": {str(p.relative_to(root)): file_digest(p) for p in artifacts}})


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, (_, description) in VARIANTS.items():
            print(f"{name:26s} {description}")
        return
    args.data_root = args.data_root.resolve()
    versions = (1, 2) if args.nv == "all" else (int(args.nv),)
    datasets = {nv: dataset_path(args, nv) for nv in versions}
    root = check_output_root(args.run_root, [args.data_root, *datasets.values()])
    amplitudes = {} if args.dry_run else amplitude_paths(args)
    if args.collect_only:
        from v17_results import collect_results
        collect_results(root, amplitudes)
        return
    variants, seeds = select_variants(args), tuple(dict.fromkeys(args.seeds))
    for variant in variants:
        build_config(variant, seeds[0], versions[0], datasets[versions[0]], args.set, args.epochs)
    print(f"Plan: {len(variants)} variants x {len(versions)} noise versions x {len(seeds)} seeds", flush=True)
    print(f"Output: {root}; primary=full validation SSD; secondary=full validation loss", flush=True)
    common = {}
    if not args.dry_run:
        for path in datasets.values():
            if not path.is_file():
                raise FileNotFoundError(f"Dataset missing: {path}; set --data-root or --pkl-file")
        common = {"sources": source_digests(), "environment": environment_info(args.device)}
    for variant in variants:
        for seed in seeds:
            for nv in versions:
                run_job(args, root, variant, seed, nv, datasets[nv], common)
    if not args.dry_run:
        from v17_results import collect_results
        collect_results(root, amplitudes)


if __name__ == "__main__":
    main()
