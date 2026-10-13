"""Run eight matched v7-initialized v15 fine-tuning conditions without test selection."""

import argparse
import json
import math
import shlex
import shutil
import string
import subprocess
import sys
from pathlib import Path

from v11_experiments import atomic_json, environment_info, file_digest


PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "src"
V7_MODEL = "lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience"
V7_REGISTERED = "lstm_dualpath_dapp_cfm_unet_bd_ecg"
V15_CONFIG = "mecge_table1_repro_lstm_v15.yaml"
# Shared sweeps may change enabled weights. Disabled factors are forced to zero
# after the sweep, so a named removal cannot silently turn into the full model.
VARIANTS = {
    "v15_v7_continue": (False, False, False, False, "joint", "A: original v7 loss, matched fine-tuning budget"),
    "v15_cos_only": (True, False, False, False, "joint", "B: add only raw cosine"),
    "v15_topk_only": (False, True, False, False, "joint", "C: add only top-eight error"),
    "v15": (True, True, False, False, "joint", "D: raw cosine plus top-eight error"),
    "v15_extra_mse": (True, True, True, False, "joint", "E: D plus extra MSE; v13 no-local objective"),
    "v15_local_gate": (True, True, False, True, "joint", "F: D plus zero-initialized local gates"),
    "v15_extra_mse_local_gate": (True, True, True, True, "joint", "G: E plus local gates; MSE/gate interaction"),
    "v15_flow_only_finetune": (True, True, False, False, "flow_only", "H: D with the entire first stage frozen"),
}
TUNABLES = {
    "model.v13_raw_cos_weight", "model.v12_tail_weight", "model.v13_extra_mse_weight",
    "training.lr_backbone", "training.lr_decoder", "training.lr_flow",
    "training.batch_size", "training.train_epochs",
}


def _template_fields(value, allowed):
    try:
        fields = set()
        for _, field, spec, conversion in string.Formatter().parse(value):
            if field is not None:
                if field not in allowed or spec or conversion:
                    raise ValueError(f"Only literal placeholders {sorted(allowed)} are supported: {value}")
                fields.add(field)
        return fields
    except ValueError as error:
        raise ValueError(f"Invalid path template: {value}: {error}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(VARIANTS))
    parser.add_argument("--nv", choices=("1", "2", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/mecge_table1_repro")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/v15_performance")
    parser.add_argument("--v7-run-root", type=Path,
                        default=PROJECT / "runs/mecge_table1_repro" / V7_MODEL,
                        help="Original v7 model root containing checkpoint/<experiment>/<registered-model>/best_model.pt")
    parser.add_argument("--v7-checkpoint", help="Explicit v7 checkpoint; use {nv}/{seed} placeholders for multiple pairs")
    parser.add_argument("--pkl-file", help="Existing dataset path; use {nv} when selecting both noise versions")
    parser.add_argument("--rnd-test", help="Noise-amplitude array path, optionally containing {nv}")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Only auxiliary weights, three group LRs, batch size and epoch count are tunable")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--baseline-csv", nargs="+", type=Path)
    parser.add_argument("--baseline-robustness-csv", nargs="+", type=Path)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="Resume optimizer and RNG state of incomplete jobs")
    restart.add_argument("--eval-only", action="store_true", help="Re-export the existing qualified checkpoint and v7 reference")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if args.collect_only and (args.dry_run or args.resume or args.eval_only):
        parser.error("--collect-only cannot be combined with --dry-run, --resume, or --eval-only")
    try:
        if args.pkl_file:
            fields = _template_fields(args.pkl_file, {"nv"})
            if args.nv == "all" and "nv" not in fields:
                raise ValueError("--pkl-file must contain {nv} with --nv all")
        if args.rnd_test:
            _template_fields(args.rnd_test, {"nv"})
        if args.v7_checkpoint:
            fields = _template_fields(args.v7_checkpoint, {"nv", "seed"})
            if args.nv == "all" and "nv" not in fields:
                raise ValueError("--v7-checkpoint must contain {nv} for multiple noise versions")
            if len(set(args.seeds)) > 1 and "seed" not in fields:
                raise ValueError("--v7-checkpoint must contain {seed} for multiple training seeds")
        for override in args.set:
            key, separator, _ = override.partition("=")
            if not separator or key not in TUNABLES:
                raise ValueError(f"Unknown or unsupported --set key: {key}")
    except ValueError as error:
        parser.error(str(error))
    return args


def select_variants(args):
    return tuple(dict.fromkeys(args.models or VARIANTS))


def versions_for(args):
    return (1, 2) if args.nv == "all" else (int(args.nv),)


def check_output_root(root, extra_protected=()):
    root = root.resolve()
    protected = [PROJECT / name for name in (
        "runs/mecge_table1_repro", "runs/v11_performance", "runs/v12_performance",
        "runs/v13_performance", "runs/v14_performance", "runs/stage_waveform_analysis",
        "data", "src", "scripts", "docs", "tests", "references", ".git", ".agents", ".codex",
    )]
    for path in [*protected, *extra_protected]:
        path = Path(path).resolve()
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError(f"Output root overlaps a protected legacy/data/source/input path: {root} / {path}")
    return root


def dataset_path(args, nv):
    value = args.pkl_file.format(nv=nv) if args.pkl_file else args.data_root / "raw" / f"dataset_bw_nv{nv}.pkl"
    return Path(value).resolve()


def initialization_path(args, nv, seed):
    if args.v7_checkpoint:
        return Path(args.v7_checkpoint.format(nv=nv, seed=seed)).resolve()
    exp = f"{V7_MODEL}__qtdb_train_qtdb_test__nv{nv}__seed{seed}"
    return (args.v7_run_root / "checkpoint" / exp / V7_REGISTERED / "best_model.pt").resolve()


def amplitude_paths(args):
    if args.skip_robustness:
        return {}
    result = {}
    for nv in versions_for(args):
        if args.rnd_test:
            path = Path(args.rnd_test.format(nv=nv)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Requested noise-amplitude array is missing: {path}")
            result[nv] = path
        else:
            options = [args.data_root / "raw" / f"rnd_test_nv{nv}.npy", args.data_root / "raw/rnd_test.npy",
                       PROJECT / "references/MECG-E" / f"rnd_test_nv{nv}.npy",
                       PROJECT / "references/MECG-E/rnd_test.npy"]
            result[nv] = next((path.resolve() for path in options if path.is_file()), None)
    return result


def build_config(variant, seed, nv, dataset, overrides=(), epochs=None):
    from omegaconf import OmegaConf

    config = OmegaConf.load(APP / "configs" / V15_CONFIG)
    for override in overrides:
        key, separator, _ = override.partition("=")
        if not separator or key not in TUNABLES:
            raise ValueError(f"Unknown or unsupported --set key: {key}")
    # Extra MSE is enabled only in E/G; a shared value tunes those conditions.
    config.model.v13_extra_mse_weight = 0.25
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    for key in TUNABLES:
        value = OmegaConf.select(config, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        integer = key in {"training.train_epochs", "training.batch_size"}
        positive = key.startswith("training.")
        if (positive and value <= 0) or (not positive and value < 0) or (integer and int(value) != value):
            raise ValueError(f"Invalid value for {key}: {value}")
    raw_cos, topk, mse, local, scope, _ = VARIANTS[variant]
    for enabled, key in ((raw_cos, "v13_raw_cos_weight"), (topk, "v12_tail_weight"),
                         (mse, "v13_extra_mse_weight")):
        if not enabled:
            config.model[key] = 0.0
    config.model.v12_local_gate = local
    config.training.finetune_scope = scope
    if epochs is not None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        config.training.train_epochs = epochs
    config.seed = seed
    config.exp_name = f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}"
    config.dataset.pkl_file = str(dataset)
    return config


def source_digests():
    # Include all Python dependencies, including opt-in model packages and the
    # collector; unrelated legacy files are read, never modified by this suite.
    files = list(APP.rglob("*.py"))
    files += [APP / "configs" / V15_CONFIG, PROJECT / "scripts/run_v15_ablation.sh"]
    return {str(path.relative_to(PROJECT)): file_digest(path) for path in sorted(files)}


def job_command(config_path, dataset, initialization, checkpoint, log, result, baseline_result,
                seed, device, resume=False, eval_only=False):
    command = [sys.executable, "-B", str(APP / "mecge_table1_run_v15_model.py"),
               "--config", str(config_path), "--dataset-pkl", str(dataset),
               "--init-checkpoint", str(initialization), "--checkpoint-dir", str(checkpoint),
               "--log-dir", str(log), "--output-pkl", str(result),
               "--baseline-output-pkl", str(baseline_result), "--seed", str(seed), "--device", device]
    if resume:
        command.append("--resume")
    if eval_only:
        command.append("--skip-train")
    return command


def prepare_job(args, root, variant, seed, nv, dataset, initialization, data_hash=None,
                init_hash=None, sources=None, environment=None):
    from omegaconf import OmegaConf

    job = root / variant / f"nv{nv}" / f"seed{seed}"
    checkpoint = job / "checkpoint"
    result = root / "official_results" / f"lstm_{variant}__qtdb_train_qtdb_test__nv{nv}__seed{seed}.pkl"
    baseline_name = f"{V7_MODEL}__qtdb_train_qtdb_test__nv{nv}__seed{seed}.pkl"
    shared_baseline = root / "baseline_v7/official_results" / baseline_name
    baseline_manifest = root / "baseline_v7/provenance" / f"nv{nv}__seed{seed}.json"
    for path in (job, checkpoint, result, shared_baseline, baseline_manifest):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Output path escapes run root through a symlink: {path}")
    config = build_config(variant, seed, nv, dataset, args.set, args.epochs)
    config.root_dir, config.checkpoint_dir = str(job), str(checkpoint)
    config.data_dir = str(args.data_root)
    config.log_dir, config.results_dir = str(job / "log"), str(job / "results")
    config_path = job / "resolved_config.yaml"
    identity = {"schema": 1, "variant": variant, "seed": seed, "nv": nv,
                "config": OmegaConf.to_container(config, resolve=True),
                "dataset": str(dataset), "dataset_sha256": data_hash,
                "init_checkpoint": str(initialization), "init_checkpoint_sha256": init_hash,
                "sources": sources or {}, "environment": environment or {}}
    baseline_identity = {"seed": seed, "nv": nv, "dataset": str(dataset), "dataset_sha256": data_hash,
                         "init_checkpoint": str(initialization), "init_checkpoint_sha256": init_hash}
    return {"root": root, "job": job, "variant": variant, "seed": seed, "nv": nv,
            "dataset": dataset, "initialization": initialization, "checkpoint": checkpoint,
            "config": config, "config_path": config_path, "result": result,
            "baseline_result": checkpoint / "baseline_test.pkl", "shared_baseline": shared_baseline,
            "baseline_manifest": baseline_manifest, "baseline_identity": baseline_identity,
            "manifest_path": job / "manifest.json", "identity": identity}


def _verify_baseline(plan):
    path, manifest_path = plan["shared_baseline"], plan["baseline_manifest"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != plan["baseline_identity"]:
            raise RuntimeError(f"Shared v7 baseline data/initialization changed: {manifest_path}")
        if not path.is_file() or file_digest(path) != manifest.get("sha256"):
            raise RuntimeError(f"Shared v7 baseline was removed or changed: {path}")
    elif path.exists():
        raise RuntimeError(f"Unmanaged shared v7 baseline: {path}")


def inspect_job(args, plan):
    """Validate every planned job before launching the first expensive process."""
    _verify_baseline(plan)
    job, manifest_path = plan["job"], plan["manifest_path"]
    if not manifest_path.exists():
        if plan["result"].exists() or (job.exists() and any(job.iterdir())):
            raise RuntimeError(f"Unmanaged output exists at {job}; choose a new --run-root")
        if args.eval_only:
            raise FileNotFoundError(f"No managed run to evaluate: {job}")
        return "run"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != plan["identity"]:
        raise RuntimeError(f"Settings, data, initial checkpoint, code, or environment changed for {job}; choose a new --run-root")
    if manifest.get("complete"):
        expected = manifest.get("artifacts_sha256", {})
        if not expected:
            raise RuntimeError(f"Completed run has no artifact inventory: {job}")
        for relative, digest in expected.items():
            path = plan["root"] / relative
            if not path.resolve().is_relative_to(plan["root"].resolve()):
                raise RuntimeError(f"Artifact path escapes output root: {relative}")
            # Eval may regenerate a missing exported PKL, but cannot overwrite
            # an altered file or bless externally changed trained weights.
            if args.eval_only and not path.exists() and path.suffix == ".pkl" and path != plan["shared_baseline"]:
                continue
            if not path.is_file() or file_digest(path) != digest:
                raise RuntimeError(f"Completed artifact changed or is missing: {path}")
        return "run" if args.eval_only else "skip"
    if not args.resume and not args.eval_only:
        raise RuntimeError(f"An incomplete run exists at {job}; use --resume or a new --run-root")
    if args.eval_only and not (plan["checkpoint"] / "validation_selection.json").is_file():
        raise FileNotFoundError(f"No saved validation selection to evaluate: {job}")
    return "run"


def publish_baseline(plan):
    _verify_baseline(plan)
    digest = file_digest(plan["baseline_result"])
    if plan["shared_baseline"].exists():
        if file_digest(plan["shared_baseline"]) != digest:
            raise RuntimeError("Frozen-v7 export differs across matched variants; existing shared baseline was preserved")
    else:
        plan["shared_baseline"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plan["baseline_result"], plan["shared_baseline"])
        atomic_json(plan["baseline_manifest"], {"identity": plan["baseline_identity"], "sha256": digest})


def run_job(args, plan, action="run"):
    from omegaconf import OmegaConf

    print(f"\n{plan['variant']} / nv{plan['nv']} / seed{plan['seed']}: {VARIANTS[plan['variant']][-1]}", flush=True)
    if action == "skip":
        print("Matching completed experiment found; skipping (including no-feasible outcomes).", flush=True)
        return
    state = plan["checkpoint"] / "training_state.pt"
    command = job_command(plan["config_path"], plan["dataset"], plan["initialization"], plan["checkpoint"],
                          plan["job"] / "log", plan["result"], plan["baseline_result"], plan["seed"],
                          args.device, args.resume and state.exists(), args.eval_only)
    if args.dry_run:
        print(f"Initialization: {plan['initialization']}")
        print(shlex.join(command))
        return
    # Files can change after preflight; do not train against a stale identity.
    for path, expected in ((plan["dataset"], plan["identity"]["dataset_sha256"]),
                           (plan["initialization"], plan["identity"]["init_checkpoint_sha256"])):
        if file_digest(path) != expected:
            raise RuntimeError(f"Input changed after preflight: {path}")
    plan["job"].mkdir(parents=True, exist_ok=True)
    OmegaConf.save(plan["config"], plan["config_path"], resolve=True)
    atomic_json(plan["manifest_path"], {"identity": plan["identity"], "command": command, "complete": False})
    subprocess.run(command, cwd=APP, check=True)
    selection_path = plan["checkpoint"] / "validation_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    status = selection.get("status")
    if status not in {"selected", "no_feasible_checkpoint"}:
        raise RuntimeError(f"Unexpected validation selection status: {status!r}")
    if selection.get("complete") is not True:
        raise RuntimeError("Runner did not finish the declared training budget")
    required = [plan["checkpoint"] / name for name in (
        "baseline.pt", "model_last.pt", "training_state.pt", "loss_history.csv")]
    required += [selection_path, plan["baseline_result"], plan["config_path"]]
    if status == "selected":
        required += [plan["checkpoint"] / "best_model.pt", plan["result"]]
    elif plan["result"].exists() or (plan["checkpoint"] / "best_model.pt").exists():
        raise RuntimeError("No-feasible experiment unexpectedly has a primary result/best checkpoint")
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Training/evaluation did not produce the required artifact: {path}")
    # Bind epoch and per-window diagnostics to the same completed experiment.
    required = list(dict.fromkeys([*required, *plan["checkpoint"].rglob("*.json"),
                                   *plan["checkpoint"].rglob("*.csv"),
                                   *plan["checkpoint"].rglob("*.npz")]))
    publish_baseline(plan)
    required += [plan["shared_baseline"], plan["baseline_manifest"]]
    atomic_json(plan["manifest_path"], {"identity": plan["identity"], "command": command,
                "complete": True, "status": status,
                "artifacts_sha256": {str(path.relative_to(plan["root"])): file_digest(path) for path in required}})


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, values in VARIANTS.items():
            print(f"{name:30s} {values[-1]}")
        return
    args.data_root, args.v7_run_root = args.data_root.resolve(), args.v7_run_root.resolve()
    variants, versions, seeds = select_variants(args), versions_for(args), tuple(dict.fromkeys(args.seeds))
    datasets = {nv: dataset_path(args, nv) for nv in versions}
    initializations = {(nv, seed): initialization_path(args, nv, seed) for nv in versions for seed in seeds}
    root = check_output_root(args.run_root, [args.data_root, args.v7_run_root, *datasets.values(), *initializations.values()])
    amplitudes = {} if args.dry_run else amplitude_paths(args)
    if args.collect_only:
        from v15_results import collect_results
        return collect_results(root, amplitudes, expected_models=[f"lstm_{name}" for name in variants],
                               expected_seeds=seeds, baseline_csvs=args.baseline_csv,
                               baseline_robustness_csvs=args.baseline_robustness_csv)
    # Validate configs and all requested input paths before creating any output.
    for name in variants:
        build_config(name, seeds[0], versions[0], datasets[versions[0]], args.set, args.epochs)
    environment, sources, data_hashes, init_hashes = {}, {}, {}, {}
    if not args.dry_run:
        missing = [path for path in [*datasets.values(), *initializations.values()] if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required existing inputs are missing (nothing was started):\n" + "\n".join(map(str, missing)))
        environment, sources = environment_info(args.device), source_digests()
        data_hashes = {nv: file_digest(path) for nv, path in datasets.items()}
        init_hashes = {key: file_digest(path) for key, path in initializations.items()}
    plans = [prepare_job(args, root, variant, seed, nv, datasets[nv], initializations[nv, seed],
                         data_hashes.get(nv), init_hashes.get((nv, seed)), sources, environment)
             for variant in variants for seed in seeds for nv in versions]
    actions = ["run" if args.dry_run else inspect_job(args, plan) for plan in plans]
    print(f"Plan: {len(variants)} variants x {len(versions)} noise versions x {len(seeds)} seeds")
    print("Matched v7 initialization; four validation metrics must qualify before a new primary checkpoint is exported.")
    print(f"Output: {root}")
    for plan, action in zip(plans, actions):
        run_job(args, plan, action)
    if not args.dry_run:
        from v15_results import collect_results
        collect_results(root, amplitudes, expected_models=[f"lstm_{name}" for name in variants],
                        expected_seeds=seeds, baseline_csvs=args.baseline_csv,
                        baseline_robustness_csvs=args.baseline_robustness_csv)


if __name__ == "__main__":
    main()
