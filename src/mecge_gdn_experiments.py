"""M0/M1/M2 BW-only ablations using the mecg_e_no_early_stop protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "src"
VARIANTS = {"m0": ("Bi-Mamba", "Bi-Mamba"), "m1": ("Bi-GDN2", "Bi-Mamba"), "m2": ("Bi-GDN2", "Bi-GDN2")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--nv", choices=("1", "2", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/mecge_table1_repro")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/mecge_gdn_ablation")
    parser.add_argument("--mecge-dir", type=Path, default=PROJECT / "references/MECG-E")
    parser.add_argument("--gdn-dir", type=Path, default=PROJECT / "references/GatedDeltaNet-2")
    parser.add_argument("--pkl-file", help="BW dataset path; use {nv} for --nv all")
    parser.add_argument("--rnd-test", help="Matched amplitude array; optionally use {nv}")
    for flag in ("list", "dry-run", "collect-only"):
        parser.add_argument("--" + flag, action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    args = parser.parse_args(argv)
    if args.collect_only and (args.resume or args.eval_only or args.dry_run):
        parser.error("--collect-only cannot be combined with --resume, --eval-only or --dry-run")
    if args.pkl_file and args.nv == "all" and "{nv}" not in args.pkl_file:
        parser.error("--pkl-file requires {nv} for --nv all")
    for template in (args.pkl_file, args.rnd_test):
        if template:
            try:
                template.format(nv=1)
            except (KeyError, ValueError, IndexError) as error:
                parser.error(f"Invalid path template: {error}")
    args.models = list(dict.fromkeys(args.models))
    return args


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def versions(args):
    return (1, 2) if args.nv == "all" else (int(args.nv),)


def dataset_path(args, nv):
    return Path(args.pkl_file.format(nv=nv) if args.pkl_file else args.data_root / "raw" / f"dataset_bw_nv{nv}.pkl").resolve()


def amplitude_path(args, nv):
    if args.rnd_test:
        return Path(args.rnd_test.format(nv=nv)).resolve()
    choices = (args.data_root / "raw" / f"rnd_test_nv{nv}.npy", args.data_root / "raw/rnd_test.npy")
    return next((p.resolve() for p in choices if p.is_file()), choices[0].resolve())


def paths(root, variant, nv):
    name = f"mecge_gdn_{variant}__qtdb_train_qtdb_test__nv{nv}__seed3407"
    job = root / f"mecge_gdn_{variant}"
    checkpoint = job / "checkpoint" / name / "mecg_e"
    return dict(name=name, job=job, checkpoint=checkpoint, log=job / "log" / name / "mecg_e",
                manifest=job / "manifests" / f"{name}.json",
                config=job / "configs" / f"{name}.yaml",
                result=root / "official_results" / f"{name}.pkl",
                metrics=job / "results" / name / "mecg_e" / "best_loss")


def training_command(args, variant, nv):
    p = paths(args.run_root, variant, nv)
    command = [sys.executable, "-B", str(APP / "mecge_table1_run_official_reference.py"),
               "--mecge-dir", str(args.mecge_dir.resolve()), "--config", str(p["config"]),
               "--backbone", variant, "--gdn-dir", str(args.gdn_dir.resolve()),
               "--n-type", "bw", "--nv", str(nv), "--dataset-pkl", str(dataset_path(args, nv)),
               "--device", args.device, "--output-pkl", str(p["result"]),
               "--checkpoint-dir", str(p["checkpoint"]), "--log-dir", str(p["log"]),
               "--max-epochs", "30", "--early-stopping-patience", "none"]
    if args.resume:
        command.append("--resume")
    if args.eval_only:
        command.append("--skip-train")
    return command


def run(command):
    print(shlex.join([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=APP, check=True)


def resolved_config(args, variant):
    import yaml
    config = yaml.safe_load((args.mecge_dir / "config/MECGE_phase.yaml").read_text())
    config["train"].update(epochs=30, early_stopping_patience_epochs=None)
    # These are audit metadata only; the official pipeline consumes train/model.
    config["ablation"] = dict(variant=variant, temporal=VARIANTS[variant][0], frequency=VARIANTS[variant][1],
                              seed=3407, noise="bw", selection="best_validation_loss",
                              gdn=dict(num_heads=4, head_dim=32, expand_v=1, conv_size=4))
    return config


def source_hashes(args):
    files = [APP / n for n in ("mecge_gdn_experiments.py", "mecge_gdn_backbone.py",
             "mecge_table1_run_official_reference.py", "mecge_table1_official_result_metrics.py",
             "mecge_table1_robustness_bins.py", "mecge_table1_collect_results.py",
             "mecge_table1_collect_official_protocol.py", "utils_ecg.py")]
    files += sorted(args.mecge_dir.rglob("*.py"))
    if any(m != "m0" for m in args.models):
        files += [args.gdn_dir / "lit_gpt/gdn2.py"]
        files += sorted((args.gdn_dir / "lit_gpt/gdn2_ops").glob("*.py"))
    return {str(p.resolve()): digest(p) for p in files}


def prepare_job(args, variant, nv, sources):
    p = paths(args.run_root, variant, nv)
    config = resolved_config(args, variant)
    # M0 identity does not depend on whether other variants are selected.
    model_sources = {k: v for k, v in sources.items() if variant != "m0" or not Path(k).is_relative_to(args.gdn_dir.resolve())}
    identity = dict(config=config, sources=model_sources, device=args.device,
                    dataset_sha256=digest(dataset_path(args, nv)), amplitude_sha256=digest(amplitude_path(args, nv)))
    old = json.loads(p["manifest"].read_text()) if p["manifest"].exists() else None
    if old and old["identity"] != identity:
        raise RuntimeError(f"Configuration/source/data changed: {p['manifest']}; use a new --run-root")
    artifacts = (p["result"], p["checkpoint"] / "best_model.pt")
    if old and old.get("complete"):
        for artifact in artifacts:
            if not artifact.is_file() or digest(artifact) != old["artifacts"][str(artifact)]:
                raise RuntimeError(f"Completed artifact changed or missing: {artifact}")
        if not args.eval_only:
            return p, config, identity, True
    if args.eval_only:
        if old is None or not artifacts[1].is_file():
            raise FileNotFoundError(f"Managed best_model.pt required for --eval-only: {artifacts[1]}")
    elif old and not old.get("complete") and (p["checkpoint"] / "training_state.pt").exists() and not args.resume:
        raise RuntimeError(f"Interrupted run: {p['checkpoint']}; use --resume")
    if old is None and (p["result"].exists() or (p["checkpoint"].exists() and any(p["checkpoint"].iterdir()))):
        raise RuntimeError(f"Unmanaged output already exists: {p['checkpoint']}")
    return p, config, identity, False


def collect(args):
    for variant in args.models:
        for nv in versions(args):
            p = paths(args.run_root, variant, nv)
            if not p["result"].is_file():
                raise FileNotFoundError(p["result"])
            run([sys.executable, "-B", APP / "mecge_table1_official_result_metrics.py",
                 "--result-pkl", p["result"], "--output-dir", p["metrics"]])
            run([sys.executable, "-B", APP / "mecge_table1_robustness_bins.py",
                 "--metrics-per-window", p["metrics"] / "metrics_per_window.csv",
                 "--rnd-test", amplitude_path(args, nv),
                 "--output-root", p["job"] / "controlled_tests", "--result-model", f"mecge_gdn_{variant}",
                 "--noise-version", f"nv{nv}", "--seed", "3407"])
    for nv in (*[f"nv{n}" for n in versions(args)], "all"):
        run([sys.executable, "-B", APP / "mecge_table1_collect_results.py",
             "--run-root", args.run_root, "--noise-version", nv, "--output-dir", args.run_root / "analysis"])
    if versions(args) == (1, 2):
        run([sys.executable, "-B", APP / "mecge_table1_collect_official_protocol.py",
             "--official-results-dir", args.run_root / "official_results",
             "--rnd-test-nv1", amplitude_path(args, 1), "--rnd-test-nv2", amplitude_path(args, 2),
             "--output-dir", args.run_root / "analysis"])


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, (time, frequency) in VARIANTS.items():
            print(f"{name}: time={time}, frequency={frequency}")
        return
    args.run_root = args.run_root.resolve()
    # Dedicated root; never mix this suite with previous experiment outputs.
    protected = [PROJECT / "src", PROJECT / "references", PROJECT / "data", PROJECT / "docs",
                 args.data_root.resolve(), args.mecge_dir.resolve(), args.gdn_dir.resolve()]
    if args.run_root in (PROJECT, PROJECT / "runs") or any(
        args.run_root.is_relative_to(p) or p.is_relative_to(args.run_root) for p in protected
    ):
        raise ValueError("--run-root must be a separate experiment output directory")
    print(f"M0/M1/M2 BW: {len(args.models)} models x {len(versions(args))} noise versions; seed=3407, epochs=30, best validation loss", flush=True)
    if args.dry_run:
        for variant in args.models:
            for nv in versions(args):
                print(shlex.join(training_command(args, variant, nv)))
        return
    for nv in versions(args):
        for path in (amplitude_path(args, nv),) if args.collect_only else (dataset_path(args, nv), amplitude_path(args, nv)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")
    if args.collect_only:
        collect(args)
        return
    import yaml
    sources = source_hashes(args)
    # Validate every job before writing or starting any training.
    prepared = [(variant, nv, prepare_job(args, variant, nv, sources)) for variant in args.models for nv in versions(args)]
    for variant, nv, (p, config, identity, skip) in prepared:
        if skip:
            print(f"Verified completed run; skipping {variant}/nv{nv}", flush=True)
            continue
        p["config"].parent.mkdir(parents=True, exist_ok=True)
        p["config"].write_text(yaml.safe_dump(config, sort_keys=False))
        command = training_command(args, variant, nv)
        atomic_json(p["manifest"], dict(identity=identity, complete=False, command=command))
        run(command)
        artifacts = (p["result"], p["checkpoint"] / "best_model.pt")
        atomic_json(p["manifest"], dict(identity=identity, complete=True, command=command,
                                       artifacts={str(a): digest(a) for a in artifacts}))
    collect(args)


if __name__ == "__main__":
    main()
