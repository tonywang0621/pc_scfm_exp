"""Use the unchanged run_mecge_table1_repro.sh reporting programs."""

import subprocess
import sys
from pathlib import Path

import yaml

from mecge_table1_collect_official_protocol import parse_name


APP = Path(__file__).resolve().parent


def run_report(script, *arguments):
    subprocess.run([sys.executable, "-B", str(APP / script), *map(str, arguments)], cwd=APP, check=True)


def collect_results(root, amplitude_files=None):
    root = Path(root).resolve()
    amplitude_files = amplitude_files or {}
    results = []
    for path in sorted((root / "official_results").glob("*.pkl")):
        parsed = parse_name(path)
        if parsed and parsed["noise_version"] in ("nv1", "nv2"):
            results.append((path, parsed))
    if not results:
        raise FileNotFoundError(f"No result PKLs found under {root / 'official_results'}")

    versions = set()
    for path, parsed in results:
        model, noise, seed = parsed["model"], parsed["noise_version"], parsed["seed"]
        versions.add(noise)
        # Preserve existing checkpoint/config paths; only reports use the legacy layout.
        config_path = root / model.removeprefix("lstm_") / noise / seed / "resolved_config.yaml"
        model_name = model
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                model_name = yaml.safe_load(handle)["model_name"]
        eval_dir = root / model / "results" / path.stem / model_name / "best_loss"
        run_report("mecge_table1_official_result_metrics.py", "--result-pkl", path, "--output-dir", eval_dir)
        amplitude = amplitude_files.get(int(noise[2:]))
        if amplitude is not None:
            run_report(
                "mecge_table1_robustness_bins.py",
                "--metrics-per-window", eval_dir / "metrics_per_window.csv",
                "--rnd-test", amplitude, "--output-root", root / model / "controlled_tests",
                "--result-model", model, "--noise-version", noise, "--seed", seed.removeprefix("seed"),
            )
        else:
            print(f"Skipping robustness bins for {path.stem}: no amplitude array supplied/found.", flush=True)

    for noise in [*sorted(versions), "all"]:
        run_report(
            "mecge_table1_collect_results.py", "--run-root", root,
            "--noise-version", noise, "--output-dir", root / "analysis",
        )
    if all(amplitude_files.get(nv) is not None for nv in (1, 2)):
        run_report(
            "mecge_table1_collect_official_protocol.py",
            "--official-results-dir", root / "official_results",
            "--rnd-test-nv1", amplitude_files[1], "--rnd-test-nv2", amplitude_files[2],
            "--output-dir", root / "analysis",
        )
    else:
        print("Skipping official nv1+nv2 reports: the existing collector requires amplitude arrays for both noise versions.", flush=True)
