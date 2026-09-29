# V11 Performance Experiments

V11 is an experimental candidate, not a measured improvement over v7. The primary
targets are lower SSD, MAD and official PRD, and higher uncentered CosSim.

## Isolation

All additions use a separate entry point. Existing model implementations,
`models/__init__.py`, configs, dependencies and scripts are unchanged. The new
entry point explicitly imports `models.lstm_v11` before invoking the existing
`mecge_table1_run_official_local_model.py` training loop.

Outputs default to `runs/v11_performance/`. The unchanged `v7` training job is
excluded. Reporting summarizes this run root without reading historical baselines.
The original v7 model, configuration, checkpoints and results remain untouched.
The launcher rejects roots overlapping the legacy Table 1 run directory, source
directories, or data directory. It does not regenerate datasets.

Each job saves `resolved_config.yaml` and `manifest.json`, including the actual
seed, dataset SHA-256, model/runner source hashes, software/device information,
command, and final checkpoint/result hashes. Completed matching jobs are skipped.
Changed data, code, environment or settings require a new `--run-root`.

## Model

- Keep v7's three time/frequency BiLSTM blocks, residual convolution context,
  feature DAPP, magnitude/phase decoders and coarse dual head.
- Use one residual state/output instead of two opposite CFM targets. Retain the
  six v7 conditioning features to avoid changing the input representation too.
- Use a straight interpolation path with the matching `target - start` velocity.
  Inference starts from zero and takes one deterministic step by default.
- Split the same residual into a moving-average component and its complement,
  with bounded, sample-adaptive correction amplitudes. This decomposition is
  complementary, not an orthogonal or ideal frequency decomposition.
- Omit unused U-Net noise auxiliary outputs and the two-path consistency loss.
- Add the existing top-8 absolute-error loss, weight 0.05, to target MAD.
  MAD is the maximum absolute error in each window, not MAE.
- Preserve the remaining v7 reconstruction losses. The single CFM weight 0.17
  equals the sum of the old channel weights 0.04 + 0.13. The curvature weight
  0.08 corresponds to two opposite channels each weighted 0.04.

The default low/high gate initial amplitudes are 0.020/0.017, near v7's effective
initial correction amplitudes after its consistency blend. Their maxima are
0.20/0.10, with correction budgets 0.95/0.08. These are initial experimental
settings, not values selected using test results.

## Ablations

| Variant | Difference | Suite |
| --- | --- | --- |
| `v7_max` | Add only top-k MAD loss to v7 | core |
| `v7_straight` | Remove only the time-dependent bridge noise from v7 | core |
| `v11` | Full candidate | core, modules |
| `v11_no_max` | Remove top-k loss | core |
| `v11_direct` | Remove CFM supervision; keep the same refiner, inference and reconstruction losses | core |
| `v11_no_band_split` | Full-band correction with one shared amplitude; keep gate capacity | core |
| `v11_no_feature_dapp` | Remove feature DAPP | modules |
| `v11_no_unet_dapp` | Remove U-Net skip DAPP | modules |
| `v11_no_resconv` | Remove convolution context; keep both LSTM axes | modules |
| `v11_no_dual_head` | Remove both coarse auxiliary heads and their losses | modules |
| `v11_fixed_gate` | Fixed initial amplitudes instead of adaptive gates | modules |

`--suite core` runs 6 variants. `--suite modules` runs the full model and 5
module ablations. `--suite all` runs all 11 without duplicating v11.
`v7_max` and `v7_straight` remain because they change v7's training objective/path.
The module ablations remove modules after initialization so retained tensors
start identically for a common seed. `v11_direct` intentionally keeps the same
parameterization to test the contribution of CFM supervision.

The v7-to-v11 structural comparison changes the residual parameterization,
gates and associated losses together. The controlled v7 path/loss variants and
v11 deletion variants diagnose contributions; they are not a complete factorial
study or proof that every retained component is necessary.

## Commands

Run from the project root in the existing training environment. The commands use
the existing official PKLs in `data/mecge_table1_repro/raw/`.

```bash
# Inspect variants and the commands without writing or training.
bash scripts/run_v11_ablation.sh --list
bash scripts/run_v11_ablation.sh --suite all --nv all --dry-run

# Start with the full model on both official noise versions.
bash scripts/run_v11_ablation.sh --models v11 --nv all --seeds 3407 --device cuda:0

# Core comparisons; matching completed v11 jobs will be skipped.
bash scripts/run_v11_ablation.sh --suite core --nv all --seeds 3407 --device cuda:0

# Add the module-removal experiments.
bash scripts/run_v11_ablation.sh --suite modules --nv all --seeds 3407 --device cuda:0

# Alternatively, run all 11 variants (22 jobs for one seed).
bash scripts/run_v11_ablation.sh --suite all --nv all --seeds 3407 --device cuda:0

# Replicate the final candidate across three seeds (no original v7 retraining).
bash scripts/run_v11_ablation.sh --models v11 --nv all --seeds 3407 42 2026 --device cuda:0

# Rebuild reports without training.
bash scripts/run_v11_ablation.sh --collect-only
```

Use `PYTHON=/path/to/python` to select an interpreter. Use `--data-root` for a
different existing data location, or a templated dataset path:

```bash
bash scripts/run_v11_ablation.sh --models v11 --nv all \
  --pkl-file '/path/to/dataset_bw_nv{nv}.pkl' \
  --rnd-test '/path/to/rnd_test_nv{nv}.npy' --device cuda:0
```

Both noise versions must already exist. Explicit `--rnd-test` paths must exist;
otherwise the launcher searches the usual data/reference directories. Missing
amplitude arrays skip robustness bins and the official combined report; the
per-noise and `all` Table 1 summaries still run. The original combined collector
requires both noise-amplitude arrays even for its main metric table.

Change training/model settings in a new output root:

```bash
bash scripts/run_v11_ablation.sh --models v11 --nv all --epochs 60 \
  --run-root runs/v11_performance_60epoch --device cuda:0
```

The first round uses 30 epochs, batch size 64, AdamW at 5e-5, no early stopping,
and the same best-validation-loss checkpoint selection as v7. Train/validation
splitting is the official 70/30 split with random state 1. Test inference batch
size is 50. Do not select settings/checkpoints from test results.

`--resume` continues interrupted jobs using the existing official runner. That
runner does not save/restore RNG states, so resumed trajectories are not promised
to match uninterrupted training. Use uninterrupted runs for final comparisons.
`--eval-only` reevaluates existing checkpoints in this isolated run root.

## Results

Reporting invokes the same unchanged programs as `run_mecge_table1_repro.sh`:

1. `mecge_table1_official_result_metrics.py` converts each result PKL to metrics.
2. `mecge_table1_robustness_bins.py` groups per-window metrics by noise strength.
3. `mecge_table1_collect_results.py` writes nv1, nv2 and `all` summary tables.
4. `mecge_table1_collect_official_protocol.py` recomputes combined nv1+nv2 metrics.

Only available noise versions are summarized individually. The `all` table
lists individual noise-version rows; it is not the combined nv1+nv2 statistic.
Combined rows require both result PKLs and both amplitude arrays.

Per-model artifacts use the original report layout:
`<run-root>/<result-model>/results/<experiment>/<model-name>/best_loss/` contains
`metrics_qtdb_pkl_test.yaml`, `metrics_summary.csv` and `metrics_per_window.csv`.
Robustness artifacts are under `<run-root>/<result-model>/controlled_tests/`.
Training/checkpoint paths are unchanged, so existing v11 jobs remain resumable.

The `analysis/` directory contains:

- `table1_comparison__qtdb_train_qtdb_test__nv1.csv`, `__nv2.csv`, and `__all.csv`.
- `robustness_comparison__qtdb__nv1.csv`, `__nv2.csv`, and `__all.csv`.
- `table1_comparison__official_nv1_nv2.csv`, recomputed on concatenated arrays.
- `robustness_comparison__official_nv1_nv2.csv`, when amplitude arrays are available.

No historical baseline option, v7 improvement report or custom seed-summary
report is generated. Files produced by previous script versions are not deleted;
use the filenames above for current reports. `--collect-only` rebuilds reports
from saved prediction PKLs without training or changing checkpoints.

All four metrics use the unchanged official functions. In particular, official
PRD has a prediction-dependent denominator; it is not interchangeable with the
other PRD function in `utils_ecg.py`. Robustness bins retain the original strict
boundaries, which exclude samples exactly on interior edges. The final bin
retains the official `alpha > 1.5` rule. Therefore bin counts need not sum to the
full test count.

## Verification

```bash
python3 -B tests/test_v11.py
```

The tests cover every variant's forward/backward pass, checkpoint compatibility,
deterministic inference, shared initialization under module removal, legacy v7
registration/RNG/output invariance, protected output paths, read-only dry runs,
and synthetic end-to-end training, collection, skip and collision behavior.
Per-noise, all-noise and combined CSVs are checked against the unchanged collectors,
including robustness tables, collection-only behavior and missing amplitude arrays.
No downloaded dataset or Mamba extension is needed for these CPU tests.
