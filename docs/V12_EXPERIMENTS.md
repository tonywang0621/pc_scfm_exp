# V12 experiments

V12 preserves the v7 BiLSTM / residual convolution / DAPP / dual-head /
two-channel CFM U-Net model and adds bounded, smooth local adjustments to its
three global gates. It is an experimental candidate; no QTDB performance gain
has been measured yet.

The original v7 is **not a training variant** in this suite. Existing v7/v11
implementations, launchers, checkpoints and reports are not modified. V12 is
registered only by importing `models.lstm_v12` from its separate entry point.
The model lives in `src/models/lstm_v12/__init__.py`, so it does not change the
legacy v11 launcher's nonrecursive model-source inventory or invalidate its
existing manifest hashes.

## Run everything

From the project root, in the existing training environment:

```bash
bash scripts/run_v12_ablation.sh
```

This runs all **8 variants x 2 noise versions x seed 3407 = 16 training jobs**
sequentially on `cuda:0`, then evaluates and aggregates the results. Defaults:
30 epochs, batch size 64, AdamW, learning rate 5e-5, no early stopping, one-step
inference. No original v7 or v11 job is launched.

Use `PYTHON=/path/to/python` to choose the interpreter. The interpreter must
contain the project's normal training dependencies. No Mamba extension is
required for this BiLSTM model.

```bash
# List variants and preview commands without creating output files.
bash scripts/run_v12_ablation.sh --list
bash scripts/run_v12_ablation.sh --dry-run

# The 3 core comparisons only: 6 jobs for one seed.
bash scripts/run_v12_ablation.sh --suite core --device cuda:0

# All models and ablations on 3 seeds: 48 jobs.
bash scripts/run_v12_ablation.sh --seeds 3407 42 2026 --device cuda:0

# Explicit existing data location; data are never regenerated.
bash scripts/run_v12_ablation.sh --data-root /path/to/mecge_table1_repro

# Or explicit per-noise-version dataset and amplitude paths.
bash scripts/run_v12_ablation.sh \
  --pkl-file '/path/to/dataset_bw_nv{nv}.pkl' \
  --rnd-test '/path/to/rnd_test_nv{nv}.npy'

# Resume interrupted jobs; completed matching jobs are skipped.
bash scripts/run_v12_ablation.sh --resume

# Rebuild CSVs from existing predictions without training.
bash scripts/run_v12_ablation.sh --collect-only
```

The default data location is `data/mecge_table1_repro/raw/`, containing
`dataset_bw_nv1.pkl` and `dataset_bw_nv2.pkl`. The runner keeps the official
70/30 train/validation window split with random state 1 and test batch size 50.
It consumes the existing PKLs directly; it does not resample them. Their true
sampling rate depends on their preparation, not merely the YAML declaration.

Missing datasets cause an error before any training. Missing amplitude arrays
skip robustness bins; combined nv1+nv2 core metrics still work when both result
PKLs exist. `--skip-robustness` disables amplitude-dependent reporting explicitly.
On recollection, unavailable robustness reports are not regenerated. Existing
files are retained with an explicit warning. Each selector's
`analysis/report_manifest.json` lists `regenerated` and
`retained_not_regenerated` paths; the latter are historical artifacts and may
be stale. If only one amplitude array is available, the new `all` robustness
table contains only that noise version.

## Variants

| Variant | Change |
| --- | --- |
| `v12` | Local temporal gates with original v7 losses |
| `v7_delayed_topk` | Loss-only control: v7 architecture with delayed small top-k; not original v7 |
| `v12_delayed_topk` | V12 plus the same delayed small top-k |
| `v12_no_smoothing` | Remove local gate smoothing, retain bounded offsets |
| `v12_no_feature_dapp` | Remove feature-domain DAPP |
| `v12_no_unet_dapp` | Remove U-Net skip DAPP |
| `v12_no_resconv` | Remove residual convolution context; retain both BiLSTM axes |
| `v12_no_dual_head` | Remove the coarse dual head and its associated losses |

`--suite core` runs the first three. `--suite modules` runs v12 plus the five
module ablations. `--suite all` is the default and runs each variant once.
`--models v12 v12_delayed_topk` selects explicit variants. Noise version is
selectable with `--nv 1`, `--nv 2`, or the default `--nv all`.

The original v7 is excluded as requested, including a configuration override
that would disable the local gate and loss experiment while retaining every
original module. Existing historical v7 results can be compared separately;
this script does not use them to select checkpoints or tune settings.

## Model and loss

For each clean, baseline and blend gate:

```text
gate(t) = gate_max * sigmoid(global_logit + local_offset(t))
local_offset = 0.5 * moving_average(tanh(local_network(condition)))
```

The local network uses the existing six conditioning channels, hidden width 16,
kernel size 7, dilations 1 and 4, and three outputs. The moving average has a
33-sample kernel. The output layer starts at zero, so initial local offsets are
zero. Its initialization does not consume the legacy model's RNG stream.
The gate bounds, CFM bridge, two residual states, one-step integration,
projection, correction budgets and existing reconstruction losses stay in use.
All module-removal variants preserve initialization of retained parameters.

The top-k experiment uses 8 largest absolute errors per window. Its training
weight is zero for epochs 1–20, then 0.001 at epoch 21 through 0.01 at epoch 30.
For another total epoch count it ramps over the last `min(10, total_epochs)`
epochs. Validation uses the **fixed terminal weight** in all epochs, keeping its
loss definition constant for checkpoint selection and the LR scheduler.
`v12` and module ablations have zero top-k weight.

The gate settings and 0.01 tail weight are initial experimental settings, not
values selected from the test set. Test metrics must not be used to choose
epochs, seeds, gate parameters or loss weights.

## Checkpoints and outputs

All output defaults to `runs/v12_performance/`.

```text
v12_performance/
  <variant>/nv1/seed3407/
    resolved_config.yaml
    manifest.json
    checkpoint/
      best_model.pt
      best_val_ssd.pt
      model_last.pt
      training_state.pt
      loss_history.csv
    log/
  official_results/                # original validation-loss selector
  analysis/                       # its aggregate CSVs
  lstm_<variant>/results/.../      # its per-window metrics
  validation_ssd/
    official_results/             # deterministic validation-SSD selector
    analysis/                     # separate aggregate CSVs
    lstm_<variant>/results/.../    # separate per-window metrics
```

The existing composite validation loss still drives `best_model.pt` and the LR
scheduler. It includes the stochastic CFM term, as in the legacy protocol.
An additional deterministic denoising pass covers **all validation windows**,
including a final partial batch, and saves `best_val_ssd.pt`. This extra pass
preserves Python, NumPy and Torch RNG state so it does not perturb training.
The two checkpoint selections are evaluated and reported separately; improving
the selector alone is not evidence of an architectural improvement.

Each selector has:

- `table1_comparison__qtdb_train_qtdb_test__nv1.csv`, `__nv2.csv`, `__all.csv`.
- `table1_comparison__official_nv1_nv2.csv`, computed on concatenated arrays.
- Per-window SSD, MAD, official PRD and CosSim.
- Robustness CSVs when the corresponding amplitude arrays are available.

The `__all.csv` contains separate noise-version rows; the `__official_nv1_nv2.csv`
contains combined metrics and requires both noise versions for each model/seed.
MAD means maximum absolute distance per window. Official PRD is the existing
prediction-dependent definition. The legacy strict robustness-bin boundaries
are retained, so bin counts need not sum to the entire test set.

Manifests record resolved settings, data/source hashes, environment, command,
and both checkpoint/result hashes. Matching finished jobs are skipped. A
different setting, source, data or environment requires a new `--run-root`.
For example:

```bash
bash scripts/run_v12_ablation.sh --models v12 --epochs 60 \
  --run-root runs/v12_60epoch
```

New training states include optimizer, scheduler, epoch, histories and RNG
states. `--resume` continues an interrupted job with the original total-epoch
schedule. Changing its total epochs changes the experiment identity; use a new
run root for such experiments. CUDA determinism still depends on the runtime
and kernels. The launcher protects the legacy Table 1 and v11 output roots.

## Verification

```bash
python3 -B -m unittest discover -s tests -p 'test_v12*.py'
```

Tests exercise initialization/RNG preservation, finite gradients, gate bounds,
tail scheduling, checkpoints/resume, validation isolation, dry runs, collision
protection and synthetic end-to-end reporting. Synthetic checks establish that
the experiment pipeline works; they do not measure QTDB denoising performance.
