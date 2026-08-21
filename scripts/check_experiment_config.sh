#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python3 - <<'PY'
from pathlib import Path
import sys
import yaml


CONFIG_DIR = Path("src/configs")
CONFIGS = sorted(CONFIG_DIR.glob("ecg_baseline_wander*.yaml"))

EXPECTED = {
    ("data_dir",): "../data/ecg_baseline_wander",
    ("root_dir",): "../runs/ecg_baseline_wander",
    ("checkpoint_dir",): "${root_dir}/checkpoint",
    ("results_dir",): "${root_dir}/results",
    ("log_dir",): "${root_dir}/log",
    ("seed",): 42,
    ("training", "batch_size"): 32,
    ("training", "train_iterations"): 20000,
    ("training", "eval_every"): 100,
    ("training", "validation_metrics_every"): 500,
    ("training", "save_every"): 5000,
    ("training", "lr"): 1.0e-4,
    ("training", "early_stopping_patience"): 25,
    ("training", "early_stopping_min_delta"): 1.0e-4,
    ("training", "resume"): False,
    ("training", "resume_checkpoint"): None,
    ("evaluation", "low_frequency_high_hz"): 0.5,
    ("dataset", "name"): "ecg_baseline_wander",
    ("dataset", "data_dir"): "${data_dir}",
    ("dataset", "train_dataset"): "ptbxl",
    ("dataset", "external_test_datasets"): ["mit_bih", "chapman", "cpsc"],
    ("dataset", "lead"): "II",
    ("dataset", "split", "strategy"): "ptbxl_official_folds",
    ("dataset", "split", "patient_wise"): True,
    ("dataset", "split", "train_folds"): [1, 2, 3, 4, 5, 6, 7, 8],
    ("dataset", "split", "validation_fold"): 9,
    ("dataset", "split", "test_fold"): 10,
    ("dataset", "split", "ratio"): [0.8, 0.1, 0.1],
    ("dataset", "clean_reference", "filter_type"): "butterworth",
    ("dataset", "clean_reference", "filter_order"): 4,
    ("dataset", "clean_reference", "zero_phase"): True,
    ("dataset", "clean_reference", "bandpass_hz"): [0.05, 40.0],
    ("dataset", "resample_hz"): 250,
    ("dataset", "window_size"): 512,
    ("dataset", "overlap_ratio"): 0.5,
    ("dataset", "normalization"): "z_score",
    ("dataset", "baseline_wander", "train_source"): "nstdb",
    ("dataset", "baseline_wander", "alpha_mode"): "peak_to_peak_ratio",
    ("dataset", "baseline_wander", "alpha_values"): [0.05, 0.1, 0.2, 0.3, 0.5],
    ("dataset", "baseline_wander", "controlled_frequencies_hz"): [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    ("dataset", "baseline_wander", "test_types"): [
        "nstdb",
        "sinusoidal",
        "multi_sine",
        "random_low_frequency_drift",
    ],
    ("dataset", "eps"): 1.0e-10,
    ("model", "fea"): "pha",
    ("model", "norm"): False,
    ("model", "compress_factor"): 0.3,
    ("model", "num_tscblocks"): 4,
    ("model", "d_state"): 16,
    ("model", "d_conv"): 4,
    ("model", "expand"): 4,
    ("model", "norm_epsilon"): 1.0e-5,
    ("model", "sampling_rate"): 250,
    ("model", "n_fft"): 64,
    ("model", "hop_size"): 8,
    ("model", "win_size"): 64,
    ("model", "beta"): 2,
}

EXPECTED_BY_MODEL = {
    "mecg_e": {
        ("model", "dense_channel"): 32,
        ("model", "fmamba"): True,
        ("model", "loss_fn"): "time+com+con",
    },
    "mambattention_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con",
    },
    "mambattention_stfrft_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con",
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
    },
    "mambattention_stfrft_bag_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+mse+cos+max+com+lf+morph",
        ("model", "lambda_mse"): 0.6,
        ("model", "lambda_cos"): 0.15,
        ("model", "lambda_max"): 0.08,
        ("model", "mad_topk"): 8,
        ("model", "lambda_lf"): 0.1,
        ("model", "lambda_morph"): 0.02,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "baseline_gate_init"): 1.0,
        ("model", "baseline_gate_min"): 0.2,
        ("model", "baseline_gate_max"): 1.3,
        ("model", "baseline_blend_init"): 0.05,
        ("model", "baseline_blend_max"): 0.6,
        ("model", "baseline_gate_smooth"): True,
    },
    "mambattention_stfrft_lf_morph_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con+lf+morph",
        ("model", "lambda_lf"): 0.1,
        ("model", "lambda_morph"): 0.05,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
    },
    "mambattention_stfrft_eddm_distill_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+mse+com+con+dual_noise+teacher",
        ("model", "lambda_mse"): 0.7,
        ("model", "lambda_dual_baseline"): 0.2,
        ("model", "lambda_dual_residual"): 0.15,
        ("model", "lambda_teacher_l1"): 0.25,
        ("model", "lambda_teacher_mse"): 0.2,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "use_dapp"): True,
        ("model", "dapp_time_scales"): [3, 5, 9, 15],
        ("model", "dapp_freq_scales"): [1, 3],
        ("model", "dual_noise_head_hidden"): 64,
        ("model", "residual_refine_scale"): 0.5,
    },
    "mambattention_stfrft_dualpath_dapp_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+mse+com+con+dual_noise+lf+morph",
        ("model", "lambda_mse"): 0.6,
        ("model", "lambda_dual_baseline"): 0.2,
        ("model", "lambda_dual_residual"): 0.15,
        ("model", "lambda_lf"): 0.03,
        ("model", "lambda_morph"): 0.02,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "use_dapp"): True,
        ("model", "dapp_time_scales"): [3, 5, 9, 15],
        ("model", "dapp_freq_scales"): [1, 3],
        ("model", "dual_noise_head_hidden"): 64,
        ("model", "residual_refine_scale"): 0.5,
        ("model", "baseline_kernel_size"): 129,
    },
    "mambattention_stfrft_dualpath_dapp_v2_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+mse+com+con+dual_noise+lf+morph",
        ("model", "lambda_mse"): 0.5,
        ("model", "lambda_dual_baseline"): 0.25,
        ("model", "lambda_dual_residual"): 0.2,
        ("model", "lambda_lf"): 0.01,
        ("model", "lambda_morph"): 0.05,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "use_dapp"): True,
        ("model", "dapp_time_scales"): [3, 5, 9, 15],
        ("model", "dapp_freq_scales"): [1, 3],
        ("model", "dual_noise_head_hidden"): 64,
        ("model", "residual_refine_scale"): 0.5,
        ("model", "baseline_refine_gate_init"): 0.08,
        ("model", "baseline_refine_gate_max"): 0.5,
        ("model", "baseline_kernel_size"): 129,
    },
    "mambattention_stfrft_dualpath_dapp_cfm_residual_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con+dual_noise+lf+morph+cfm",
        ("model", "lambda_dual_baseline"): 0.2,
        ("model", "lambda_dual_residual"): 0.15,
        ("model", "lambda_lf"): 0.03,
        ("model", "lambda_morph"): 0.02,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "use_dapp"): True,
        ("model", "dapp_time_scales"): [3, 5, 9, 15],
        ("model", "dapp_freq_scales"): [1, 3],
        ("model", "dual_noise_head_hidden"): 64,
        ("model", "residual_refine_scale"): 0.5,
        ("model", "baseline_kernel_size"): 129,
        ("model", "cfm_channels"): 64,
        ("model", "cfm_blocks"): 6,
        ("model", "cfm_time_dim"): 128,
        ("model", "cfm_inference_steps"): 4,
        ("model", "cfm_use_adaptive_gate"): True,
        ("model", "cfm_gate_hidden"): 64,
        ("model", "cfm_clean_delta_budget"): 0.6,
        ("model", "cfm_baseline_delta_budget"): 0.8,
        ("model", "cfm_project_baseline_delta"): True,
        ("model", "cfm_clean_lowpass_keep"): 0.25,
        ("model", "cfm_baseline_gate_init"): 0.15,
        ("model", "cfm_baseline_gate_max"): 0.6,
        ("model", "cfm_consistency_blend_init"): 0.25,
        ("model", "cfm_consistency_blend_max"): 0.5,
        ("model", "lambda_cfm"): 0.2,
        ("model", "lambda_cfm_baseline"): 0.2,
        ("model", "lambda_cfm_recon"): 0.25,
        ("model", "lambda_cfm_stft"): 0.05,
        ("model", "lambda_cfm_lf"): 0.03,
        ("model", "lambda_cfm_deriv"): 0.03,
        ("model", "lambda_cfm_residual_smooth"): 0.005,
        ("model", "lambda_cfm_clean_baseline_consistency"): 0.05,
    },
    "mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+mse+com+con+dual_noise+lf+morph+cfm+noise_aux",
        ("model", "lambda_mse"): 0.45,
        ("model", "lambda_dual_baseline"): 0.30,
        ("model", "lambda_dual_residual"): 0.06,
        ("model", "lambda_lf"): 0.05,
        ("model", "lambda_morph"): 0.08,
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
        ("model", "use_dapp"): True,
        ("model", "dapp_time_scales"): [3, 5, 9, 15],
        ("model", "dapp_freq_scales"): [1, 3],
        ("model", "dual_noise_head_hidden"): 64,
        ("model", "residual_refine_scale"): 0.5,
        ("model", "baseline_kernel_size"): 129,
        ("model", "cfm_channels"): 64,
        ("model", "cfm_blocks"): 6,
        ("model", "cfm_time_dim"): 128,
        ("model", "cfm_inference_steps"): 3,
        ("model", "cfm_use_adaptive_gate"): True,
        ("model", "cfm_gate_hidden"): 64,
        ("model", "cfm_clean_delta_budget"): 0.30,
        ("model", "cfm_baseline_delta_budget"): 1.20,
        ("model", "cfm_project_baseline_delta"): True,
        ("model", "cfm_clean_lowpass_keep"): 0.25,
        ("model", "cfm_baseline_gate_init"): 0.22,
        ("model", "cfm_baseline_gate_max"): 0.85,
        ("model", "cfm_consistency_blend_init"): 0.30,
        ("model", "cfm_consistency_blend_max"): 0.65,
        ("model", "lambda_cfm"): 0.10,
        ("model", "lambda_cfm_baseline"): 0.40,
        ("model", "lambda_cfm_recon"): 0.30,
        ("model", "lambda_cfm_stft"): 0.03,
        ("model", "lambda_cfm_lf"): 0.05,
        ("model", "lambda_cfm_deriv"): 0.07,
        ("model", "lambda_cfm_residual_smooth"): 0.015,
        ("model", "lambda_cfm_clean_baseline_consistency"): 0.10,
        ("model", "cfm_unet_base_channels"): 64,
        ("model", "cfm_unet_channel_mults"): [1, 2, 4],
        ("model", "cfm_unet_time_dim"): 192,
        ("model", "cfm_unet_pool_scales"): [3, 5, 9, 15],
        ("model", "cfm_unet_attention_heads"): 4,
        ("model", "cfm_unet_aux_channels"): 2,
        ("model", "lambda_ecg_noise_aux"): 0.35,
        ("model", "lambda_gaussian_noise_aux"): 0.10,
        ("model", "gaussian_aux_scale"): 0.2,
    },
    "pc_scfm": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "pcscfm_enabled"): True,
    },
    "eddm": {
        ("model", "dense_channel"): 64,
        ("model", "timesteps"): 50,
        ("model", "inference_steps"): 50,
        ("model", "base_channels"): 64,
        ("model", "channel_mults"): [1, 2, 4, 8],
        ("model", "time_dim"): 256,
        ("model", "dropout"): 0.0,
        ("model", "gaussian_scale"): 0.2,
        ("model", "ecg_noise_weight"): 1.0,
        ("model", "gaussian_noise_weight"): 1.0,
        ("model", "pool_scales"): [3, 5, 9, 15],
        ("model", "attention_heads"): 4,
    },
    "fir_filter": {
        ("model", "dense_channel"): 64,
        ("model", "filter_kind"): "fir",
        ("model", "cutoff_hz"): 0.5,
        ("model", "fir_order"): 56,
        ("model", "fir_window"): "kaiser",
        ("model", "kaiser_beta"): 8.6,
        ("model", "zero_phase"): True,
    },
    "iir_filter": {
        ("model", "dense_channel"): 64,
        ("model", "filter_kind"): "iir",
        ("model", "cutoff_hz"): 0.5,
        ("model", "iir_order"): 1,
        ("model", "iir_method"): "butterworth",
        ("model", "zero_phase"): True,
    },
    "drnn": {
        ("model", "dense_channel"): 64,
        ("model", "input_size"): 1,
        ("model", "lstm_hidden_sizes"): [64, 32, 32, 32, 32],
        ("model", "dense_layers"): [],
        ("model", "output_size"): 1,
        ("model", "dropout"): 0.0,
        ("model", "residual"): False,
        ("model", "loss_fn"): "mse",
    },
    "fcn_dae": {
        ("model", "dense_channel"): 64,
        ("model", "input_channels"): 1,
        ("model", "kernel_size"): 16,
        ("model", "encoder_channels"): [40, 20, 20, 20, 40, 1],
        ("model", "encoder_strides"): [2, 2, 2, 2, 2, 1],
        ("model", "decoder_channels"): [1, 40, 20, 20, 20, 40, 1],
        ("model", "decoder_strides"): [1, 2, 2, 2, 2, 2, 1],
        ("model", "loss_fn"): "mse",
    },
    "deepfilter": {
        ("model", "dense_channel"): 64,
        ("model", "input_channels"): 1,
        ("model", "layers"): [64, 64, 32, 32, 16, 16],
        ("model", "dilated_pattern"): [False, True, False, True, False, True],
        ("model", "kernels"): [3, 5, 9, 15],
        ("model", "dilated_kernels"): [3, 5, 9, 15],
        ("model", "dilation"): 3,
        ("model", "dropout"): 0.4,
        ("model", "output_kernel_size"): 9,
        ("model", "loss_fn"): "ssd+mad",
        ("model", "ssd_weight"): 1.0,
        ("model", "mad_weight"): 1.0,
    },
    "descod_ecg_1shot": {
        ("model", "feats"): 80,
        ("model", "num_steps"): 50,
        ("model", "beta_start"): 1.0e-4,
        ("model", "beta_end"): 0.5,
        ("model", "schedule"): "quad",
        ("model", "num_shots"): 1,
        ("model", "clip_denoised"): False,
        ("model", "loss_reduction"): "sum",
        ("model", "loss_fn"): "noise_l1",
    },
    "descod_ecg_5shot": {
        ("model", "feats"): 80,
        ("model", "num_steps"): 50,
        ("model", "beta_start"): 1.0e-4,
        ("model", "beta_end"): 0.5,
        ("model", "schedule"): "quad",
        ("model", "num_shots"): 5,
        ("model", "clip_denoised"): False,
        ("model", "loss_reduction"): "sum",
        ("model", "loss_fn"): "noise_l1",
    },
    "descod_ecg_10shot": {
        ("model", "feats"): 80,
        ("model", "num_steps"): 50,
        ("model", "beta_start"): 1.0e-4,
        ("model", "beta_end"): 0.5,
        ("model", "schedule"): "quad",
        ("model", "num_shots"): 10,
        ("model", "clip_denoised"): False,
        ("model", "loss_reduction"): "sum",
        ("model", "loss_fn"): "noise_l1",
    },
}

EXPECTED_MAMBATTENTION_VARIANTS = {
    "ecg_baseline_wander_mambattention.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_no_time_attention.yaml": ("before_mamba", False, True),
    "ecg_baseline_wander_mambattention_no_freq_attention.yaml": ("before_mamba", True, False),
    "ecg_baseline_wander_mambattention_post_attention.yaml": ("after_mamba", True, True),
    "ecg_baseline_wander_mambattention_post_no_time_attention.yaml": ("after_mamba", False, True),
    "ecg_baseline_wander_mambattention_post_no_freq_attention.yaml": ("after_mamba", True, False),
    "ecg_baseline_wander_mambattention_stfrft.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_bag.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_lf_morph.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h4.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h16.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h32.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_v2.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_cfm_residual.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_eddm_distill.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_no_time_attention.yaml": ("before_mamba", False, True),
    "ecg_baseline_wander_mambattention_stfrft_no_freq_attention.yaml": ("before_mamba", True, False),
    "ecg_baseline_wander_mambattention_stfrft_no_attention.yaml": ("before_mamba", False, False),
}

DUALPATH_DAPP_HEAD_SWEEP_MODELS = {
    "mambattention_stfrft_dualpath_dapp_h4_ecg": 4,
    "mambattention_stfrft_dualpath_dapp_h16_ecg": 16,
    "mambattention_stfrft_dualpath_dapp_h32_ecg": 32,
}

MAMBATTENTION_MODEL_NAMES = {
    "mambattention_ecg",
    "mambattention_stfrft_ecg",
    "mambattention_stfrft_bag_ecg",
    "mambattention_stfrft_lf_morph_ecg",
    "mambattention_stfrft_dualpath_dapp_ecg",
    "mambattention_stfrft_dualpath_dapp_v2_ecg",
    "mambattention_stfrft_dualpath_dapp_cfm_residual_ecg",
    "mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg",
    "mambattention_stfrft_eddm_distill_ecg",
    *DUALPATH_DAPP_HEAD_SWEEP_MODELS.keys(),
}

EXPECTED_PCSCFM_VARIANTS = {
    "ecg_baseline_wander_pc_scfm.yaml": {
        ("model", "phase_representation"): "raw",
        ("model", "attention_position"): "before_mamba",
        ("model", "use_time_attention"): True,
        ("model", "use_freq_attention"): True,
        ("model", "use_flow_proposal"): None,
        ("model", "flow_nfe"): 4,
        ("model", "flow_samples"): 4,
        ("model", "loss_fn"): "time+com+con+lf+morph+flow+bc+value+risk",
        ("model", "lambda_flow"): 0.01,
    },
    "ecg_baseline_wander_pc_scfm_rl_no_flow.yaml": {
        ("model", "phase_representation"): "raw",
        ("model", "attention_position"): "before_mamba",
        ("model", "use_time_attention"): True,
        ("model", "use_freq_attention"): True,
        ("model", "use_flow_proposal"): False,
        ("model", "flow_nfe"): 1,
        ("model", "flow_samples"): 1,
        ("model", "loss_fn"): "time+com+con+lf+morph+bc+value+risk",
        ("model", "lambda_flow"): 0.0,
    },
    "ecg_baseline_wander_pc_scfm_rl_no_flow_no_attention.yaml": {
        ("model", "phase_representation"): "raw",
        ("model", "attention_position"): "before_mamba",
        ("model", "use_time_attention"): False,
        ("model", "use_freq_attention"): False,
        ("model", "use_flow_proposal"): False,
        ("model", "flow_nfe"): 1,
        ("model", "flow_samples"): 1,
        ("model", "loss_fn"): "time+com+con+lf+morph+bc+value+risk",
        ("model", "lambda_flow"): 0.0,
    },
}


def get_value(data, path):
    value = data
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check_value(errors, data, path, expected, config_name):
    actual = get_value(data, path)
    if actual != expected:
        errors.append(
            f"{config_name}: {'.'.join(path)} expected {expected!r}, got {actual!r}"
        )


def check_optional_value(errors, data, path, expected, config_name):
    actual = get_value(data, path)
    if actual not in {None, expected}:
        errors.append(
            f"{config_name}: {'.'.join(path)} expected omitted or {expected!r}, got {actual!r}"
        )


errors = []
if not CONFIGS:
    errors.append("No configs found.")

for path in CONFIGS:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.name
    model_name = data.get("model_name")

    for key_path, expected in EXPECTED.items():
        if model_name in {"descod_ecg_1shot", "descod_ecg_5shot", "descod_ecg_10shot"} and key_path[0] == "model":
            continue
        check_value(errors, data, key_path, expected, name)

    for key_path, expected in EXPECTED_BY_MODEL.get(model_name, {}).items():
        check_value(errors, data, key_path, expected, name)

    if model_name in DUALPATH_DAPP_HEAD_SWEEP_MODELS:
        base_expectations = EXPECTED_BY_MODEL["mambattention_stfrft_dualpath_dapp_ecg"]
        for key_path, expected in base_expectations.items():
            if key_path == ("model", "attention_heads"):
                expected = DUALPATH_DAPP_HEAD_SWEEP_MODELS[model_name]
            check_value(errors, data, key_path, expected, name)

    if model_name in MAMBATTENTION_MODEL_NAMES:
        expected_variant = EXPECTED_MAMBATTENTION_VARIANTS.get(name)
        if expected_variant is None:
            errors.append(f"{name}: unknown MambAttention variant config name.")
        else:
            position, use_time, use_freq = expected_variant
            check_value(errors, data, ("model", "attention_position"), position, name)
            check_value(errors, data, ("model", "use_time_attention"), use_time, name)
            check_value(errors, data, ("model", "use_freq_attention"), use_freq, name)

    if model_name == "pc_scfm":
        expected_variant = EXPECTED_PCSCFM_VARIANTS.get(name)
        if expected_variant is None:
            errors.append(f"{name}: unknown PC-SCFM variant config name.")
        else:
            for key_path, expected in expected_variant.items():
                if key_path == ("model", "use_flow_proposal") and expected is None:
                    check_optional_value(errors, data, key_path, True, name)
                else:
                    check_value(errors, data, key_path, expected, name)

    if model_name in {
        "mambattention_stfrft_dualpath_dapp_cfm_residual_ecg",
        "mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg",
    }:
        loss_tokens = set(str(get_value(data, ("model", "loss_fn"))).split("+"))
        forbidden_tokens = {"teacher", "distill", "prd", "cos", "max"}
        overlap = sorted(loss_tokens & forbidden_tokens)
        if overlap:
            errors.append(
                f"{name}: CFM model must not use evaluation-metric or EDDM teacher loss tokens: {overlap}"
            )
            loss_terms = set(str(get_value(data, ("model", "loss_fn"))).split("+"))
            uses_flow = get_value(data, ("model", "use_flow_proposal"))
            if uses_flow is False and "flow" in loss_terms:
                errors.append(f"{name}: no-flow PC-SCFM must not include 'flow' in model.loss_fn.")

if errors:
    print("Experiment config check failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print(f"Experiment config check OK ({len(CONFIGS)} configs).")
PY
