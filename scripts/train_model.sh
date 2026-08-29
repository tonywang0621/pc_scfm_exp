#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"

MODEL="${1:-pc_scfm}"
shift || true

if [[ "$MODEL" == configs/*.yaml || "$MODEL" == *.yaml ]]; then
  CONFIG="$MODEL"
else
case "$MODEL" in
  pc_scfm)
    CONFIG="configs/ecg_baseline_wander_pc_scfm.yaml"
    ;;
  pc_scfm_rl_no_flow|pc_scfm_no_flow)
    CONFIG="configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml"
    ;;
  pc_scfm_rl_no_flow_no_attention|pc_scfm_no_flow_no_attention)
    CONFIG="configs/ecg_baseline_wander_pc_scfm_rl_no_flow_no_attention.yaml"
    ;;
  mecg_e)
    CONFIG="configs/ecg_baseline_wander_mecg_e.yaml"
    ;;
  mambattention|mambattention_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention.yaml"
    ;;
  mambattention_stfrft|mambattention_stfrft_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft.yaml"
    ;;
  mambattention_stfrft_bag|mambattention_stfrft_bag_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_bag.yaml"
    ;;
  mambattention_stfrft_lf_morph|mambattention_stfrft_lf_morph_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_lf_morph.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp|mambattention_stfrft_dualpath_dapp_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_h4|mambattention_stfrft_dualpath_dapp_h4_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h4.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_h16|mambattention_stfrft_dualpath_dapp_h16_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h16.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_h32|mambattention_stfrft_dualpath_dapp_h32_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_h32.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_v2|mambattention_stfrft_dualpath_dapp_v2_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_v2.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_cfm_residual|mambattention_stfrft_dualpath_dapp_cfm_residual_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_cfm_residual.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_cfm_unet_bd|mambattention_stfrft_dualpath_dapp_cfm_unet_bd_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_dualpath_dapp_cfm_unet_bd.yaml"
    ;;
  mambattention_dualpath_dapp_cfm_unet_bd|mambattention_dualpath_dapp_cfm_unet_bd_ecg)
    CONFIG="configs/mecge_table1_repro_mambattention_dualpath_dapp_cfm_unet_bd.yaml"
    ;;
  mambattention_stfrft_dualpath_dapp_stable_cfm_unet|mambattention_stfrft_dualpath_dapp_stable_cfm_unet_ecg)
    CONFIG="configs/mecge_table1_repro_mambattention_stfrft_dualpath_dapp_stable_cfm_unet.yaml"
    ;;
  mambattention_stfrft_eddm_distill|mambattention_stfrft_eddm_distill_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention_stfrft_eddm_distill.yaml"
    ;;
  eddm)
    CONFIG="configs/ecg_baseline_wander_eddm.yaml"
    ;;
  eddm_flow_matching|eddm_fm)
    CONFIG="configs/mecge_table1_repro_eddm_flow_matching.yaml"
    ;;
  eddm_flow_matching_mamba|eddm_fm_mamba)
    CONFIG="configs/mecge_table1_repro_eddm_flow_matching_mamba.yaml"
    ;;
  fcn_dae|fcndae)
    CONFIG="configs/ecg_baseline_wander_fcn_dae.yaml"
    ;;
  deepfilter|deep_filter)
    CONFIG="configs/ecg_baseline_wander_deepfilter.yaml"
    ;;
  descod_ecg_1shot|descod_1shot|descod1)
    CONFIG="configs/ecg_baseline_wander_descod_ecg_1shot.yaml"
    ;;
  descod_ecg_5shot|descod_5shot|descod5)
    CONFIG="configs/ecg_baseline_wander_descod_ecg_5shot.yaml"
    ;;
  descod_ecg_10shot|descod_10shot|descod10)
    CONFIG="configs/ecg_baseline_wander_descod_ecg_10shot.yaml"
    ;;
  drnn|drrn)
    CONFIG="configs/ecg_baseline_wander_drnn.yaml"
    ;;
  fir_filter|fir)
    CONFIG="configs/ecg_baseline_wander_fir_filter.yaml"
    ;;
  iir_filter|iir)
    CONFIG="configs/ecg_baseline_wander_iir_filter.yaml"
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    echo "Expected one of: pc_scfm, pc_scfm_rl_no_flow, pc_scfm_rl_no_flow_no_attention, mecg_e, mambattention, mambattention_stfrft, mambattention_stfrft_bag, mambattention_stfrft_lf_morph, mambattention_stfrft_dualpath_dapp, mambattention_stfrft_dualpath_dapp_h4, mambattention_stfrft_dualpath_dapp_h16, mambattention_stfrft_dualpath_dapp_h32, mambattention_stfrft_dualpath_dapp_v2, mambattention_stfrft_dualpath_dapp_cfm_residual, mambattention_stfrft_dualpath_dapp_cfm_unet_bd, mambattention_dualpath_dapp_cfm_unet_bd, mambattention_stfrft_dualpath_dapp_stable_cfm_unet, mambattention_stfrft_eddm_distill, eddm, eddm_flow_matching, eddm_flow_matching_mamba, fcn_dae, deepfilter, descod_ecg_1shot, descod_ecg_5shot, descod_ecg_10shot, drnn, fir_filter, iir_filter, or a configs/*.yaml path" >&2
    exit 2
    ;;
esac
fi

cd "$APP_DIR"
python3 train_supervised.py --config "$CONFIG" "$@"
