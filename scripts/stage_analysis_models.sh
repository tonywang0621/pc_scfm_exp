#!/usr/bin/env bash
# Register only models with a verified stage-extraction analyzer.
# key|experiment model|checkpoint model directory|config relative to project|analyzer relative to project
STAGE_ANALYSIS_MODELS=(
  "v7|lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience|lstm_dualpath_dapp_cfm_unet_bd_ecg|src/configs/mecge_table1_repro_lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience.yaml|src/analyze_v7_stages.py"
)
