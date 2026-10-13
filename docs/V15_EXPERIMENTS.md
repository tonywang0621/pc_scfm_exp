# V15：從 v7 初始化的受控微調與八個實驗条件

在專案根目錄、原本具備 PyTorch 的訓練環境執行：

```bash
bash scripts/run_v15_ablation.sh
```

預設 **8 個條件 × NV1/NV2 × seed3407，共 16 次微調**。每次 10 epochs、batch size 64，
使用 CUDA 0。v7 已訓練的 30 epochs 加上這次微調是額外訓練成本，不宣稱與原 v7 相同總成本。
本程式不保證實驗能超越 v7，或 CosSim 達到 0.945。

## 模型與初始化

唯一的初始化基準為：

```text
lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience
```

自動尋找相同 NV、seed 的既有權重：

```text
runs/mecge_table1_repro/<上述完整模型名稱>/checkpoint/
  <上述完整模型名稱>__qtdb_train_qtdb_test__nv1__seed3407/
    lstm_dualpath_dapp_cfm_unet_bd_ecg/best_model.pt
```

資料預設為 `data/mecge_table1_repro/raw/dataset_bw_nv1.pkl` 與 `dataset_bw_nv2.pkl`。
全部所需資料與權重會在第一個實驗開始前檢查；缺檔會顯示完整路徑，不會改用隨機權重。
Checkpoint 接受完整 tensor state_dict，或 `state_dict`／`model_state_dict` 包裝的模型權重檔。
strict loading 只允許新增的零輸出 local gate 權重；不能漏載 backbone、decoder 或 flow。

所有條件保留 v7 的時頻 LSTM、ResConv、兩處 DAPP、dual heads、雙通道 CFM、原有全域 gates、
投影、budget、bridge、單步零起點推論。每次先在完整 validation 核對初始化輸出與原始 v7 相同，
容許張量數值誤差 `rtol=1e-5, atol=1e-6`，並保存實際最大誤差。保護指標使用獨立原 v7 的預測計算。

主模型使用分層學習率：backbone `1e-6`，decoder/dual heads `3e-6`，CFM/gates `1e-5`。
AdamW、betas `[0.8, 0.99]`、weight decay 0、gradient clip 0.5，固定 LR，無 scheduler、無 early stopping。
這些是第一輪預先設定的候選，不是由 test 最佳化的參數。

## 八個條件

所有新增損失從微調第 1 epoch 生效，不沿用 v13/v14 最後 10 epochs 的 ramp。
原 v7 loss 與其所有原有權重保持不變。

| 代號／名稱 | raw cosine | top-8 | extra MSE | 額外 local gate | 更新範圍 |
|---|---:|---:|---:|---|---|
| A `v15_v7_continue` | 0 | 0 | 0 | 無 | 全模型分層微調 |
| B `v15_cos_only` | 0.01 | 0 | 0 | 無 | 同 A |
| C `v15_topk_only` | 0 | 0.01 | 0 | 無 | 同 A |
| D `v15` | 0.01 | 0.01 | 0 | 無 | 同 A |
| E `v15_extra_mse` | 0.01 | 0.01 | 0.25 | 無 | 同 A |
| F `v15_local_gate` | 0.01 | 0.01 | 0 | 有 | 同 A，含 local gate |
| G `v15_extra_mse_local_gate` | 0.01 | 0.01 | 0.25 | 有 | 同 F |
| H `v15_flow_only_finetune` | 0.01 | 0.01 | 0 | 無 | 只更新 CFM／原 gates |

A/B/C/D 是 cosine × top-k 的 2×2 對照；D/E/F/G 是 extra MSE × local gate 的 2×2 對照；
D/H 比較是否需要更新第一階段。F/G 使用 v12 原有的有界、平滑、零輸出初始化 local gate。
H 保留已訓練第一階段，固定其權重、關閉其 dropout 並阻斷梯度；不等同歷史以 noisy 直接作起點的 `flow_only`。

```text
L_v15 = L_v7 + w_cos × mean(1 - raw_cosine)
               + w_topk × mean(top8(abs(prediction-clean)))
               + w_mse × MSE_original_amplitude
```

Cosine/MSE 在原始振幅域，top-k 沿用 v12/v13 模型域定義；預設 `norm=false` 即原始振幅。
MAD 是每窗最大絕對誤差，不是 MAE，top-8 是訓練 surrogate 而非 MAD 本身。
不加入 relative loss 或 teacher penalty；不改成多 step、多 shot 或新的 attention。
E 的損失配方接近歷史 v13_no_local_gate，但初始化、排程及選模不同，不宣稱重現歷史 CSV。

## 驗證與選模

沿用官方 PKL train 部分的 70/30 分割、`random_state=1`。訓練保留原 drop_last，
validation 包含最後不足一批的視窗；test batch size 沿用 50。
模型建構及驗證的 RNG 與訓練隔離，所有條件使用一致的訓練 seed 與流程。

在同一完整 validation 上重新量測 frozen v7，候選必須：

```text
SSD <= v7，MAD <= v7，PRD <= v7，CosSim >= v7
而且至少一項嚴格改善
```

指標判斷不四捨五入、不放寬容差。合格候選以 CosSim 最高、MAD 最低、epoch 最早依序排序。
初始化 epoch 0 永遠不算新模型。test 與歷史 v7 CSV 不参与選模或訓練。

如果沒有合格候選，實驗仍正常完成，`validation_selection.json` 標為 `no_feasible_checkpoint`。
不產生 `best_model.pt` 或主 v15 test PKL，不把 baseline 或最後一個 epoch 冒充合格模型。
驗證合格也不保證 test 四項合格。

## 輸出

```text
runs/v15_performance/
  v15/nv1/seed3407/
    resolved_config.yaml
    manifest.json
    checkpoint/
      baseline.pt
      baseline_test.pkl
      best_model.pt                     # 僅有合格候選時
      model_last.pt
      training_state.pt
      initialization_parity.json
      warm_start_provenance.json
      validation_selection.json
      loss_history.csv
      validation/
        epoch_000_windows.csv
        epoch_000_diagnostics.json
        epoch_001_windows.csv
        epoch_001_diagnostics.json
        ...
    log/
  official_results/                     # 僅各 NV 合格候選
  baseline_v7/
    official_results/                   # 本次載入的 v7 重新推論
    provenance/
    analysis/
  analysis/
    validation_ablation_summary.csv     # 含未找到合格模型的條件
    table1_comparison__official_nv1_nv2.csv
    robustness_comparison__official_nv1_nv2.csv
    table1_comparison__qtdb_train_qtdb_test__nv1.csv
    table1_comparison__qtdb_train_qtdb_test__nv2.csv
    per_window/
    v15_acceptance.json
    v15_acceptance_summary.csv
    report_manifest.json
```

`loss_history.csv` 包含每 epoch 原 v7 loss、各新增 loss 的原始值和加權值、三組 LR、第一／第二階段指標。
`epoch_*_windows.csv` 保存逐窗的兩階段指標、差值及 flow 修正大小。
診斷 JSON 包含 gates 的平均／極值／時間變化、原始／投影／限制後修正量、budget 觸發比例。
這裡的 budget 比例是觸及逐窗幅度預算的視窗比例，不是截斷時間點比例。
`validation_ablation_summary.csv` 匯總每個條件最後 epoch、選中 epoch 與 frozen-v7 指標，
讓沒有合格模型的實驗也保留可分析的結果；最後 epoch 不參與正式 test 主結果。

官方 PRD 使用 prediction-dependent 分母和整個 scope 的 clean 總體均值；
NV1+NV2 先串接訊號再計算，不能平均兩份 PRD。robustness 切分已計算的整體指標，沿用官方開區間規則。
缺 noise-amplitude 陣列仍能產生整體指標，robustness 標為資料不足；不推定 validation 的噪聲標籤。

驗收同時列出與使用者歷史 v7 基準、與本次載入 v7 的配對比較，兩者不混用。
歷史 snapshot 只有 seed3407；其他 seed 需提供自己的歷史基準 CSV，否則歷史验收不完整。
`baseline_reproduction` 另外檢查載入基準是否重現歷史數字；容差只用於重現診斷，不放寬模型驗收。
基本 test 驗收沿用 SSD/PRD 不退步、MAD 嚴格下降、CosSim 嚴格上升；另報告 combined CosSim ≥ 0.945。
`target_achieved` 要同時符合歷史與本次載入的基準及 0.945；缺任一所需 NV／seed、未合格或輸入不合法都不算通過。

所有來源、設定、初始化、資料、checkpoint、診斷及成果以雜湊記錄；共用 v7 baseline 必須跨條件相同。
不同資料或權重不能靜默重用已完成成果。只把 `report_manifest.json` 列出的報表視為本次重新產生的證據。

## 常用命令

```bash
# 列出條件／預覽 16 次微調，不需權重或資料且不寫結果。
bash scripts/run_v15_ablation.sh --list
bash scripts/run_v15_ablation.sh --dry-run

# 只跑主模型，NV1/NV2。
bash scripts/run_v15_ablation.sh --models v15

# 中斷後恢復；已完成的匹配實驗（含未合格）會跳過。
bash scripts/run_v15_ablation.sh --resume

# 只重新彙整，或由已選定 checkpoint 重做推論。
bash scripts/run_v15_ablation.sh --collect-only
bash scripts/run_v15_ablation.sh --eval-only

# v7 位於其他位置：此 root 是完整 v7 模型資料夾，內含 checkpoint/。
bash scripts/run_v15_ablation.sh --v7-run-root /path/to/v7_model_directory

# 自訂權重與資料模板，兩個 NV 必須明確配對。
bash scripts/run_v15_ablation.sh \
  --v7-checkpoint '/path/v7/nv{nv}/seed{seed}/best_model.pt' \
  --pkl-file '/path/data/dataset_bw_nv{nv}.pkl' \
  --rnd-test '/path/data/rnd_test_nv{nv}.npy'

# Validation 調參請換新的輸出根目錄。
bash scripts/run_v15_ablation.sh --run-root runs/v15_cos_trial \
  --set model.v13_raw_cos_weight=0.005
```

支援 `PYTHON=/path/to/python` 選擇環境；`--device cpu` 可做 CPU 測試。
共享 weight override 只作用於啟用該項的條件，名稱指定移除的項目仍強制為零。
只允許 loss weights、三組 LR、batch size、epochs 的明確覆寫，避免記錄不會生效的選项。

修改設定、初始化、程式、資料或環境後應換 `--run-root`。`--resume` 恢復完整 optimizer 與 RNG，
保留原 epoch 預算；不是另外增加 10 epochs。GPU 跨環境的完全決定性不在 CPU 測試保證範圍內。

```bash
python -B -m unittest discover -s tests -p 'test_v15*.py'
```

合成資料與隨機 v7 權重只驗證程式；不能用其結果宣稱真實 ECG 去噪效能。

## 本次實作驗證

- 43 項 CPU 測試通過，包含初始化與 v7 一致、原 loss/梯度/RNG 對照、八個變體、
  第一階段凍結、四指標選模、epoch 邊界精確續訓、來源核對及官方報表。
- 8 條件 × NV1/NV2 的 16 次一個 epoch 合成流程全部完成，包含 15 次有合格候選、
  1 次沒有合格候選的程式分支；這些合格僅指合成 validation。
- 產生 16 列完整 validation 摘要；合成測試樣本數與正式基準不符，驗收沒有誤報達成效能目標。
- 原命令加 `--resume` 重跑時，16 個已完成條件全部跳過，沒有重訓。
- 測試環境 Python 3.12.3、PyTorch 2.14.1+cpu、NumPy 2.5.3；未驗證 CUDA 或正式 ECG 訓練表現。
- 203 個既有程式／腳本／測試／文件檔案保持不變，本次只新增 v15 檔案。
