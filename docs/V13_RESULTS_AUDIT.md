# v7、v11、v12 結果稽核與 v13 驗收

本次驗收依使用者確認，只比較 NV1＋NV2 合併結果；v7 僅作既有結果基準，不包含於本次訓練腳本。

本次讀取既有 CSV 後，**沒有任何 v11、v12 本體或其消融版本，同時符合「SSD、MAD、PRD、CosSim 不退步，且 MAD、CosSim 嚴格改善」**。最接近的是 `lstm_v12_delayed_topk`：合併 NV1+NV2 的 MAD、PRD、CosSim 改善，但 SSD 仍增加 1.242%。這是 v13 設計的線索，不能當作 v13 已達標的結果。

## 資料來源與適用範圍

來源目錄：`C:\Users\JAVA\OneDrive\桌面\icassp_2026_0915`，本機掛載為 `/mnt/c/Users/JAVA/OneDrive/桌面/icassp_2026_0915`。

- `table1_comparison__official_nv1_nv2.csv`
- `robustness_comparison__official_nv1_nv2.csv`
- `v11_comparison__official_nv1_nv2.csv`
- `v11_robustness_comparison__official_nv1_nv2.csv`
- `v12_table1_comparison__official_nv1_nv2.csv`
- `v12_robustness_comparison__official_nv1_nv2.csv`

以下 v7 均指 `lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience`。所有列標示 `seed3407`、`nv1_nv2`，Table 1 每項指標的樣本數均為 26,632。CSV 的標準差描述不同視窗的差異，並非不同訓練 seed 的差異；不能據此宣稱跨 seed 顯著改善。

程式中的指標定義來自 `src/utils_ecg.py` 與 `src/mecge_table1_collect_official_protocol.py`：

- SSD：每個視窗的平方誤差總和，再取視窗平均，越低越好。
- MAD：每個視窗的 **maximum absolute distance**，即最大絕對誤差，再取視窗平均；不是平均絕對誤差，越低越好。
- PRD：沿用官方 `prd_mecge_official`，分母使用預測訊號與整個輸入 clean 陣列的總體平均值。這和一般以 clean 能量為分母的 PRD 不同，必須保持相同定義。
- CosSim：未去均值的每個視窗 cosine similarity，再取視窗平均，越高越好。

NV1+NV2 的官方指標是在串接 clean/prediction 後重算，不能直接平均兩份 per-NV PRD 報表。本次未重新計算或更動任何既有指標。

## 合併 NV1+NV2：本體與全部既有消融

數值為 CSV mean，顯示至小數點後六位；退步判定使用原始精度。

| 模型 | SSD ↓ | MAD ↓ | PRD ↓ | CosSim ↑ | 相對 v7 退步 |
|---|---:|---:|---:|---:|---|
| v7 基準 | 3.284623 | 0.317720 | 35.673649 | 0.939513 | — |
| v11 | 3.470312 | 0.303084 | 35.614642 | 0.936955 | SSD、CosSim |
| v11_direct | 3.576266 | 0.304424 | 35.207232 | 0.935899 | SSD、CosSim |
| v11_fixed_gate | 3.556141 | 0.315556 | 37.266581 | 0.934899 | SSD、PRD、CosSim |
| v11_no_band_split | 3.681818 | 0.301659 | 35.009402 | 0.935036 | SSD、CosSim |
| v11_no_dual_head | 3.562638 | 0.299097 | 35.265209 | 0.936641 | SSD、CosSim |
| v11_no_feature_dapp | 3.534873 | 0.305939 | 34.715093 | 0.936001 | SSD、CosSim |
| v11_no_max | 3.358597 | 0.317585 | 36.048804 | 0.937400 | SSD、PRD、CosSim |
| v11_no_resconv | 3.473184 | 0.310127 | 36.150373 | 0.937564 | SSD、PRD、CosSim |
| v11_no_unet_dapp | 3.438077 | 0.303232 | 35.292766 | 0.936926 | SSD、CosSim |
| v7_max | 3.342262 | 0.302385 | 34.671582 | 0.939288 | SSD、CosSim |
| v7_straight | 3.412744 | 0.320104 | 36.900270 | 0.937464 | 全部 |
| v12 | 3.300911 | 0.314169 | 36.890199 | 0.939192 | SSD、PRD、CosSim |
| v12_delayed_topk | 3.325411 | 0.312666 | 34.885333 | 0.939852 | SSD |
| v12_no_dual_head | 3.620127 | 0.327114 | 38.063809 | 0.933755 | 全部 |
| v12_no_feature_dapp | 3.437416 | 0.321251 | 35.992420 | 0.937547 | 全部 |
| v12_no_resconv | 3.518943 | 0.318794 | 36.431102 | 0.935559 | 全部 |
| v12_no_smoothing | 3.417608 | 0.329292 | 38.414433 | 0.938059 | 全部 |
| v12_no_unet_dapp | 3.564528 | 0.324814 | 37.578747 | 0.934081 | 全部 |
| v7_delayed_topk | 3.369596 | 0.313181 | 36.390008 | 0.938334 | SSD、PRD、CosSim |

`v12_delayed_topk` 相對 v7 的完整精度差異：

| 指標 | v7 | v12_delayed_topk | 候選減基準 |
|---|---:|---:|---:|
| SSD | 3.2846233757376697 | 3.325410855 | +0.0407874792623303（+1.242%） |
| MAD | 0.3177198895497192 | 0.312665926 | -0.0050539635497192（-1.591%） |
| PRD | 35.6736494840964 | 34.88533255 | -0.7883169340964（-2.210%） |
| CosSim | 0.93951292502518 | 0.939851751 | +0.00033882597482 |

既有 v7 消融也支持保留粗估 backbone 與 flow 搭配：`no_flow` 的四項均退步；`flow_only` 的 SSD 高達 5457.953154，顯示此既有實驗失敗，不能將其解讀成「單獨 flow 只略有劣勢」。`no_dual_noise_head` 雖然 MAD 小幅改善，但 SSD、PRD、CosSim 退步。既有 3/5/10-shot、antithetic mean、median、step4、step8 均比指定 v7 的四項平均更差。

## 強度區間

以下每列為 SSD / MAD / PRD / CosSim，沿用官方區間名稱與計算方式。

| 模型 | alpha | SSD ↓ | MAD ↓ | PRD ↓ | CosSim ↑ |
|---|---|---:|---:|---:|---:|
| v7 | 0.2–0.6 | 1.603430 | 0.218387 | 25.364371 | 0.970943 |
| v7 | 0.6–1.0 | 2.441384 | 0.269580 | 31.256494 | 0.954759 |
| v7 | 1.0–1.5 | 3.571079 | 0.342214 | 37.974874 | 0.934022 |
| v7 | 1.5–2.0 | 5.029924 | 0.412076 | 45.243231 | 0.907150 |
| v11 | 0.2–0.6 | 1.722109 | 0.202630 | 25.889398 | 0.967907 |
| v11 | 0.6–1.0 | 2.563907 | 0.254508 | 31.485215 | 0.952483 |
| v11 | 1.0–1.5 | 3.780819 | 0.327311 | 37.836349 | 0.931390 |
| v11 | 1.5–2.0 | 5.293517 | 0.398973 | 44.582140 | 0.904881 |
| v12 | 0.2–0.6 | 1.689065 | 0.216087 | 26.698169 | 0.969849 |
| v12 | 0.6–1.0 | 2.533765 | 0.268222 | 32.679285 | 0.953378 |
| v12 | 1.0–1.5 | 3.575551 | 0.338154 | 39.099940 | 0.934066 |
| v12 | 1.5–2.0 | 4.936649 | 0.406416 | 46.312763 | 0.907935 |
| v12_delayed_topk | 0.2–0.6 | 1.647907 | 0.213632 | 25.180296 | 0.970566 |
| v12_delayed_topk | 0.6–1.0 | 2.475416 | 0.265179 | 30.742012 | 0.955036 |
| v12_delayed_topk | 1.0–1.5 | 3.598284 | 0.335753 | 36.954674 | 0.934671 |
| v12_delayed_topk | 1.5–2.0 | 5.088422 | 0.407867 | 43.973424 | 0.907820 |

`v12_delayed_topk` 的 MAD、PRD 在四個區間均改善，但 SSD 在四個區間均退步。CosSim 僅在最低強度區間退步（-0.000377118），其餘三區改善。因此僅看合併平均會漏掉低強度 CosSim 的問題。所有 v11 本體／消融的區間 CosSim 均低於 v7；同份表中的 `v7_max` 僅在最高強度區間略優。

四個區間樣本數分別為 5,716、5,718、7,358、7,230，加總 26,022，比 Table 1 少 610。程式前三區使用 `alpha > low` 且 `alpha < high`，最後一區僅使用 `alpha > 1.5`；因此邊界值會被排除，最後一區也沒有顯式上界檢查。這是現有官方相容流程的行為，不能偷偷修改後直接和舊表比較。缺少 amplitude 陣列，無法逐筆確認被排除 610 筆的實際值。

## 對 v13 的設計證據

1. 保留 v7 雙路 backbone、dual head、feature DAPP、U-Net DAPP、ResConv 與原始 bridge。v12 移除 dual head、U-Net DAPP 或 gate smoothing 時，四項相對 v12 本體均退步；feature DAPP、ResConv 的消融則犧牲 SSD、MAD、CosSim，雖然 PRD 有改善。
2. 低權重、延後加入的 top-k 誤差比強權重 top-k 更接近目標。`v7_max` 的 MAD 改善約 4.827%，但 SSD 增加約 1.755%，CosSim 降低 0.000225311。不能只增加 MAD loss 就宣稱滿足多指標約束。
3. 平滑局部 gate 與延後 top-k 有搭配線索：`v12_delayed_topk` 比 `v7_delayed_topk` 四項皆佳，差值依序為 SSD -0.044185558、MAD -0.000514701、PRD -1.50467574、CosSim +0.001517779。
4. 尚需解決 SSD 回退與低強度 CosSim 回退；優先採取保留 v7 能力的受限修正，並在 validation 決定 checkpoint／超參數，再一次報告 test 結果。現有摘要不足以證明任何新機制必然改善。

## 可比性與缺漏

CSV 的 protocol、seed、樣本數相符，原始碼中的基本設定也同為 30 epochs、batch size 64、AdamW、learning rate 5e-5、固定 train/validation 切分 random_state=1。v11 使用既有官方 runner；v12 除原始 composite validation loss checkpoint，也另外輸出 deterministic validation SSD checkpoint。

然而，提供的 v11/v12 CSV 沒有 checkpoint selector 欄位；本機沒有對應的 resolved config、manifest、loss history、prediction PKL、checkpoint 或每個 NV 的結果，無法確認該 CSV 是哪個 selector，或確認實際訓練期間有無額外 override。`runs/mecge_table1_repro/analysis` 現有兩份 NV1 CSV 僅有標頭，無法補足資料。也不能由目前原始碼推定歷史報表一定由相同版本產生。

需要保留或取得的證據包括：每個 seed／NV 的資料與程式 hash、已選定 checkpoint 及選擇規則、resolved config、完整預測與 clean/noisy 對齊資料、四個區間的 amplitude 陣列，以及 NV1／NV2／合併三種 scope 的報表。僅有摘要 CSV 無法做逐視窗配對檢定；多 seed 的穩定改善也尚未驗證。

## 自動驗收工具

`src/v13_acceptance.py` 僅用 Python 標準函式庫，讀取指定模型結果並產生可稽核 JSON，不改動來源 CSV，也不依 test 選模型或 checkpoint。API 為 `assess(candidate_csvs, baseline_csvs, *, candidate_model, baseline_model, scope="combined", candidate_robustness_csvs=(), baseline_robustness_csvs=(), require_robustness=False)`。

本次預設 `--scope combined` 只驗收 NV1+NV2 合併表，不要求逐 NV 或 robustness 區間通過。工具保留明確指定 `--scope all`、`--scope per-nv` 與 `--require-robustness` 的功能，供日後擴充分析使用。每個 supplied seed 都必須有配對結果且各自通過；不得用其他 seed 的改善掩蓋失敗 seed。MAD 必須嚴格降低，CosSim 必須嚴格提高，SSD／PRD 可持平。開啟 `--require-robustness` 後，每個 scope／seed 的四個官方區間也必須通過同一規則。

```bash
python3 -B src/v13_acceptance.py \
  --candidate-csv candidate_combined.csv \
  --baseline-csv baseline_combined.csv \
  --candidate-model lstm_v13 \
  --baseline-model lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience \
  --scope combined \
  --output-json v13_acceptance.json
```

| 狀態 | Exit code | 意義 |
|---|---:|---|
| passed | 0 | 所要求的每個配對都符合指標規則 |
| failed | 1 | 資料齊全且可配對，但至少一項指標未達標 |
| incomplete | 2 | 實驗未產生結果或缺少要求的 scope／區間／配對，尚不能驗收 |
| invalid | 3 | 重複列、seed 不匹配、protocol／樣本數不一致、非有限值或輸入格式錯誤 |

`incomplete` 報告仍會保留已可比較部分的失敗紀錄。JSON 記錄輸入 CSV 的 SHA-256、行號、逐項差值、要求的 scope 與所有問題。這是對實際報告平均值的描述性驗收，不保證未知資料或每一個視窗都較好，也不能單靠 CSV 自動證實訓練與資料 provenance。

初版驗證：13 項 unittest 通過，涵蓋嚴格／非嚴格比較、缺少 scope、duplicate、nonfinite、count／protocol／seed 問題、robustness 遺漏與退步、多 seed 不可挑選，以及來源檔保護。以真實 `v12_delayed_topk` CSV 執行 combined + robustness，Table 1 與四區共 5 個配對皆未通過，exit code 為 1；明確指定 all scope 則回報缺少 NV1／NV2、exit code 為 2，並保留合併 SSD 失敗證據。
