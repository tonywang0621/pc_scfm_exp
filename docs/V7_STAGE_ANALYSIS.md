# V7：LSTM 與 flow matching 兩階段波形分析

共用入口：`scripts/analyze_model_stages.sh`；v7 分析器：`src/analyze_v7_stages.py`。
第一版僅支援：

```text
lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience
```

使用既有 checkpoint 做推論，不需要重新訓練，也不修改模型程式或 checkpoint。

## 執行

在專案根目錄可直接執行：

```bash
bash scripts/analyze_model_stages.sh
```

Shell 腳本沿用訓練腳本的目錄規則，預設選取 v7、NV1、seed3407 的
`runs/mecge_table1_repro/<模型>/checkpoint/<實驗名稱>/<model_name>/best_model.pt`，
以及 `data/mecge_table1_repro/raw/dataset_bw_nv1.pkl`，輸出第 0、10、100 筆。
自動選擇 CUDA，無 CUDA 時使用 CPU。NV2 只需加 `--nv 2`。
權重及資料必須已存在；缺檔時顯示預期位置並停止。
若訓練使用其他根目錄，可沿用 `RUN_ROOT`、`DATA_ROOT` 環境變數。
下方完整參數僅在需要覆寫預設值時使用；Python 入口仍需明確指定輸入。


在專案根目錄，用原本具備 PyTorch、OmegaConf、NumPy、Matplotlib 的訓練環境執行。
這些套件已列於專案 requirements；繪圖採 Agg，不需要桌面或顯示器。

```bash
bash scripts/analyze_model_stages.sh \
  --checkpoint /path/to/v7/nv1/best_model.pt \
  --dataset-pkl data/mecge_table1_repro/raw/dataset_bw_nv1.pkl \
  --indices 0 10 100 \
  --nv 1 --seed 3407 \
  --device cuda:0
```

Windows PowerShell：

```powershell
cd C:\Users\中研院\rl_exp
python src/analyze_v7_stages.py `
  --checkpoint "C:\path\to\v7\nv1\best_model.pt" `
  --dataset-pkl "data\mecge_table1_repro\raw\dataset_bw_nv1.pkl" `
  --indices 0 10 100 `
  --nv 1 --seed 3407 `
  --device cuda:0
```

請將 checkpoint 範例路徑換成實際檔案。`--device` 預設 `cpu`，有 GPU 才指定 `cuda:0`。
NV2 分析請同時換成 NV2 的 checkpoint、dataset 與 `--nv 2`。

`--config` 預設為上述 v7 的 YAML。若訓練時修改過模型參數，請指定該次實驗保存的 resolved config：

```bash
bash scripts/analyze_model_stages.sh \
  --config /path/to/resolved_config.yaml \
  --checkpoint /path/to/best_model.pt \
  --dataset-pkl /path/to/dataset_bw_nv1.pkl \
  --indices 5 8 --output-dir runs/stage_waveform_analysis/my_v7_example
```

Config 的 `model_name` 必須是 `lstm_dualpath_dapp_cfm_unet_bd_ecg`，
`exp_name` 必須是指定 v7 名稱或其 `__...` 實驗名稱。其他版本／消融、multi-shot、
多步 flow 或隨機推論起點會明確拒絕，不會套用錯誤的階段定義。
Checkpoint 支援完整 tensor state_dict，或包在 `model_state_dict` / `state_dict` 的 tensor checkpoint。
使用 strict loading；缺少任何 flow 權重都會報錯。請使用模型權重檔，非含 optimizer/RNG 的完整 training_state。

資料使用現有官方 PKL 格式 `[X_train, y_train, X_test, y_test]`，
或含 `X_test`、`y_test` 的字典。接受 `[N,T]`、`[N,1,T]`、`[N,T,1]` 單導程資料。
結果 PKL `[noisy, clean, prediction]` 不是這個腳本的輸入。
資料字典可省略 `y_test` 或設成 None；此時仍輸出兩階段波形與修正量，指標欄位留空。
與原訓練流程相同，只讀取可信任的本地實驗 PKL。

## 兩階段的精確定義

```text
noisy ECG
  → 完整時頻 LSTM backbone、DAPP、magnitude/phase 解碼
  → decoder_direct
  → 加上 dual head 的 residual_refine_scale × residual_delta_hat
  → lstm_pre_flow                 第一階段
  → 原模型 _refine_from_base：雙通道 CFM、投影、幅度預算與 gate 混合
  → flow_final                    第二階段
```

第一階段不是第一個 LSTM block 的 feature map，也不是單獨 decoder 的輸出。
腳本先依原模型正規化 noisy，再呼叫原有的 `_restore_components()` 與 `_refine_from_base()`；
兩階段及額外保存的 `decoder_direct` 都還原成原始輸入振幅單位。
每批都再呼叫 `model.denoising()` 作一致性檢查，第二階段須符合 `rtol=1e-5, atol=1e-6`。
這會增加一次正常推論的成本；不會改權重。

```text
flow_correction = flow_final - lstm_pre_flow
```

這是套用所有限制與 gate 後，真正加到 ECG 的淨修正量；不是原始雙通道 flow velocity。

## 輸出

預設每次建立新的 UTC 時間戳資料夾：

```text
runs/stage_waveform_analysis/
  v7_nv1_seed3407_best_model_<UTC>/
    manifest.json
    stage_metrics.csv
    sample_000000/
      waveform_comparison.png
      flow_correction.png
      waveforms.npz
    sample_000010/
      ...
```

- `waveform_comparison.png`：三列，依序為 noisy、LSTM pre-flow、flow 後結果；
  有 clean 時每列疊加參考波形，三列使用相同振幅刻度。
- `flow_correction.png`：上方為實際修正量，下方比較兩階段的逐點絕對誤差；
  無 clean 時，下方改為兩階段 ECG 疊圖。
- `waveforms.npz`：包含 `noisy`、`lstm_pre_flow`、`flow_final`、`flow_correction`、
  `decoder_direct`、`sample_index`、`sample_position`；有 clean 時加上 `clean`，
  指定取樣率時加上 `time_seconds`。波形均為一維完整視窗，沒有降採樣或截圖範圍裁切。
- `stage_metrics.csv`：每筆 noisy、第一／第二階段的 SSD、MAD、PRD、CosSim，
  以及每項 `delta`、`improved` 和 flow 修正量的 mean/max absolute value。
- `manifest.json`：記錄輸入檔案／程式／輸出雜湊、resolved config、樣本編號、
  PRD 均值基準、環境與推論一致性誤差。`complete=true` 才表示整次分析成功。

重複的 `--indices` 會去重並保留指定順序。`--output-dir` 必須是尚不存在的新資料夾，
避免覆蓋先前分析。未提供 `--indices` 時只分析測試集第 0 筆。

## 如何讀指標與座標

`delta = 第二階段 − 第一階段`。SSD、MAD、PRD 的 delta 負值表示改善，
CosSim 的 delta 正值表示改善；相等不標為嚴格改善。MAD 是最大絕對誤差，不是 MAE。

PRD 保持專案官方的 prediction-dependent 分母：

```text
100 × sqrt(sum((prediction-clean)^2) /
           (sum((prediction-mean(full_dataset_y_test))^2) + 1e-10))
```

完整 y_test 的均值只計算一次，不會因 `--indices` 或 batch size 改變。
這是單一 dataset 的測試範圍；不是 NV1+NV2 合併 Table 1，不能拿少數視窗的分數直接代表整體模型表現。

預設橫軸為視窗內的樣本點。確認資料的真實取樣率後，可加上 `--sampling-rate <Hz>` 改用秒；
不會重新取樣，也不直接把 YAML 的宣告當作 PKL 已驗證的取樣率。振幅標為 input units，不擅自宣稱 mV。

`--seed` 是你提供的訓練 seed 標籤，不控制推論，也不能用來證明 checkpoint 的訓練來源。
原始 v7 模板標示 seed42，因此腳本不拿模板 seed 推定實際 checkpoint seed；未指定時目錄使用 `seedunknown`。
`--nv` 未指定時只從資料檔名推定，無法辨識則為 `nvunknown`。資料與 checkpoint 的配對仍需依原實驗確認。

## 驗證

```bash
python -B -m unittest discover -s tests -p 'test_v7_stage_analysis.py'
```

測試涵蓋實際 v7 的階段輸出、normalization 還原、與正常推論相等、權重不變、strict checkpoint、
官方 PRD 的完整測試集均值、不同陣列布局、PNG/NPZ/CSV/manifest 匯出及錯誤狀態。
合成資料及隨機測試 checkpoint 只用來驗證程式；不能當成已訓練 v7 的去噪結果。

## 共用模型入口與擴充

```bash
bash scripts/analyze_model_stages.sh --list-models
bash scripts/analyze_model_stages.sh --model v7
bash scripts/analyze_model_stages.sh --model v7 --nv 2
```

目前只登記 v7；尚未支援的名稱會在推論前報錯。`--model` 也接受完整模型名稱。
舊 `analyze_v7_stages.sh` 保留為 v7 相容入口。

模型登記集中於 `scripts/stage_analysis_models.sh`，每列依序為：
短名稱、完整實驗名稱、checkpoint 模型目錄、設定檔相對路徑、分析器相對路徑。
新增模型時先實作並驗證分析器，再新增一列；共用入口即可套用相同的資料、權重尋址及參數流程。
分析器需接受現有 CLI 參數、在預設輸出資料夾使用自身模型名稱，並核對第二階段與原始推論一致。
v7 分析器保留其嚴格的 v7 驗證，其他架構不能直接套用而略過檢查。
