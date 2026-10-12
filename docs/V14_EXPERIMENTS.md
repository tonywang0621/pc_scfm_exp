# V14：保留 v7 推論結構，平衡相對誤差、CosSim 與 MAD

V14 是可執行的實驗候選，尚未完成真實 QTDB 訓練，不能宣稱已優於 v7 或達到 CosSim 0.945。
這版改進的是訓練目標與驗證選模流程，沒有新增推論模組、參數或推論步數。

## 一鍵執行

在 `rl_exp` 根目錄、原本具備 PyTorch 的訓練環境執行：

```bash
bash scripts/run_v14_ablation.sh
```

Windows PowerShell 可使用同一個 Python 入口：

```powershell
cd C:\Users\中研院\rl_exp
python src/v14_experiments.py
```

預設依序跑 **7 個變體 × NV1/NV2 × seed3407 = 14 次訓練**。
每次 30 epochs、batch size 64、AdamW、lr 5e-5、無 early stopping、單步零起點推論，
沿用官方 PKL 的 train/validation 70/30 切分（random_state=1）及 test batch size 50。
不重新生成資料，不啟動原始 v7 或 v11～v13 的訓練。

```bash
# 列出變體；預覽完整命令，不寫檔、不訓練。
bash scripts/run_v14_ablation.sh --list
bash scripts/run_v14_ablation.sh --dry-run

# 只跑完整 v14，共兩次訓練。
bash scripts/run_v14_ablation.sh --models v14

# 中斷續訓；已有完整、hash 相符的實驗會跳過。
bash scripts/run_v14_ablation.sh --resume

# 只重新彙整，或從既有兩種 checkpoint 重新推論。
bash scripts/run_v14_ablation.sh --collect-only
bash scripts/run_v14_ablation.sh --models v14 --eval-only

# 指定現有資料與裝置。
bash scripts/run_v14_ablation.sh --data-root /path/to/mecge_table1_repro --device cuda:0
bash scripts/run_v14_ablation.sh \
  --pkl-file '/path/dataset_bw_nv{nv}.pkl' \
  --rnd-test '/path/rnd_test_nv{nv}.npy'
```

預設資料為 `data/mecge_table1_repro/raw/dataset_bw_nv1.pkl` 與 `dataset_bw_nv2.pkl`。
缺任一所需 PKL，會在開始訓練前報錯；缺 amplitude 陣列時仍產生合併指標，robustness 標為缺資料。
`--skip-robustness` 可明確略過區間分析。`PYTHON=/path/to/python` 可為 shell 腳本指定直譯器。

## 為什麼採用這個方向

以下數值來自使用者提供的 `icassp_2026_0915` CSV，均為 NV1+NV2 合併、seed3407、26,632 視窗。
v7 指 `lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience`。

| 模型 | SSD ↓ | MAD ↓ | 官方 PRD ↓ | CosSim ↑ |
|---|---:|---:|---:|---:|
| v7 | 3.284623376 | 0.317719890 | 35.673649484 | 0.939512925 |
| v11 | 3.470311583 | 0.303083742 | 35.614642450 | 0.936954683 |
| v12_delayed_topk | 3.325410855 | 0.312665926 | 34.885332550 | 0.939851751 |
| v13 | 3.389666410 | 0.320010675 | 37.261184510 | 0.938152850 |
| v13_no_local_gate | 3.274731565 | 0.310882625 | 35.706509240 | 0.939651372 |

v13_no_local_gate 改善 SSD、MAD、CosSim，只有 PRD 略退步；它在低強度兩區仍有 SSD/PRD/CosSim 退步。
v12 的 backbone 模組移除實驗則普遍傷害 SSD、CosSim。因此 v14 保留 v7 的時頻 BiLSTM、ResConv、
feature DAPP、U-Net DAPP、dual head、雙通道 CFM、原始 bridge 與全域 adaptive gates，移除 v12 局部 gate。
低噪聲與低 clean 能量不是同一分類；新相對誤差項只是針對視窗間能量差異的假設，不能由區間結果直接推定有效。

v13 的 raw cosine 權重為 0.01。V14 將其提高至 0.05，搭配相對重建項，
同時保留額外絕對 MSE 與小權重 top-k，避免只追求方向而忽略振幅和尖峰誤差。
這些都是預先設定的第一輪假設，尚未經 validation 調參；沒有使用 test 0.945 或 v7 test 數值挑 checkpoint。

## 模型與損失

模型：`src/models/lstm_v14/__init__.py`，註冊名稱 `lstm_relative_balanced_v14_ecg`。
Opt-in import 不改 `models/__init__.py`，也不改任何既有模型的實作或 launcher source inventory。

令 clean 為 y、模型最終輸出為 p，新增項目為：

```text
L_relative = mean_windows[ mean_t((p-y)^2) / (mean_t(y^2) + 0.01) ]
L_raw_cos = mean_windows[1 - uncentered_cosine(p, y)]
L_top8 = mean_windows[8 個最大絕對誤差的平均]

L_v14 = L_v7 + ramp(epoch) * (
    0.25 * MSE_original_amplitude
  + 0.05 * L_raw_cos
  + 0.01 * L_top8
  + 0.02 * L_relative
)
```

原本 `lambda_mse=0.95` 與其餘 v7 loss 不變。30 epochs 的前 20 epochs 新增項目權重為零，
第 21～30 epochs 線性從終值的 0.1 增至 1.0；短實驗則使用最後 `min(10, epochs)` epochs。
驗證 composite loss 始終採終值權重，但僅作診斷，不控制 checkpoint 或 scheduler。
`v14_no_curriculum` 則從第 1 epoch 使用全部終值權重。

MSE、raw cosine、relative loss 都回到原始振幅域計算；top-k 沿用 v13 的模型域定義，
預設 `norm=false` 時即原始振幅。Mask 會排除 padding；全遮罩視窗不納入 relative loss 的平均。
clean 為零的視窗仍受 MSE 與 relative loss 監督，raw cosine 沿用 v13 排除無參考方向視窗的行為。

0.01 是 clean mean-square 的加性下限尺度，使 inverse-energy 權重不超過 100；
它不是從 test 估計的常數，也沒有把誤差本身截斷。
相對重建可增加低能量視窗的權重，但也可能傷害 SSD，因此必須透過 `no_relative` 與 `no_extra_mse` 檢查。

**官方 PRD 不等於此 relative loss。** 專案的官方 PRD 分母是 prediction 相對於完整 clean 陣列總體均值的能量。
V14 不直接最佳化這個預測相依分母，報表仍完整保留原公式。
MAD 是每個視窗的最大絕對誤差，不是 MAE；top-8 是其訓練 surrogate。

## 七個必要對照

| 變體 | 與完整 v14 的差異 | 要回答的問題 |
|---|---|---|
| `v14` | 完整候選 | 是否守住四項並提高 CosSim？ |
| `v14_no_relative` | relative 權重歸零 | 能量正規化是否有額外貢獻？ |
| `v14_no_raw_cos` | raw cosine 權重歸零 | 方向損失是否必要？ |
| `v14_no_topk` | top-k 權重歸零 | MAD 損失是否造成其他指標取捨？ |
| `v14_no_extra_mse` | 只移除額外 0.25 MSE，保留原 v7 MSE | 額外振幅監督是否守住 SSD？ |
| `v14_no_curriculum` | 四項新增權重從 epoch 1 生效 | 延後訓練是否有幫助？ |
| `v14_v13_control` | relative=0、raw cosine=0.01 | 同一新流程下，v13_no_local_gate 的損失目標表現如何？ |

`no_relative` 與 `v13_control` 的預設差異只有 raw cosine 0.05 vs 0.01，
可區分「提高 cosine 權重」和「新增 relative loss」的效果。
最後一個 control 在預設設定下與 v13_no_local_gate 的模型初始化、loss、梯度及 RNG 消耗相同；
但它使用新的選模/scheduler/驗證 RNG 流程，因此不要求重現舊 CSV。
共享 `--set` 調參也會套用於 control，只有列出的消融移除項會在最後強制套用。
不要把改過共享設定的 control 當作完全相同的歷史配方。

歷史 `v13_tail_only` 與 `v12_delayed_topk` 理應使用相同目標，實際 CSV 卻不同。
目前缺少它們的 resolved config、manifest 與 checkpoint，差異原因仍未確定；v14 保留這些證據供後續追查。

## 固定的驗證選模流程

V13 各變體的 composite val_loss 定義不同，且包含隨機 CFM 項。
V14 在完整 validation 上做 deterministic denoising，所有變體共用：

```text
val_score = val_MSE + 0.05 * (1 - val_CosSim) + 0.01 * val_top8
```

主 checkpoint `best_model.pt` 依此 score 的嚴格最小值保存；ReduceLROnPlateau 與 early stopping 也看此 score。
`early_stopping_min_delta` 只影響 patience，不會阻止更佳 checkpoint 保存。
另輸出 `best_val_ssd.pt` 作為獨立對照，兩種 test 結果分開，不自動選 test 較好的版本。
主要驗收固定使用 `best_val_score` 報表。SSD selector 只用於診斷。
這是固定加權的選模準則，不是能保證四個 test 指標都不退步的約束最佳化。

驗證包含最後不足一個 batch 的視窗；官方 PRD 使用整個 validation 的 clean 總體均值。
隨機 composite loss 的診斷及 deterministic metrics 都使用隔離的固定 RNG，結束後還原訓練狀態。
相較 v13，這會改變 scheduler 和訓練 RNG 的歷史軌跡，不能將與歷史 CSV 的全部差異都歸因於新 loss。

`checkpoint/validation_selection.json` 記錄兩個 checkpoint 的實際選定 epoch、score 與四項 validation 指標。
`loss_history.csv` 記錄 train/composite validation loss、四項指標、score、top-8 及各項 train/validation 權重。

## 輸出與驗收

```text
runs/v14_performance/
  <variant>/nv1/seed3407/
    resolved_config.yaml
    manifest.json
    checkpoint/
      best_model.pt
      best_val_ssd.pt
      model_last.pt
      training_state.pt
      loss_history.csv
      validation_selection.json
    log/
  official_results/                    # best_val_score 預測
  analysis/
    table1_comparison__official_nv1_nv2.csv
    robustness_comparison__official_nv1_nv2.csv
    report_manifest.json
    collection_provenance.json
    v14_acceptance.json
    v14_acceptance_summary.csv
  validation_ssd/
    official_results/
    analysis/                          # best_val_ssd，獨立彙整與驗收
```

另保留逐 NV 表、逐視窗誤差及四個官方 robustness 區間。
NV1+NV2 先串接原始 clean/prediction 再計算，不能直接平均兩份 PRD。
官方區間的開放邊界行為不改；分區樣本數可能小於完整測試數。

內附 `src/configs/v14_baseline_v7.csv` 與 `v14_baseline_v7_robustness.csv` 是使用者結果中
指定 v7 的精確資料列，未四捨五入；`v14_baseline_provenance.json` 記錄來源檔及 snapshot SHA-256。
它們只供報表比較，不會輸入訓練、scheduler 或 checkpoint 選擇。

基本驗收為同 seed、同 protocol、同 count 下：

```text
SSD <= v7，MAD < v7，PRD <= v7，CosSim > v7
```

`cosine_target.status` 另外報告 combined CosSim 是否達到 0.945；
`target_achieved=true` 必須同時通過基本驗收及所有配對 seed 的 0.945 門檻。
個別 NV 與 robustness bins 不套用 0.945 門檻；robustness 改善檢查是診斷，不是主要通過條件。
摘要的 basic_status 是模型全部配對 seed 的狀態，不能用一個成功 seed 掩蓋另一個失敗 seed。

```bash
# 獨立驗收（僅標準函式庫）；exit code 0/1/2/3 分別為基本驗收通過/未通過/缺資料/無效。
python src/v14_acceptance.py \
  --candidate-csv runs/v14_performance/analysis/table1_comparison__official_nv1_nv2.csv \
  --output-json runs/v14_performance/analysis/v14_single_acceptance.json

# 多 seed 可以訓練，但每個 seed 驗收都需要它自己的 v7 基準。
bash scripts/run_v14_ablation.sh --models v14 --seeds 3407 42 2026 \
  --baseline-csv /path/to/matched_v7_combined.csv \
  --baseline-robustness-csv /path/to/matched_v7_robustness.csv
```

預設 snapshot 只有 seed3407。它不能代替其他 seed 的基準，也不能證明統計顯著改善。
缺資料、seed/count/protocol 不合、來源非有限值會明確回報，不能算通過。
自動彙整會檢查受 manifest 管理的 prediction/checkpoint hash，並記錄報表來源；
沒有訓練 manifest 的外部 PKL 僅能做來源未驗證的描述性比較。
歷史或未重新產生的報表不作本次驗收證據。

## 改設定與測試

```bash
# 只用 validation 決定調參方向，換新的 run root。
bash scripts/run_v14_ablation.sh --models v14 --run-root runs/v14_relative_trial \
  --set model.v14_relative_weight=0.01

# 測試程式行為，不代表實際模型效能。
python -B -m unittest discover -s tests -p 'test_v14*.py'
```

修改資料、程式、環境或訓練設定後，舊 job manifest 會拒絕混用，請使用新的 `--run-root`。
`--resume` 恢復 optimizer、scheduler、完整 RNG 與原 epoch budget；不是把 v7 checkpoint 當初始化。
不同 selector、不同超參數不得根據 test 分數挑最好結果後當作預先指定的實驗。
模型效能必須在真實資料上測得；本次 CPU 合成資料驗證只確認程式及整條流程可執行。

## 本次實作驗證

- Windows 專案安裝後，62 項 v14 測試全部通過，涵蓋七個變體的 forward/backward、
  v7 初始化與推論一致性、v13 objective 對照、loss 數值、續訓 RNG、selector、報表及驗收。
- 舊版 v11、v12、v13 共 79 項測試通過；181 個既有 source/script/test/doc 檔案的 SHA-256 保持不變。
- 七個完整變體各跑 NV1/NV2、1 epoch 的合成資料流程，共 14 次訓練，全部完成。
  兩種 selector 各產生 7 列合併結果、28 列 robustness、驗收摘要與來源驗證。
- 合成資料僅有 16 個合併測試視窗，與真實 v7 的 26,632 不符；驗收正確標記 `invalid`，
  不把合成測試當作超越 v7 的證據。
- 測試環境為 Python 3.12.3、PyTorch 2.14.1+cpu、NumPy 2.5.3。
  本次未在 CUDA 或 requirements 指定的舊版 PyTorch 上執行驗證，也未執行真實 QTDB 訓練。
