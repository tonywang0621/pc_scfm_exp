# v16：讓 gate 看見 flow 的實際修正

v16 從對應 NV、seed 的完整 v7 checkpoint 開始。第一階段、原始 gate、投影、budget、兩條修正路徑與原 v7 損失都保留。主要模型只訓練新增的小型 gate adapter；不是重新從零訓練 v7。

## 完整實驗指令

在已能執行 v15 的 Python 環境中，進入專案根目錄：

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/run_v16_ablation.sh --suite all
```

這會執行 5 組 × NV1/NV2 × seed 3407，共 10 個工作，每個工作 10 epochs。會從 `runs/mecge_table1_repro/` 的原 v7 目錄載入相符 checkpoint，資料預設為 `data/mecge_table1_repro/raw/dataset_bw_nv{nv}.pkl`。啟動前先檢查所有工作所需的資料與 checkpoint。

```bash
# 先看工作清單與指令，不建立輸出、不訓練
bash scripts/run_v16_ablation.sh --suite all --dry-run

# 預設 core：先跑 3 個核心消融，共 6 個工作
bash scripts/run_v16_ablation.sh

# 核心跑完後補齊完整實驗；已完成且檔案驗證相符的工作會跳過
bash scripts/run_v16_ablation.sh --suite all --resume

# 只跑主要模型
bash scripts/run_v16_ablation.sh --models v16

# 重新彙整既有結果，不需要重新訓練
bash scripts/run_v16_ablation.sh --suite all --collect-only

# 列出全部消融
bash scripts/run_v16_ablation.sh --list
```

可用 `PYTHON=/path/to/python bash scripts/run_v16_ablation.sh ...` 指定 Python。Windows PowerShell 可在專案目錄直接用 `python src/v16_experiments.py --suite all`。

若資料或 checkpoint 不在預設位置：

```bash
bash scripts/run_v16_ablation.sh --suite all \
  --pkl-file '/path/to/dataset_bw_nv{nv}.pkl' \
  --v7-checkpoint '/path/to/v7_nv{nv}_seed{seed}.pt' \
  --rnd-test '/path/to/rnd_test_nv{nv}.npy'
```

多 seed 用 `--seeds 3407 42 2026`，但必須先有每個 NV/seed 的匹配 v7 checkpoint。單一 seed 的結果不等於跨 seed 的穩定提升。

## 模型與必要消融

| 名稱 | Adapter 輸入 | 可訓練部分 | 要回答的問題 |
|---|---|---|---|
| frozen v7 reference | 原 v7 | 無 | 同一 checkpoint 的真實基準，suite 自動產生 |
| `v16_constant_gate`（A） | 三個固定偏移 | 3 個 adapter 參數 | 只需重新校準整體 gate 強度嗎？ |
| `v16_condition_gate`（B） | 原有條件統計；修正特徵欄位填 0 | Adapter | 增加 gate 容量是否已足夠？ |
| `v16`（C，主要模型） | 條件統計＋實際修正資訊 | Adapter | 修正資訊是否比只有條件資訊更有用？ |
| `v16_condition_gate_flow`（D） | 同 B | Adapter＋residual flow | 解凍 flow 後，只有條件資訊的效果？ |
| `v16_flow`（E） | 同 C | Adapter＋residual flow | 加入修正資訊與學習 flow 是否互補？ |

`--suite core` 是 A/B/C；`--suite flow` 是 B/C/D/E，構成「有無修正資訊 × flow 凍結與否」的 2×2；`--suite all` 是全部五組。`--models` 可指定子集，優先於 `--suite`。

B/C 的 MLP 都是 28→24→3，各有 771 個參數，初始化相同；B 將後 16 個修正特徵置零。D/E 也遵循相同對照。原有 12 個特徵是 6 個 condition channel 的 mean-absolute 與 standard deviation；新增 16 個特徵包含兩條 flow 的 raw/projected/limited 統計、budget 比例、兩通道反號一致性誤差，以及 baseline head 與 `noisy − pre_flow` 的差異。特徵不使用 clean target，並 detach，避免 adapter 輸入反向改動 flow。

每個 gate 使用：

```text
offset = 0.5 × tanh(adapter(features))
gate = 原 gate 最大值 × sigmoid(原 gate logit + offset)
```

最後一層的 weight/bias 全部初始化為 0，A 的三個偏移也初始化為 0。初始預測必須與載入的 v7 一致。三個 gate 保留原上限 0.10、0.44、0.35。±0.5 是預先指定的實驗範圍，不是已知最優值。

全部組別都凍結第一階段與原 v7 gate，並維持這些模組為 eval mode。A/B/C 的 flow 也維持 eval；D/E 只解凍 residual flow，使用原訓練模式與 dropout。D/E 因此測的是整個 flow 微調程序，包含其原訓練隨機性。

原 v7 loss（包含 path consistency）不改權重、不加 cosine/top-k/relative loss。Adapter-only 時，CFM 與凍結第一階段的部分損失仍列入總值，但不會更新 adapter；不要把這些常數的變動解讀為 gate 學習效果。應結合 loss 分項、梯度與 validation 指標判讀。

AdamW、batch 64、10 epochs、無 scheduler、無 early stopping；adapter LR=`1e-4`，可訓練 flow LR=`1e-5`。可用下列白名單參數建立新的 validation 調參實驗；變更設定後請用新的 run root：

```bash
bash scripts/run_v16_ablation.sh --suite core \
  --set training.lr_adapter=0.00005 \
  --set model.v16_gate_logit_bound=0.25 \
  --run-root runs/v16_lr5e5_bound025
```

## 選模與輸出

結果位於 `runs/v16_performance/`。每個 NV/seed 使用同一份資料切分與 v7 初始化，並核對來源、設定、資料與 checkpoint hash。shuffle 使用獨立、按 epoch 設定的 generator，因此 flow 的 dropout 不會改變各組的批次順序。恢復訓練同時恢復 optimizer 與 RNG，從最後完整 epoch 接續；已完成工作也會先核對輸出檔案 hash。

每個 epoch 評估完整 validation：SSD、MAD、PRD 必須不高於 frozen v7，CosSim 必須不低於 frozen v7，且至少一項嚴格改善，才可成為候選。候選依 CosSim、MAD、較早 epoch 排序。test 不參與選 epoch 或 oracle 選擇。

為維持對照，這次沿用 v7 的 window 隨機 70/30 切分（random_state=1），並非 patient-wise 切分。v16 的 gate 設計本身沒有解決 validation 代表性問題；validation 改善仍可能無法轉移到 test。若後續有可靠 record/patient 標籤，應另建分組 holdout 並重新建立匹配基準，不能拿不同切分直接宣稱提升。

若沒有可行 checkpoint，記錄 `no_feasible_checkpoint`，保留最後權重、loss history 與 validation 診斷，但不產生假冒成功的 `best_model.pt` 或主要 test PKL。frozen v7 的 test 預測另存並在不同消融間核對一致性。

`analysis/` 會產生 NV1/NV2 與合併指標、robustness、`validation_ablation_summary.csv`、`validation_diagnostics_summary.csv`、`per_nv_comparison_summary.csv` 和 `ablation_contrasts.csv`。配對差異表分開列出最後 validation、選中 validation 與 test；缺少合格模型時標記 unavailable，不以 v7 補值。SSD/MAD/PRD 的負差值、CosSim 的正差值代表對照中的改善。

Acceptance 延續既有報表格式：`basic_status`／`loaded_v7_status` 描述四指標比較；`cosine_target_status` 另外描述合併 CosSim≥0.945 的既有延伸目標。`target_achieved` 需要基本比較與延伸目標都通過。0.945 不參與 checkpoint 選擇，也不套用在單一 NV。

每個工作的 `checkpoint/validation/` 包含逐 epoch、全部 validation windows 的指標及診斷。waveform NPZ 則在 epoch 0、選中與最後 epoch 保存，固定取 validation 前至多 1024 窗並記錄原始索引，以控制儲存量；不根據結果挑窗。可追溯 direct decoder → pre-flow → final，以及 raw → projected → limited → gated 的兩條修正。`pre_flow` 是同一 checkpoint 的中間結果，不能當成另外重訓的 no-flow 模型。

診斷包括：

- 每窗修正是否改善 SSD/MAD/PRD/CosSim，以及 gate、offset、budget scale。
- 設 `e=clean−pre_flow`、`c=final−pre_flow`，檢查 `ΔSSD=||c||²−2〈c,e〉`、修正大小與方向對齊。
- validation 專用的 scalar shrink oracle，與原 gate logit ±bound 內有限網格的最佳修正，估計 gate 是否還有改善空間。這些使用 clean target，不能作為可部署的預測結果；有限網格也不是連續最佳化的精確上界。
- 固定實際 gate 的 budget bypass 反事實診斷，分別測試 clean、baseline、兩者同時取消限制。budget 經常飽和本身不能證明 budget 太緊。
- loss、梯度、學習率、凍結參數完整性、選中 epoch 與來源資訊。

moving-average 投影不是正交投影，因此不能把兩路能量直接視為互相獨立；診斷使用實際波形計算，保留交叉項。沒有可靠 validation 病人或 noise alpha 標籤時，不猜測分組；test robustness 只在有匹配 amplitude 檔時產生。

## 如何解讀沒有超過 v7 的結果

- A 與 C 相近：收益可能主要來自整體 gate 強度校準，新增修正資訊的必要性較弱。
- C 穩定優於 B：支持「實際修正資訊有幫助」，仍需不同 seed 與 NV 驗證。
- scalar shrink 有空間、bounded gate grid 幾乎沒有：目前 gate 可調範圍或三個 gate 的表示能力可能限制收益；不能直接推論需增加模型規模。
- bounded grid 有空間、C 沒學到：檢查特徵、adapter 梯度、loss 與評估指標的一致性。
- E 優於 C，且 D/E 的對照支持改善：flow 的可訓練性可能重要；同時檢查是否只在某個 NV 改善。
- bypass 傷害結果：budget 可能在保護模型；若 bypass 有益，仍須用正式 validation 實驗確認新的 budget，不能直接以 oracle 結果宣稱超越。

這組實驗驗證的是一個受控假設，不能保證超越 v7。正式結果需在原資料、原 checkpoint 與預定 GPU 環境完成後判斷。

## 程式驗證

```bash
python -B -m unittest discover -s tests -p 'test_v16*.py'
```

測試包括零偏移與 v7 的輸出／loss／RNG 一致性、參數凍結與梯度、B/C 輸入遮罩與相同容量、oracle 幾何、續訓、選模、未合格結果與報表來源核對。CPU 的初始化測試要求逐值相同；runner 沿用 `rtol=1e-5, atol=1e-6` 的數值容差，並保存實際最大差異。程式測試不代表原資料上的效能驗證。

本次本機已通過 53 項 v16 測試與 43 項 v15 回歸測試；另用合成資料、完整模型尺寸完成五組 × NV1/NV2 的一個 epoch 試跑，涵蓋合格與不合格 checkpoint、波形輸出、robustness 與消融報表。這些資料僅用於程式驗證，不是正式 ECG 實驗結果；GPU 尚未實測。
