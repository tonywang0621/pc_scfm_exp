# v17：dualhead、單一 flow 速度場與局部 gate 的合作

v17 沿用目前專案的 QTDB／NSTDB baseline-wander 資料、NV1/NV2、512 samples、360 Hz、LSTM/residual-convolution context 與 DAPP 第一階段。從頭訓練 30 epochs，AdamW、batch 64、LR `5e-5`、無 early stopping。Flow matching、第一階段 dualhead、局部 gate、v7_max 的 top-8／0.05 loss 均保留。

## 執行

使用原本能訓練本專案的 Python 環境，在專案根目錄執行：

```bash
# 查看全部模型與消融
bash scripts/run_v17_ablation.sh --list

# 核對全部 32 個工作，不建立輸出、不讀取訓練資料
bash scripts/run_v17_ablation.sh --suite all --dry-run

# 完整實驗：16 組 × NV1/NV2 × seed 3407，各 30 epochs
bash scripts/run_v17_ablation.sh --suite all

# 預設 core：7 組 × NV1/NV2，共 14 個工作
bash scripts/run_v17_ablation.sh

# 只先訓練 v17
bash scripts/run_v17_ablation.sh --models v17

# 接續中斷工作；已完成且 hash 相符的工作跳過
bash scripts/run_v17_ablation.sh --suite all --resume

# 三個訓練 seed：共 96 個工作；不需要預先準備 v7 checkpoint
bash scripts/run_v17_ablation.sh --suite all --seeds 3407 42 2026

# 只重新彙整結果
bash scripts/run_v17_ablation.sh --collect-only
```

可指定 `PYTHON=/path/to/python bash scripts/run_v17_ablation.sh ...`。PowerShell 可直接使用 `python src/v17_experiments.py --suite all`。

資料預設為專案內 `data/mecge_table1_repro/raw/dataset_bw_nv{nv}.pkl`。伺服器可使用既有位置：

```bash
bash scripts/run_v17_ablation.sh --suite all \
  --data-root /work/tonyalpha1/pc_scfm_exp/data/mecge_table1_repro

# 自訂實際檔案路徑；花括號須保留給腳本替換
bash scripts/run_v17_ablation.sh --suite all \
  --pkl-file '/path/to/dataset_bw_nv{nv}.pkl' \
  --rnd-test '/path/to/rnd_test_nv{nv}.npy' \
  --run-root runs/v17_custom
```

`rnd_test` 必須與 test windows 順序相符。未明確指定時，只搜尋 `--data-root/raw/rnd_test_nv{nv}.npy` 或該處的 `rnd_test.npy`，不自動拿其他資料來源的 amplitude 陣列配對。缺少 amplitude 時仍輸出四指標，明確跳過 robustness；也可使用 `--skip-robustness`。

修改 LR、batch 或模型寬度後應使用新的 `--run-root`。`--set` 使用白名單，不能把 `v17_no_flow` 等因素偷偷改回來。資料、程式、環境與設定 hash 一起保存；工作目錄與舊實驗、資料及來源路徑不能重疊。

## 主要模型的實際計算

令 noisy 為 `y`、clean target 為 `x`。第一階段共用特徵，保留直接解碼 `d`、baseline head `b`、residual head `r`：

```text
z_clean = d + 0.22 r
z_baseline = y - b
condition = [y, z_clean, b]
```

不額外把 `y-z_clean`、`z_baseline`、候選差值等線性可推導量塞進同一個輸入卷積，避免重複的輸入權重自由度。

兩條完整修正若都以同一個 `x` 為終點，必有：

```text
delta_clean* = x - z_clean
delta_baseline* = z_baseline - x
delta_baseline* = (z_baseline - z_clean) - delta_clean*
```

因此 v17 的第二階段只有 **一個 state channel、一個 velocity output channel、一套 backbone**，學習 `delta_clean`。不另建可由條件與第一條修正推導的第二個 flow head。

兩個 gate 決定實際融合：

```text
delta = integrate(v_theta(state, condition, t))
z_refined = z_clean + delta
g = flow trust gate
m = baseline blend gate

z_coarse_mix = (1-m) z_clean + m z_baseline
output = g z_refined + (1-g) z_coarse_mix
       = z_clean + g delta + (1-g)m(z_baseline-z_clean)
```

在一般情況下，最終修正只有兩個有效係數；沒有另設第三個重複控制量。`g`、`m` 均在 `(0,1)`，初始約 `0.25`、`0.10`，可退回粗估候選。未沿用 v7 的 clean gate 上限 `0.10` 或 residual budget，以免再次把完整誤差目標與可施加的修正量限制成不同尺度；這項變動屬完整 redesign，需用實驗確認效果。

Gate 看 `[y, z_clean, b, delta]`，共用 7-point、dilation 1/4 的 temporal feature extractor 與一個 logit projection。保留 v12 局部 logit 調節與 33-point smoothing 的概念，改成：

```text
global_logits = temporal_mean(logits)
local = smooth(tanh(logits - global_logits))
local_offset = 0.5 × bound × (local - temporal_mean(local))
gate = sigmoid(global_logits + local_offset)
```

`bound=0.5`，local offset 時間平均為零且絕對值不超過 bound。Global 與 local 不各自學一份常數偏移，兩者共用投影。這是保留局部控制功能後的精簡版本，並非逐行複製 v12 的額外 gate network。

## Flow matching 與 loss

主要模型使用 zero-source linear conditional flow：

```text
target = stop_gradient(x - z_clean)
state_t = t × target
target_velocity = target
```

訓練的 `t` 有 25% 取 0，其他均勻取 `[0,1)`，讓部署起點也接受速度監督。Condition 在 FM loss 中 detach，final reconstruction 的梯度仍可經 flow、gate 回傳第一階段。推論從零開始，用真正左端點的 explicit Euler，預設 1 NFE；四步版本單獨比較。

`v17_bridge` 使用 `state_t = t target + 0.02 sin(pi t) epsilon`，並把 `0.02 pi cos(pi t) epsilon` 正確加入 target velocity。它不沿用舊程式中加了 bridge 卻缺少其導數的目標。

主要 loss 包含：

- 沿用 ECG 的 time、MSE、complex-spectrum、consistency、LF、morphology 項。
- Top-8 absolute error，權重 `0.05`，從第一個 epoch 啟用且只加一次。
- 第一階段 `z_clean` MSE，權重 `0.25`。
- 原 baseline/residual head 監督，權重 `0.30`／`0.008`。
- 單一 FM velocity MSE，權重 `0.17`，取原兩項速度 loss 權重 `0.04+0.13` 的總和作起始設定。
- 未經 gate 衰減的 `z_refined` endpoint MSE，權重 `0.12`，使 flow 本身學習可用的完整修正。

不另加入與 baseline head 監督代數相同的 baseline-candidate MSE。所有權重是待驗證的預設值，不代表已達到最優。

## 實驗組別

| 名稱 | 與主要模型／參考的差異 |
|---|---|
| `v7_max_control` | 舊 v7＋top-k；用本次 runner 重訓 |
| `v12_topk_control` | 舊 v12 local gate＋相同 top-k |
| `v12_topk_compact` | 舊合作方式與 loss 保留，只做等價參數精簡 |
| `v17` | 完整單速度場、雙有效 gate 合作模型 |
| `v17_no_baseline_fusion` | baseline head 只提供 flow 條件；刪除 baseline blend gate |
| `v17_condition_gate` | gate 不看 delta；直接刪除該輸入欄位的權重 |
| `v17_no_local_gate` | gate 僅使用全域 logits |
| `v17_no_topk` | 移除 top-k loss |
| `v17_no_flow` | 移除 flow、FM／endpoint loss 與 flow trust gate，重訓雙粗估候選融合 |
| `v17_no_dual_head` | 移除兩個粗估 head、其監督及相關條件／融合 gate |
| `v17_fixed_gate` | gate 固定為初始值，刪除可訓練 gate network |
| `v17_tcn` | 以 64-channel、6-block dilated residual TCN 取代 U-Net 速度場；保留 FM |
| `v17_half_flow` | U-Net base width 由 32 減半為 16 |
| `v17_four_steps` | 訓練重建與推論都使用 4 NFE |
| `v17_bridge` | 加入導數一致的 stochastic bridge |
| `v17_direct` | 直接預測 residual，不使用 FM；合併固定 time conditioning 並移除零 state 對應權重 |

`core` 是表中前七組；`modules` 是 v17、no-topk、no-flow、no-dual-head、fixed-gate、direct；`backbones` 是 v17、TCN、half-flow、four-steps、bridge；`all` 是全部 16 組。

V17 對 compact control 是整體 redesign 比較，不能歸因於單一修改。No-flow 與 no-dual-head 同時移除模組及其專屬監督；TCN 與 U-Net 也有容量差異。各配對的解讀記錄在 `paired_ablation_contrasts.csv`。

`v17_direct` 使用相同 U-Net 的一個 Euler 評估所對應的直接函數：把 `t=0` 的 time embedding／projection 合併進各 block 的 bias，並刪除恆為零的 state 輸入權重。相同 seed 下，初始推論與主要模型數值相符；它保留 final／endpoint 重建監督，移除 FM 監督與 direct mapping 不需要的參數。這比保留一套永遠不變的 time/state 參數更符合本次精簡要求。它比較的是 FM 訓練與精簡的直接回歸訓練，optimizer 參數化與訓練參數量仍有差異。

## 參數冗餘的檢查範圍

主要模型及消融不建立未啟用的 auxiliary outputs、fallback scalar gates、關閉模組的權重或零填補的輸入欄位。精簡還涵蓋：

- Mask decoder 的相鄰 transpose／pointwise linear convolutions 合併。
- InstanceNorm 前會被抵消的卷積 bias 移除。
- GroupNorm 前的 bias 只保留每組 `C_group-1` 個零均值座標；不誤刪仍有作用的組內差異。
- LSTM 兩組相加的 bias 合併；另一份為固定零 buffer。前面的 LayerNorm affine 併入 LSTM input weights/bias。
- Residual context scalar gain 併入輸出卷積；depthwise→pointwise 的重複 bias 合併。
- Time embedding 最後一層線性映射併入各 block 的 time projection；與 normalization beta 重複的 additive bias 合併。
- Residual skip bias 與末端 normalization beta 合併。

合併保留實數運算下的函數，浮點輸出以容差檢查；它不保證相同的 optimizer 軌跡。兩個原始 control 特意保留歷史參數，作為比較依據。

每個工作訓練前以獨立模型副本進行三次更新，逐參數追蹤是否連接 loss、梯度有限與曾出現非零梯度的座標比例。V17 與 compact control 若存在整組持續無梯度的參數，直接停止；原始 controls 的已知問題則如實記錄。稽核不改動正式模型權重或 RNG。單個座標在少量樣本中未啟動不等於結構冗餘，不能只靠一次零梯度刪除。

這些檢查消除已知結構冗餘，不能證明所有神經元在所有資料上皆不可替代。Half-width 與 backbone 消融用來檢查容量是否仍可縮減，不會宣稱已證明數學上的全域「零冗餘」。

預設正式尺寸的實測參數數量：

| 模型 | 可訓練參數 |
|---|---:|
| v7_max control | 1,502,792 |
| v12 top-k control | 1,505,339 |
| v12 top-k compact | 1,489,041 |
| v17 U-Net | 1,487,884 |
| v17 half-width U-Net | 729,340 |
| v17 TCN | 639,405 |
| v17 direct regression control | 1,410,892 |

上述七組已在預設模型尺寸、512-sample 合成波形上執行參數稽核。V17 與 compact control 未發現整組持續沒有梯度的參數；有限樣本中仍有少數 PReLU 座標未啟動，保留於稽核紀錄，不宣稱每個座標在這些樣本上都有梯度。

## 選模、續訓與報表

保留 official PKL 的 window-random 70/30 validation split，`random_state=1`，並非 patient-wise split。每個 epoch 檢查完整 validation，包含最後不足 batch 的資料。Primary checkpoint 用 deterministic validation SSD；secondary 用固定 RNG 的完整 composite validation loss。Scheduler 同樣使用 validation SSD，test 不參與選模。

本次所有 control 與 ablation 共用這個 runner；與歷史舊 runner 的選模、validation 隨機流程不同，歷史 CSV 不能當成完全匹配的控制組。模型建立與參數稽核後，fresh training 重新設定相同 seed，避免不同初始化建構流程消耗的隨機數影響起始訓練 RNG。Train shuffle 使用獨立的 `seed+epoch` generator，不會因不同 flow 消耗 RNG 而改變批次順序。Checkpoint 保存 optimizer、scheduler、RNG 與完整 epoch 狀態；恢復時沿用原 epoch 預算及儲存的 RNG，不重設訓練 seed。

主要輸出在 `runs/v17_performance/`：

- `official_results/`：primary SSD checkpoint 的官方格式 `[X_test,y_test,y_pred]`。
- `validation_loss/official_results/`：secondary loss checkpoint 的結果，與 primary 分開彙整。
- 每組 `checkpoint/`：best、best-loss、last、training state、loss history、selection、parameter audit、CPU/GPU 實測 latency profile。
- `validation_selected_waveforms.npz`：選中 epoch 的第一個 validation batch，至多 16 窗，包含 coarse、flow、gate、final。沒有依誤差挑窗。
- `analysis/`：NV1、NV2、串接後的 official 四指標與 noise-strength bins、配對消融差值、跨 seed 平均／標準差、階段診斷、參數與 latency 比較。

Combined PRD 在串接 NV1/NV2 原始預測後重新計算，不直接平均兩個 NV 的 PRD。單 seed 報表標示為 descriptive；不把相關的 ECG windows 當成獨立受試者做顯著性宣稱。配對缺少某組時標為 `missing_pair`，不補成零。

Gate 診斷含平均、時間標準差與接近 0／1 的比例，以及 flow 前後 SSD 改善比例。`coarse_fusion` 使用該次實際 gate，是同模型的診斷；正式 no-flow 結論請看獨立重訓的 `v17_no_flow`。

## 驗證

```bash
python -B -m unittest discover -s tests -p 'test_v17*.py'
```

測試涵蓋所有 16 組的前向／梯度、參數精簡等價性、direct control 初始推論等價性、bridge 導數、Euler 時間／狀態一致性、local gate 零均值、模組移除、top-k 只計一次、完整 validation 與 RNG 隔離、逐值相同的續訓，以及完整 16 組 × NV1/NV2 合成資料流程。

正式 ECG PKL 不在本次本機工作區；本機驗證使用 PyTorch `2.2.2+cpu` 的獨立測試環境。合成資料測試只證明程式流程可執行；原資料的 30-epoch 效果與 GPU 效能需要在實驗機執行上述腳本後評估。
