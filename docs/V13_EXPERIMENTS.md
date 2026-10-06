# V13：針對 MAD、CosSim 與 SSD 取捨的實驗候選

V13 已有程式與消融設計，但尚無真實資料的訓練結果，**不能宣稱已優於 v7**。
比較基準為 `lstm_dualpath_dapp_cfm_unet_bd_no_attention_v7_resconvctx_30epoch_no_patience`。
既有結果分析見 [V13_RESULTS_AUDIT.md](V13_RESULTS_AUDIT.md)。

## 設計依據

目前最接近目標的是 `v12_delayed_topk`：合併 NV1/NV2 的 MAD、PRD、CosSim 均優於 v7，
但 SSD 從 3.284623376 上升至 3.325410855（約 +1.24%），仍不合格。
因此保留 v7 的 BiLSTM、ResConv、兩處 DAPP、dual head、雙通道 CFM 與單步推論，
沿用 v12 的有界平滑局部門控和後期 top-k，僅加入兩項可獨立消融的訓練損失：

- 原始振幅域的額外 MSE，目的是抑制 SSD 退步。
- 未扣除均值的 CosSim loss，直接對應官方 CosSim 的方向。

既有 `+cos` 損失先將訊號去均值，與官方 CosSim 定義不同，v13 不以它代替新損失。
新增損失不增加推論參數或推論步數；v13 的推論結構與 v12 相同。

```text
L_v13 = L_v7 + w_tail(epoch) * L_top8
                 + w_mse(epoch) * MSE_original_amplitude
                 + w_cos(epoch) * (1 - CosSim_uncentered)
```

預設末期權重為 top-k 0.01、額外 MSE 0.25、CosSim 0.01。
30 epochs 時，前 20 epochs 的三項新增訓練損失權重為零，最後 10 epochs 線性增加。
驗證 loss 全程使用固定的末期權重，以保持 checkpoint 選擇與 scheduler 的比較尺度一致。
原本 `lambda_mse=0.95` 等 v7 損失權重不變。

因此新增項目仍可能在前 20 epochs 透過 validation loss 影響學習率與 checkpoint。
消融衡量的是訓練目標加上這套驗證／排程流程的整體效果，不能宣稱只隔離了新增梯度的效果。
另列 `best_val_ssd` 能對照 selector 的差異，但不會排除 scheduler 的差異。

這些權重是第一輪假設，尚未經 validation 調參；額外 MSE 並不構成 SSD 必然改善的保證。
不直接最佳化官方 PRD 的預測相依分母，以免只靠改變輸出尺度改善該數字。
若要調參，應使用 validation，並另行命名 run root。

## 必要消融

| 變體 | 改變 | 要回答的問題 |
| --- | --- | --- |
| `v13` | 完整候選 | 能否同時改善 MAD/CosSim 並維持 SSD/PRD？ |
| `v13_no_extra_mse` | 額外 MSE 權重歸零 | 額外 MSE 是否緩解 SSD 退步？ |
| `v13_no_raw_cos` | 新 CosSim 權重歸零 | 官方定義的形狀損失是否有助益？ |
| `v13_tail_only` | 上述兩項新損失皆歸零 | 對照 `v12_delayed_topk`，辨識兩者交互作用。 |
| `v13_no_topk` | top-k 權重歸零 | 尾端誤差損失是否仍有必要？ |
| `v13_no_local_gate` | 移除 v12 局部門控 | 改善來自局部控制或新增損失？ |

前四個變體構成 MSE/CosSim 的 2×2 對照。這輪不重複所有 v12 的舊模組消融。
本次腳本只包含 v13 與上述五個消融，不提供 v7 或 `v7_reference` 訓練選項，
也不接受 `--suite reference`。v7 僅使用已完成的合併結果作比較，不重新訓練。
若以 `--set` 關閉全部新增機制而使設定退回原始 v7，腳本會在啟動訓練前拒絕。
共享的 `--set` 先套用，再執行各消融的移除設定，避免調權重時意外重新開啟被消融的項目。

## 執行

在 `rl_exp` 專案根目錄、原本具備 PyTorch 的訓練環境執行：

```bash
bash scripts/run_v13_ablation.sh --list
bash scripts/run_v13_ablation.sh --dry-run

# 六個必要變體 × NV1/NV2 × seed3407，共 12 次訓練。
bash scripts/run_v13_ablation.sh

# 先執行本體，共 2 次訓練。
bash scripts/run_v13_ablation.sh --models v13

# 指定現有資料；不重新生成或改切分。
bash scripts/run_v13_ablation.sh --data-root /path/to/mecge_table1_repro

# 在 Linux/WSL 上選擇原有 Python 環境。
PYTHON=/path/to/python bash scripts/run_v13_ablation.sh --device cuda:0

# 多個 seed：僅在已有各 seed 的 v7 結果時比較；本腳本不補跑 v7。
bash scripts/run_v13_ablation.sh --seeds 3407 42 2026

# 中斷續訓、僅重新彙整。
bash scripts/run_v13_ablation.sh --resume
bash scripts/run_v13_ablation.sh --collect-only
```

Windows 的 PowerShell 可直接使用同一訓練環境：

```powershell
python src/v13_experiments.py --list
python src/v13_experiments.py --models v13 --device cuda:0
python src/v13_experiments.py
```

預設資料位置是 `data/mecge_table1_repro/raw/dataset_bw_nv1.pkl` 與 `dataset_bw_nv2.pkl`。
robustness 另需正確配對的 `rnd_test_nv1.npy`、`rnd_test_nv2.npy`；缺少時不宣稱完成區間驗證。
可用 `--pkl-file '/path/dataset_bw_nv{nv}.pkl'` 與 `--rnd-test '/path/rnd_test_nv{nv}.npy'` 指定。
現有資料切分、batch size 64、AdamW、lr 5e-5、30 epochs、無 early stopping、單步推論均沿用。

## 評估與驗收

`runs/v13_performance/` 是獨立輸出根目錄。
主要 checkpoint 仍依 composite validation loss 選擇，`best_val_ssd.pt` 是獨立的輔助對照；
兩者的 test 報表分開輸出。應在看 test 結果前固定使用哪個 selector，不能看 test 選最好的一個。
deterministic validation 另外記錄 SSD、MAD、官方 PRD、CosSim，包含最後不足一批的樣本，
且不消耗訓練 RNG。官方 PRD 的 clean 均值在完整 validation 上計算，而不是逐 batch 計算。

依本次確認，驗收預設只看 NV1＋NV2 合併結果（`--scope combined`），要求：

```text
SSD_v13 <= SSD_v7
PRD_v13 <= PRD_v7
MAD_v13 <  MAD_v7
CosSim_v13 > CosSim_v7
```

驗收針對四個指標的平均值；std 保留作描述，單一 seed 的比較不代表統計顯著。
逐 NV 與 robustness 報表可供診斷，但本次不列入通過條件，也不啟用 `--require-robustness`。
為產生合併結果，訓練仍需涵蓋 NV1、NV2，六個變體合計 12 次訓練。
請用 `python src/v13_acceptance.py --help` 查看 CSV、模型名稱與範圍參數。
缺資料、seed 或樣本數不匹配、非有限值均不能判為達標。
工具不挑選模型、checkpoint 或最佳 seed，也不以未達標時退回 v7 的方式冒充 v13 改善。

原始結果資料夾目前只有合併 CSV，無法從平均值重建可靠的逐 NV 結果或配對信賴區間。
要正式驗收，請保留每個 NV 的原始 predictions、resolved config、資料／程式 hash、checkpoint selector。
本次比較沿用歷史合併表與 seed3407，合併指標達標不表示每個 NV 或每個強度區間均改善。

## 程式驗證

```bash
python -B -m unittest discover -s tests -p 'test_v13*.py'
```

合成資料的 forward/backward、RNG、續訓、評估與驗收測試，只驗證程式行為，不代表 QTDB 效能。

本次實際驗證：

- 初版安裝至 Windows 專案目錄後，39 項 v13 測試全部通過。
- 本次合併驗收／排除 v7 調整通過 8 項 pipeline 與 16 項 acceptance 測試，
  包含拒絕 v7 訓練選項、拒絕以 override 退回原始 v7，以及僅有合併 CSV 即可驗收。
- 27 項既有 v12 回歸測試通過；159 個既有 source/script/test 檔案的 SHA-256 保持不變。
- 使用完整 v13 模型在兩份合成 NV 資料上各訓練 1 epoch，成功輸出兩個 selector 的
  checkpoint、逐 NV／合併表、四個 robustness 區間報表與完整 manifest。
- CPU 測試環境為 Python 3.12、PyTorch 2.14.1+cpu、NumPy 2.5.3、SciPy 1.18.1。
  這次未在專案 requirements 指定的 PyTorch 2.2.2 或 CUDA 環境驗證。
- 真實 `data/mecge_table1_repro/raw/` 目錄目前為空，未執行 QTDB 正式訓練。
