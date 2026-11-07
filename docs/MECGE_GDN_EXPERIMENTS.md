# MECG-E / GDN2：M0、M1、M2 消融實驗

本輪只比較時間與頻率模組，使用既有 `mecg_e_no_early_stop` 官方 pipeline，雜訊限定 baseline wander。

| Variant | 時間模組 | 頻率模組 |
| --- | --- | --- |
| m0 | 原始 Bi-Mamba | 原始 Bi-Mamba |
| m1 | Bi-GDN2 | 原始 Bi-Mamba |
| m2 | Bi-GDN2 | Bi-GDN2 |

四個 time-frequency blocks 均依此替換。保留 STFT/iSTFT、Encoder、Magnitude/Phase Decoder、原始 tlinear/flinear 投影、residual 與三項 loss。

## 固定協定

- 官方 `references/MECG-E/config/MECGE_phase.yaml` 與 `pipeline.py`。
- 30 epochs，無 early stopping；測試使用最低 validation loss 的 `best_model.pt`。
- seed 3407、batch size 96、learning rate 1e-4。
- AdamW betas=(0.8, 0.99)，ExponentialLR gamma=0.99；其餘 optimizer 行為沿用官方 pipeline。
- 原有 train/validation split（validation 30%，random_state=1）、512 samples、資料前處理與 loss。
- BW nv1/nv2，預設三個模型各跑兩個 noise versions，共六組。
- 指標直接呼叫既有 SSD、CosSim、PRD、MAD 實作；MAD 是 maximum absolute distance。
- Robustness 直接沿用既有 `rnd_test` 分組與邊界規則，以及 nv1/nv2 合併統計。

## GDN2 實作設定

直接載入 `references/GatedDeltaNet-2/lit_gpt/gdn2.py` 和其官方 kernels，不載入 Lit-GPT 語言模型。
每個方向使用獨立 GDN2：hidden size 32、4 heads、head dimension 32、expand_v=1、short convolution size 4、conv_bias=False、allow_neg_eigval=False、norm_eps=1e-5。
內部 key/value 維度為 128；這是本次固定的容量設定，並不宣稱與 Mamba 參數量相等。

雙向處理保留原本的 pre-norm 與方向內 residual，反向輸出翻回原序列，再串接為 64 維，交由原有投影層處理。
GDN2 初始化保持官方方法；未替換的層保留原始初始化，額外初始化不推進 CPU RNG，以維持資料載入器的隨機狀態。
官方 training chunk kernel 與短序列 inference recurrent kernel 均保持原樣，沒有增加混合精度或修改 loss。

## 執行

以下命令在專案根目錄 `rl_exp` 執行，使用已安裝 MECG-E/Mamba 與官方 GDN2 所需 PyTorch、CUDA、Triton、FLA 等套件的環境。
GDN2 相依版本與安裝方式以本地 `references/GatedDeltaNet-2/requirements.txt`、`Dockerfile` 為準。

```bash
# 查看計畫，不建立輸出、不訓練
bash scripts/run_mecge_gdn_ablation.sh --dry-run

# 在訓練 GPU 上先檢查真正的三個模型：forward/backward/optimizer/inference
python3 -B src/mecge_gdn_smoke.py --device cuda:0

# 一次依序執行六組實驗並彙整
bash scripts/run_mecge_gdn_ablation.sh --device cuda:0

# 單獨執行指定消融
bash scripts/run_mecge_gdn_ablation.sh --models m1 m2 --nv all --device cuda:0

# 中斷後續訓；已完成且檔案驗證通過的組合會略過
bash scripts/run_mecge_gdn_ablation.sh --resume

# 重新使用最佳 checkpoint 推論，或只重建統計表
bash scripts/run_mecge_gdn_ablation.sh --eval-only
bash scripts/run_mecge_gdn_ablation.sh --collect-only
```

資料預設位於 `data/mecge_table1_repro/raw/dataset_bw_nv{1,2}.pkl`。
Amplitude 優先使用對應的 `rnd_test_nv{1,2}.npy`，其次使用既有共用 `rnd_test.npy`。
可以指定 `--data-root`、`--pkl-file '/path/dataset_bw_nv{nv}.pkl'`、`--rnd-test '/path/rnd_test_nv{nv}.npy'` 和 `--run-root`。
自訂資料需搭配同順序、同長度的 amplitude array。

## 輸出與重跑

預設輸出根目錄為 `runs/mecge_gdn_ablation/`：

- `mecge_gdn_m{0,1,2}/`：各組 configs、manifests、checkpoint、log、results、controlled_tests。
- `official_results/`：標準 `[X_test, y_test, y_pred]` pickle。
- `analysis/table1_comparison__official_nv1_nv2.csv`：合併兩個 noise versions 的三模型比較。
- `analysis/robustness_comparison__official_nv1_nv2.csv`：合併兩個 noise versions 的分組比較。
- `analysis/` 同時包含 nv1、nv2、all 的既有格式比較表。

Manifest 記錄 resolved config、資料／amplitude／來源 SHA256 與完成後結果、最佳 checkpoint 的 SHA256。
設定、資料或來源改動時，使用新的 `--run-root`；已完成的產物若遺失或變動會報錯，避免混用。
`--models` 控制此次執行的模型；比較表會彙整同一 run root 中已存在的結果。

## 驗證

```bash
python3 -B -m unittest discover -s tests -p 'test_mecge_gdn_*.py' -v
```

Runner 測試包含固定協定、六組排程、dry-run、manifest 保護及六組合成資料通過既有統計流程。
CPU adapter 測試需 PyTorch，驗證反向還原、residual、梯度、替換範圍與 RNG。
GPU smoke test 必須在實際訓練環境執行；合成資料的統計測試不能替代官方 CUDA kernels 或完整訓練驗證。
