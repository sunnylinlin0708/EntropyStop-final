# B: 實驗框架與熵／AUC 分析（交接說明）

本資料夾是 **新增／覆蓋到 A 的 `EntropyStop-final/` 專案根目錄** 的 B 部分檔案，不是完整原始專案。
請先 `git clone https://github.com/sunnylinlin0708/EntropyStop-final.git`，再將 `src/`、`scripts/`、`tests/` 合併到 clone 下的同名路徑。**不修改** A 的程式碼、基準 CSV 或 `configs/datasets.txt`。

## A → B 驗收（Windows / VS Code）

在 PowerShell 開啟專案資料夾。請先依 A 的 README 建立虛擬環境、安裝 `requirements.txt`，確認 `import torch_geometric` 成功：

```powershell
python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available())"
python scripts/b_validate.py
```

預設檢查 `18_Ionosphere.npz`、`14_glass.npz`，各 seed 0，輸出 `results/analysis/b_validation.csv`。與 A 的 `results/baseline/baseline_results.csv` 比較。若 AUC／AP 相差大於 0.03 會回報錯誤；小於 0.03 也不保證每一步都完全一致，仍須確認 `stop_iter`、`total_iter`、套件版本及浮點差異。

## 新停止方法介面（給 C、D）

`src/common.py` 提供 `run_experiment(...)`，不需要重複寫資料載入／AE 訓練。自訂停止規則須有：

- `step(metric_value: float) -> bool`：每次 optimizer 更新後被呼叫，回傳是否提早停止。
- `best_iter: int`：0-based 的「被選最佳模型」步數索引。
- `improved: bool`：這次更新是否應保存最佳模型快照。**第一次呼叫必須為 True**。

例如：

```python
from src.common import run_experiment, FixedStepStopRule
r = run_experiment("data/18_Ionosphere.npz", method="Fixed50AE",
                   seed=0, stop_rule=FixedStepStopRule(50))
print(r.baseline_row())
```

可直接執行 `python scripts/b_fixed50_example.py`。

C 可以再提供 `metric_fn(losses: np.ndarray) -> float`，只輸入「評估子集每筆 reconstruction loss」，不允許使用標籤。若新指標越大越好，令 `metric_fn` 回傳負值，以共用的 `EntropyStopRule` 執行相同停止邏輯。

D 則可自行建立使用歷史熵值的 EMA／MA stopping rule，必須僅使用過去和現在的資料。

## 逐步紀錄工具

```python
from src.logger import CSVStepLogger
with CSVStepLogger("results/analysis/steps.csv") as logger:
    r = run_experiment("data/18_Ionosphere.npz", method="EntropyAE", seed=0,
                       epochs=500, full_run=True, step_logger=logger)
```

產出欄位保持 PDF 指定：`dataset,method,seed,iter,metric_value,AUC,AP`。

**重要：** 作者的 EntropyStop 每個 mini-batch 更新後檢查一次，因此 `iter` 是 **0-based optimizer update / mini-batch step**，不是 epoch。`stop_iter` 沿用 A 的命名，但意思是 **最佳 checkpoint 的 update index**，並不一定是觸發停止的 update；`total_iter` 是實際 optimizer updates 的數量。

為了節省時間，預設每個 update 都記錄熵，但 AUC/AP 僅在 **epoch 結束** 時計算；其他列的 AUC/AP 留空。這些標籤只供事後分析，不能用於停止判斷。如果為了精細分析要每 update 算 AUC，改用 `auc_every_step=True`（大型資料集非常慢）。

## B：熵 vs. AUC 曲線／失敗分析

**先跑小資料集，別一開始跑 12 × 5：**

```powershell
python scripts/b_curves.py --datasets 18_Ionosphere.npz 14_glass.npz --seeds 0 --epochs 500
```

先測管線可以把 `--epochs 500` 暫改 `--epochs 10`。確認兩筆沒問題，再依 PDF 跑全部資料集 × seed 0~4：

```powershell
python scripts/b_curves.py --seeds 0 1 2 3 4 --epochs 500
```

輸出：

- `results/analysis/curves/<dataset>_seed<seed>.csv`：每個 update 的 loss entropy，AUC/AP 在 epoch 結束時才有值。
- `results/analysis/correlation_summary.csv`：Pearson r、選中 checkpoint 的 AUC、epoch-sampled 最佳 AUC、gap、正常標籤中 loss 超過異常平均 loss 的比例。
- `figures/analysis/*_entropy_auc.png`：雙軸曲線；綠線是最佳 checkpoint、紅線是 *would-trigger* 的停止步數（full-run 模擬）。
- `figures/analysis/*_loss_hist.png`：用標籤**事後**分析正常／異常損失重疊。
- `figures/analysis/pearson_r_distribution.png`：各 dataset×seed 相關係數分布。

`correlation_summary.csv` 的分類是先設定的**描述性判準**，不是論文數值也不是使用 AUC 調 stop hyperparameter；最終應仔細閱讀曲線再撰寫成功／失敗與原因的一頁分析。

## 必須注意的限制

1. A 的基準跑 250 epochs，而 B 完整曲線使用 500 epochs；兩者用途不同，B 不得混為基準對照數據。
2. 在 500-epoch full-run 模式中，`EntropyStopRule` 在首次觸發時就凍結其選中 checkpoint，但 **AE 會繼續訓練**以提供完整曲線；所以 500-epoch 訓練耗時不能直接和 A 早停耗時相比。
3. `false_normal_ratio` 使用 `y` 真實標籤，只能作為**事後**失敗分析；不能放入停止指標、挑參數或挑種子。
4. 真正跨硬體可重現性尚未驗證：必須在自己的 VS Code 環境執行 `b_validate.py`。
5. 目前 C、D 尚未驗收；合併 main 前先請 C 實作固定 50 步停止法，確認欄位及結果格式符合 PDF。

## B 實驗框架與分析

`src/common.py` 提供 `run_experiment` 與可替換的 `StopRule`；
`src/logger.py` 的逐步欄位為 `dataset,method,seed,iter,metric_value,AUC,AP`。
`iter` 是從 0 起算的 optimizer update；停止判斷只讀取無標籤指標，
AUC 與 AP 僅作事後評估，預設在每個 epoch 結束時記錄。

先驗證與 A 基準的一致性、再跑 50 更新的替換停止範例：

```powershell
python scripts/b_validate.py
python scripts/b_fixed50_example.py
```

跑滿 500 epochs 的全部資料集與 0–4 種子：

```powershell
python scripts/b_curves.py --seeds 0 1 2 3 4 --epochs 500
```

逐步紀錄在 `results/analysis/curves/`；彙整表在
`results/analysis/correlation_summary.csv`；曲線與 loss 分布在
`figures/analysis/`。分析結論見 `docs/EntropyStop_B_Analysis_zhTW.docx`，
完整方法與限制見 `B_README.md`。

