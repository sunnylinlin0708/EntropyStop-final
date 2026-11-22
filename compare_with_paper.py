"""
compare_with_paper.py  —  第一棒（A）：把我們的基準結果和論文、作者附的結果對照

在作者 repo 根目錄執行（先跑完 run_baseline.py）：

    python compare_with_paper.py

讀取：
  results/baseline/baseline_results.csv        我們跑的結果（5 個種子）
  configs/paper_table9_10.csv                  論文 Table 9（AUC）與 Table 10（AP）的數字
  Exp1/results/EntropyAE-*.csv                 作者 repo 附的 EntropyAE 結果（3 次）
輸出：
  results/baseline/compare_with_paper.csv      每個資料集 × 方法一列
  畫面上列出差距超過 0.03 的項目，需要在對照表中寫下可能原因
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OURS = os.path.join(ROOT, "results", "baseline", "baseline_results.csv")
PAPER = os.path.join(ROOT, "configs", "paper_table9_10.csv")
OUT = os.path.join(ROOT, "results", "baseline", "compare_with_paper.csv")
THRESHOLD = 0.03

ours = pd.read_csv(OURS)
summary = (ours.groupby(["dataset", "method"])
           .agg(our_AUC_mean=("AUC", "mean"), our_AUC_std=("AUC", "std"),
                our_AP_mean=("AP", "mean"), our_AP_std=("AP", "std"),
                n_seeds=("seed", "nunique"), mean_time_sec=("train_time_sec", "mean"))
           .reset_index())

paper = pd.read_csv(PAPER)
table = summary.merge(paper, on=["dataset", "method"], how="left")

author_files = glob.glob(os.path.join(ROOT, "Exp1", "results", "EntropyAE-*.csv"))
if author_files:
    author = pd.concat([pd.read_csv(f) for f in author_files])
    author = (author.groupby("Dataset").agg(author_repo_AUC=("AUC", "mean"), author_repo_AP=("AP", "mean"))
              .reset_index().rename(columns={"Dataset": "dataset"}))
    author["method"] = "EntropyAE"
    table = table.merge(author, on=["dataset", "method"], how="left")

table["diff_AUC_vs_paper"] = table["our_AUC_mean"] - table["paper_AUC"]
table["diff_AP_vs_paper"] = table["our_AP_mean"] - table["paper_AP"]
table = table.round(4).sort_values(["dataset", "method"])
table.to_csv(OUT, index=False)

pd.set_option("display.width", 200)
print(table[["dataset", "method", "our_AUC_mean", "paper_AUC", "diff_AUC_vs_paper",
             "our_AP_mean", "paper_AP", "diff_AP_vs_paper"]].to_string(index=False))

print("\n平均（各方法在所有資料集上）：")
print(table.groupby("method")[["our_AUC_mean", "paper_AUC", "our_AP_mean", "paper_AP"]].mean().round(4))

big = table[(table["diff_AUC_vs_paper"].abs() > THRESHOLD) | (table["diff_AP_vs_paper"].abs() > THRESHOLD)]
print(f"\n與論文差距超過 {THRESHOLD} 的項目（共 {len(big)} 項），請在對照表寫下可能原因：")
print(big[["dataset", "method", "diff_AUC_vs_paper", "diff_AP_vs_paper"]].to_string(index=False)
      if len(big) else "（無）")
print(f"\n已輸出：{OUT}")
