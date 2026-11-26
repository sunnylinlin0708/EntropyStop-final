
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# B 的相關性結果
b = pd.read_csv(
    ROOT / "results/analysis/correlation_summary.csv"
)

# C 的相關性結果
c = pd.read_csv(
    ROOT / "results/altmetric/analysis/correlation.csv"
)

# 將 Entropy 相關性方向反轉
b["EntropyAE"] = -b["pearson_r_entropy_auc"]

b = b[["dataset", "seed", "EntropyAE"]]

# 將兩個新方法轉成欄位
c = c.pivot(
    index=["dataset", "seed"],
    columns="method",
    values="pearson_r"
).reset_index()

# 依照資料集與 seed 合併
merged = b.merge(
    c,
    on=["dataset", "seed"],
    how="inner"
)

# 確認配對完整
if len(merged) != 60:
    raise ValueError(
        f"Expected 60 dataset-seed pairs, got {len(merged)}"
    )

out = ROOT / "results/altmetric/analysis"

merged.to_csv(
    out / "correlation_comparison.csv",
    index=False
)

# 每個資料集取五次 seed 平均
summary = merged.groupby("dataset")[
    ["EntropyAE", "BSSAE", "SilhouetteAE"]
].mean()

summary.to_csv(
    out / "correlation_mean.csv"
)

print(summary)

# 繪製長條圖
ax = summary.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Direction-aligned Pearson r")
ax.set_xlabel("Dataset")
ax.set_title("Correlation with AUC")
ax.set_ylim(-1, 1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

figures = ROOT / "figures/altmetric"
figures.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures / "correlation_comparison.png",
    dpi=300
)

plt.show()
