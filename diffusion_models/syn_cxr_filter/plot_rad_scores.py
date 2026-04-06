import pandas as pd
import matplotlib.pyplot as plt

csv_path = "sample_radgraph_scores.csv"
df = pd.read_csv(csv_path)

score_cols = [
    "radgraph_rg_e",
    "radgraph_rg_er",
    "radgraph_rg_bar_er",
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, score_cols):
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(col)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("radgraph_score_histograms.png", dpi=300, bbox_inches="tight")
print("Saved as radgraph_score_histograms.png")