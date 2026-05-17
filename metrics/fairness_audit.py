# Equity analysis through metric analysis on test set predictions and the dataset's original demographic column 
# Compute AUROC, recall, precision, and positive prediction rate broken out by each demographic subgroup 
# Flag where performance gaps exceed a threshold (default: 0.5) as well as a gap threshold for AUROC (default 0.05)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "nn")
os.makedirs(output_dir, exist_ok=True)

class FairnessAuditor:
    SUBGROUPS = {"Race": "race", "Gender": "gender", "Age": "age"}
    AGE_LABELS = {
        0: "[0-10)",  1: "[10-20)", 2: "[20-30)", 3: "[30-40)", 4: "[40-50)",
        5: "[50-60)", 6: "[60-70)", 7: "[70-80)", 8: "[80-90)", 9: "[90-100)"
    }

    def __init__(self, X_demo, y_test, threshold=0.5, gap=0.05):
        self.X   = X_demo.reset_index(drop=True).copy()
        self.y   = np.array(y_test)
        self.t   = threshold
        self.gap = gap
        self.models  = {}
        self.results = {}

        # convert ordinal age back to readable label if needed
        if self.X["age"].dtype in [np.int64, np.float64]:
            self.X["age"] = self.X["age"].map(self.AGE_LABELS).fillna("Unknown")

    def add_model(self, name, probs):
        self.models[name] = np.array(probs)

    # core computations for metric
    def _metrics(self, mask, probs):
        y, p = self.y[mask], probs[mask]
        if len(y) < 30 or y.sum() < 5:
            return None
        preds = (p >= self.t).astype(int)
        try:
            auroc = roc_auc_score(y, p)
        except Exception:
            auroc = np.nan
        return {
            "n":          int(mask.sum()),
            "prevalence": round(y.mean(), 4),
            "auroc":      round(auroc, 4),
            "recall":     round(recall_score(y, preds, zero_division=0), 4),
            "precision":  round(precision_score(y, preds, zero_division=0), 4),
            "ppr":        round(preds.mean(), 4),
        }

    def run_audit(self):
        for name, probs in self.models.items():
            self.results[name] = {}
            overall = roc_auc_score(self.y, probs)
            print(f"\n{'='*60}\n  {name}  |  Overall AUROC={overall:.4f}\n{'='*60}")

            for label, col in self.SUBGROUPS.items():
                if col not in self.X.columns:
                    continue
                rows = []
                for val in sorted(self.X[col].dropna().unique()):
                    m = self._metrics((self.X[col] == val).values, probs)
                    if m:
                        rows.append({"subgroup": str(val), **m})

                if not rows:
                    continue

                df = pd.DataFrame(rows).set_index("subgroup")
                self.results[name][label] = df
                best = df["auroc"].max()

                print(f"\n  ── {label}")
                for sg, row in df.iterrows():
                    flag = " !" if (best - row["auroc"]) >= self.gap else ""
                    print(f"    {sg:<20} N={row['n']:<6} "
                          f"AUROC={row['auroc']:.3f}  "
                          f"Recall={row['recall']:.3f}  "
                          f"PPR={row['ppr']:.3f}{flag}")
        return self.results

    def summary(self):
        print(f"\n{'='*50}\n  Max AUROC Gap by Subgroup\n{'='*50}")
        for name in self.models:
            for label, df in self.results.get(name, {}).items():
                gap  = df["auroc"].max() - df["auroc"].min()
                flag = "! YES" if gap >= self.gap else "no"
                print(f"  {name:<22} {label:<10} gap={gap:.4f}  flagged={flag}")

    # plot visual
    def plot_audit(self, figsize=(16, 12), save_path=None):
        labels = [l for l in self.SUBGROUPS if any(
            l in self.results.get(m, {}) for m in self.models
        )]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        fig, axes = plt.subplots(len(labels), 1, figsize=figsize)
        if len(labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            subgroups = list(dict.fromkeys(
                sg for m in self.models
                for sg in self.results.get(m, {}).get(label, pd.DataFrame()).index
            ))
            if not subgroups:
                continue

            x     = np.arange(len(subgroups))
            width = 0.8 / len(self.models)

            for i, (name, color) in enumerate(zip(self.models, colors)):
                df     = self.results.get(name, {}).get(label, pd.DataFrame())
                aurocs = [df.loc[sg, "auroc"] if sg in df.index else np.nan
                          for sg in subgroups]
                offset = (i - len(self.models) / 2 + 0.5) * width
                bars   = ax.bar(x + offset, aurocs, width * 0.9,
                                label=name, color=color, alpha=0.85,
                                edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, aurocs):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.003, f"{val:.3f}",
                                ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(subgroups, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("AUROC")
            ax.set_title(f"AUROC by {label}", fontweight="bold")
            ax.set_ylim(0.45, 0.85)
            ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle("Fairness Audit — AUROC by Demographic Subgroup",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = save_path or os.path.join(output_dir, "fairness_audit.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()