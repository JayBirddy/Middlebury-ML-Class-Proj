# Threshold calibration analysis and utility analysis
# Threshold sweep to from 0.01-0.99 for Precision, Recall, F1, F-beta to find sweet spot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    brier_score_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve


class ThresholdAnalyzer:
    """
    Threshold sweep, calibration, and clinical cost analysis for binary
    classification models predicting hospital readmission.
    """

    def __init__(self, y_test, thresholds=None):
        self.y = np.array(y_test)
        self.thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 100)
        self.models = {}       # name -> probs
        self.sweep_results = {}  # name -> DataFrame of metrics per threshold

    def add_model(self, name, probs):
        self.models[name] = np.array(probs)

    # ── 1. Threshold Sweep ────────────────────────────────────────────────────
    def run_threshold_sweep(self, beta=2.0):
        """
        Compute precision, recall, F1, F-beta across all thresholds.

        beta : float
            Beta for F-beta score. beta>1 weights recall more than precision.
            beta=2 is common in clinical settings where false negatives are costly.
            This is different from F1 (beta=1) and should be discussed in your paper.
        """
        self.beta = beta
        print(f"\n{'='*65}")
        print(f"  THRESHOLD SWEEP  (F-beta with beta={beta})")
        print(f"{'='*65}")

        for name, probs in self.models.items():
            rows = []
            for t in self.thresholds:
                preds = (probs >= t).astype(int)
                p  = precision_score(self.y, preds, zero_division=0)
                r  = recall_score(self.y, preds, zero_division=0)
                f1 = f1_score(self.y, preds, zero_division=0)
                # F-beta: beta^2 * precision * recall / (beta^2 * precision + recall)
                denom = (beta**2 * p + r)
                fb = (1 + beta**2) * p * r / denom if denom > 0 else 0.0
                ppr = preds.mean()  # positive prediction rate
                rows.append({
                    "threshold": round(t, 3),
                    "precision": round(p, 4),
                    "recall":    round(r, 4),
                    "f1":        round(f1, 4),
                    f"f{beta}":  round(fb, 4),
                    "pos_rate":  round(ppr, 4),
                })

            df = pd.DataFrame(rows)
            self.sweep_results[name] = df

            best_f1_row = df.loc[df["f1"].idxmax()]
            best_fb_row = df.loc[df[f"f{beta}"].idxmax()]
            auroc = roc_auc_score(self.y, probs)
            brier = brier_score_loss(self.y, probs)

            print(f"\n  Model: {name}")
            print(f"  Overall AUROC: {auroc:.4f}   Brier Score: {brier:.4f}")
            print(f"  Best F1    threshold: {best_f1_row['threshold']:.3f}  "
                  f"→ Precision={best_f1_row['precision']:.3f}  Recall={best_f1_row['recall']:.3f}")
            print(f"  Best F{beta}   threshold: {best_fb_row['threshold']:.3f}  "
                  f"→ Precision={best_fb_row['precision']:.3f}  Recall={best_fb_row['recall']:.3f}")

        return self.sweep_results

    def plot_threshold_sweep(self, figsize=(15, 5)):
        """Plot Precision-Recall-F curves across thresholds for each model."""
        if not self.sweep_results:
            raise RuntimeError("Call run_threshold_sweep() first.")

        n = len(self.models)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        if n == 1:
            axes = [axes]

        colors = {"precision": "#2196F3", "recall": "#FF5722",
                  "f1": "#4CAF50", f"f{self.beta}": "#9C27B0"}

        for ax, (name, df) in zip(axes, self.sweep_results.items()):
            for metric, color in colors.items():
                label = metric if metric not in [f"f{self.beta}"] else f"F-{self.beta} (clinical)"
                ax.plot(df["threshold"], df[metric], color=color,
                        linewidth=2, label=label)

            # Mark best F-beta threshold
            best_fb = df.loc[df[f"f{self.beta}"].idxmax()]
            ax.axvline(best_fb["threshold"], color="#9C27B0",
                       linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(best_fb["threshold"] + 0.01, 0.05,
                    f"Clinical\noptimum\n{best_fb['threshold']:.2f}",
                    fontsize=7, color="#9C27B0")

            # Mark default 0.5
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(0.51, 0.92, "0.5\ndefault", fontsize=7, color="gray")

            ax.set_xlabel("Decision Threshold", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        fig.suptitle("Threshold Sensitivity Analysis\n(Purple dashed = clinically optimal threshold)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("threshold_sweep.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: threshold_sweep.png")

    # ── 2. Calibration Plot ───────────────────────────────────────────────────
    def plot_calibration(self, n_bins=10, figsize=(7, 6)):
        """
        Reliability diagram: fraction of positives vs. mean predicted probability.

        A perfectly calibrated model lies on the diagonal.
        Points above = model underestimates risk (bad: misses real cases).
        Points below = model overestimates risk (bad: unnecessary interventions).

        Brier Score: mean squared error of probabilities. Lower is better.
        A naive always-predict-prevalence baseline is included for reference.
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

        for (name, probs), color in zip(self.models.items(), colors):
            frac_pos, mean_pred = calibration_curve(self.y, probs, n_bins=n_bins, strategy="uniform")
            brier = brier_score_loss(self.y, probs)
            ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2,
                    markersize=6, label=f"{name} (Brier={brier:.4f})")

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

        # Naive baseline: always predict prevalence
        prev = self.y.mean()
        naive_brier = brier_score_loss(self.y, np.full_like(self.y, prev, dtype=float))
        ax.axhline(prev, color="gray", linestyle=":", linewidth=1,
                   label=f"Naive baseline (Brier={naive_brier:.4f})")

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives (Actual Rate)", fontsize=11)
        ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig("calibration_plot.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: calibration_plot.png")

    # ── 3. Clinical Cost Analysis ─────────────────────────────────────────────
    def plot_clinical_cost(self, fn_cost_ratio=5, figsize=(10, 5)):
        """
        Compute and plot expected clinical cost across thresholds.

        CLINICAL FRAMING:
        A false negative (missed readmission) means a patient is sent home
        without intervention and returns in crisis — costly and harmful.
        A false positive (unnecessary intervention) wastes resources but
        is less harmful. If we say FN costs 5× more than FP (fn_cost_ratio=5),
        we can compute total expected cost at each threshold and find the minimum.

        Cost at threshold t =
            fn_cost_ratio * FN_rate  +  1 * FP_rate

        where FN_rate and FP_rate are normalized by N.

        This framing turns your project from "we compared models" to
        "we found the clinically optimal operating point for each model."
        """
        self.fn_cost_ratio = fn_cost_ratio
        fig, axes = plt.subplots(1, len(self.models),
                                 figsize=figsize, sharey=True)
        if len(self.models) == 1:
            axes = [axes]

        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        self.optimal_cost_thresholds = {}

        for ax, (name, probs), color in zip(axes, self.models.items(), colors):
            costs = []
            for t in self.thresholds:
                preds = (probs >= t).astype(int)
                fn = ((preds == 0) & (self.y == 1)).sum()
                fp = ((preds == 1) & (self.y == 0)).sum()
                cost = (fn_cost_ratio * fn + fp) / len(self.y)
                costs.append(cost)

            costs = np.array(costs)
            opt_idx = np.argmin(costs)
            opt_t   = self.thresholds[opt_idx]
            opt_c   = costs[opt_idx]
            self.optimal_cost_thresholds[name] = opt_t

            ax.plot(self.thresholds, costs, color=color, linewidth=2)
            ax.axvline(opt_t, color="red", linestyle="--", linewidth=1.5,
                       label=f"Optimal t={opt_t:.2f}")
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1,
                       label="Default t=0.50")
            ax.scatter([opt_t], [opt_c], color="red", s=80, zorder=5)
            ax.text(opt_t + 0.02, opt_c + 0.002,
                    f"t={opt_t:.2f}\ncost={opt_c:.3f}",
                    fontsize=8, color="red")

            ax.set_xlabel("Decision Threshold", fontsize=10)
            ax.set_ylabel(f"Expected Cost\n(FN weight={fn_cost_ratio}x FP)", fontsize=9)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Clinical Cost Analysis  (FN costs {fn_cost_ratio}× FP)\n"
                     f"Red dashed = cost-minimizing threshold",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("clinical_cost.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: clinical_cost.png")

    # ── 4. Final Report ───────────────────────────────────────────────────────
    def optimal_thresholds_report(self, fn_cost_ratio=5):
        """
        Print a clean summary table comparing default vs optimal thresholds.
        This is what you put in the Results section of your paper.
        """
        if not self.sweep_results:
            raise RuntimeError("Call run_threshold_sweep() first.")

        print(f"\n{'='*75}")
        print(f"  OPTIMAL THRESHOLD REPORT  (FN cost = {fn_cost_ratio}× FP cost)")
        print(f"{'='*75}")
        print(f"  {'Model':<22} {'Metric':<15} {'Default (0.50)':<20} {'Optimal'}")
        print(f"  {'-'*68}")

        for name, probs in self.models.items():
            df = self.sweep_results[name]

            # Metrics at default 0.5
            idx_05 = (df["threshold"] - 0.5).abs().idxmin()
            row_05 = df.iloc[idx_05]

            # Metrics at cost-optimal threshold
            opt_t = self.optimal_cost_thresholds.get(name)
            if opt_t is None:
                continue
            idx_opt = (df["threshold"] - opt_t).abs().idxmin()
            row_opt = df.iloc[idx_opt]

            brier = brier_score_loss(self.y, probs)
            auroc = roc_auc_score(self.y, probs)

            print(f"\n  ── {name}  (AUROC={auroc:.4f}, Brier={brier:.4f})")
            for metric in ["precision", "recall", "f1", f"f{self.beta}"]:
                if metric not in df.columns:
                    continue
                label = metric if "f" not in metric or metric == "f1" else f"F-{self.beta}"
                v_05  = row_05[metric]
                v_opt = row_opt[metric]
                delta = v_opt - v_05
                sign  = "+" if delta >= 0 else ""
                print(f"    {label:<15} {v_05:<20.4f} {v_opt:.4f}  ({sign}{delta:.4f})")

            print(f"    {'Threshold':<15} {'0.500':<20} {opt_t:.4f}")