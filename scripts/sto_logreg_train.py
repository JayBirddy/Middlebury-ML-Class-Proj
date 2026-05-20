import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

from src.sto_logreg_model import StochasticLogisticRegression
from src.load_data import load_data, split_and_scale
from metrics.fairness_audit import FairnessAuditor

# output directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "sto_logreg")
graph_dir    = os.path.join(PROJECT_ROOT, "graphs",  "sto_logreg")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(graph_dir,  exist_ok=True)

# training
def train(X_train, y_train, pos_weight_val, batch_size=256, epochs=50, lr=1e-3):
    X_t    = torch.tensor(X_train, dtype=torch.float32)
    y_t    = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model     = StochasticLogisticRegression(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = []

    print(f"\nTraining: {epochs} epochs | batch={batch_size} | lr={lr} | pos_weight={pos_weight_val:.2f}")
    print(f"Steps per epoch: {len(loader)}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — loss: {avg_loss:.4f}")

    return model, history


# full evaluations
def evaluate(model, X_test, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_test, dtype=torch.float32))
        ).squeeze().numpy()

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    # comprehensive metrics
    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    auroc     = roc_auc_score(y_test, probs)

    print(f"  Accuracy:             {accuracy:.4f}  ")
    print(f"  Precision (PPV):      {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-score:             {f1:.4f}")
    print(f"  AUROC:                {auroc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")

    # confusion matrix plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, preds, labels=[0, 1]),
        display_labels=['Not Readmitted', 'Readmitted']
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Stochastic LogReg — Confusion Matrix (τ={threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "confusion_matrix.png"), dpi=180)
    plt.show()
    print("Saved: confusion_matrix.png")

    return probs, {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auroc':     auroc,
    }


# plots
def plot_loss(history):
    plt.figure(figsize=(7, 3))
    plt.plot(range(1, len(history) + 1), history, color='steelblue', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Avg batch loss')
    plt.title('Stochastic LogReg — Training Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "training_loss.png"), dpi=150)
    plt.show()
    print("Saved: training_loss.png")


def threshold_sweep(probs, y_test):
    """
    Plot sensitivity and precision across thresholds 0.05–0.95.
    Shows that default τ=0.5 is arbitrary and suboptimal for clinical use.
    """
    thresholds    = np.arange(0.05, 0.95, 0.01)
    sensitivities = []
    ppvs          = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            sensitivities.append(0); ppvs.append(0); continue
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sensitivities.append(tp / (tp + fn + 1e-9))
        ppvs.append(tp / (tp + fp + 1e-9))

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, sensitivities, label='Sensitivity (Recall)', color='steelblue')
    plt.plot(thresholds, ppvs,          label='Precision (PPV)',      color='coral')
    plt.axvline(0.5, linestyle='--', color='gray', alpha=0.5, label='default τ=0.5')
    plt.xlabel('Classification threshold τ')
    plt.ylabel('Score')
    plt.title('Stochastic LogReg — Threshold Sweep')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "threshold_sweep.png"), dpi=150)
    plt.show()
    print("Saved: threshold_sweep.png")

if __name__ == '__main__':

    # load and split
    X, y, demo_raw = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols, demo_test = \
        split_and_scale(X, y, demo_raw)

    # class weights
    n_neg          = (y_train == 0).sum()
    n_pos          = (y_train == 1).sum()
    pos_weight_val = n_neg / n_pos
    print(f"\nClass weight (pos_weight): {pos_weight_val:.2f}")

    # train
    model, history = train(
        X_train, y_train,
        pos_weight_val=pos_weight_val,
        batch_size=256, epochs=50, lr=1e-3
    )

    # evals
    probs, metrics = evaluate(model, X_test, y_test, threshold=0.5)

    # plots
    plot_loss(history)
    threshold_sweep(probs, y_test)

    # fairness audit
    print("\nRunning fairness audit...")
    auditor = FairnessAuditor(
        X_demo    = demo_test,
        y_test    = y_test,
        threshold = 0.5,
        gap       = 0.05
    )
    auditor.add_model("Stochastic LogReg", probs)
    auditor.run_audit()
    auditor.summary()
    auditor.plot_audit(
        save_path=os.path.join(graph_dir, "fairness_audit.png")
    )

    # save artifacts
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    demo_test.to_csv(os.path.join(output_dir, "demo_test.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(output_dir, "y_test.csv"),
                              index=False, header=True)
    print("\nAll artifacts saved:")