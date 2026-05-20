import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.nn_model_2_layer import TwoLayerNet
from src.load_data import load_data_no_cci, split_and_scale
from metrics.fairness_audit import FairnessAuditor

# output paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "nn_base")
graph_dir    = os.path.join(PROJECT_ROOT, "graphs",  "nn_base")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(graph_dir,  exist_ok=True)


# training
def train(model, loader, X_test_t, y_test_t, k_epochs=30, lr=1e-3):
    """
    Training loop with per-epoch validation loss and accuracy tracking.
    Uses StepLR scheduler: halves learning rate every 10 epochs.
    Returns train_losses, val_losses, val_accs history lists.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses, val_accs = [], [], []

    print(f"\nTraining: {k_epochs} epochs | lr={lr}")
    print("-" * 55)

    for epoch in range(k_epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()

        # Validation pass
        model.eval()
        with torch.no_grad():
            logits_val = model(X_test_t)
            val_loss   = criterion(logits_val, y_test_t).item()
            preds_val  = (torch.sigmoid(logits_val) >= 0.5).float()
            acc        = (preds_val == y_test_t).float().mean().item()

        train_losses.append(np.mean(batch_losses))
        val_losses.append(val_loss)
        val_accs.append(acc)

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch+1:>3}/{k_epochs} | "
                f"Train Loss: {train_losses[-1]:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {acc:.4f}"
            )

    return train_losses, val_losses, val_accs


# evals
def evaluate(model, X_test_t, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy().flatten()

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    auroc     = roc_auc_score(y_test, probs)

    print(f"\n{'='*48}")
    print(f"  Evaluation (threshold = {threshold})")
    print(f"{'='*48}")
    print(f"  Accuracy:             {accuracy:.4f}  ")
    print(f"  Precision (PPV):      {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-score:             {f1:.4f}")
    print(f"  AUROC:                {auroc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_test, preds,
                                target_names=["No Readmit", "Readmitted"]))

    return probs, {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auroc':     auroc,
    }


# plots
def plot_training_curves(train_losses, val_losses, val_accs):
    """Loss and accuracy curves across epochs — both train and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses,   label="Val Loss",   color="coral")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(val_accs, color="green", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "training_curves.png"), dpi=150)
    plt.show()
    print("Saved: training_curves.png")


def plot_confusion_matrix(y_test, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1])
    disp  = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Readmit", "Readmitted"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Two-Layer Net — Confusion Matrix (τ={threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def threshold_sweep(probs, y_test):
    """
    Plot sensitivity and precision across thresholds 0.05-0.95.
    Shows that default τ=0.5 is arbitrary and suboptimal for clinical use.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    sens, ppvs = [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            sens.append(0); ppvs.append(0); continue
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn + 1e-9))
        ppvs.append(tp / (tp + fp + 1e-9))

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, sens, label="Sensitivity (Recall)", color="steelblue")
    plt.plot(thresholds, ppvs, label="Precision (PPV)",      color="coral")
    plt.axvline(0.5, linestyle="--", color="gray", alpha=0.5, label="default τ=0.5")
    plt.xlabel("Classification threshold τ")
    plt.ylabel("Score")
    plt.title("Two-Layer Net — Threshold Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "threshold_sweep.png"), dpi=150)
    plt.show()
    print("Saved: threshold_sweep.png")


if __name__ == "__main__":

    # load and split
    X, y, demo = load_data_no_cci()
    X_train, X_test, y_train, y_test, scaler, feature_cols, demo_test = \
        split_and_scale(X, y, demo)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
    )

    # model
    model = TwoLayerNet(input_dim=X_train_t.shape[1])
    print(f"\nModel: TwoLayerNet | input_dim={X_train_t.shape[1]}")

    # train
    train_losses, val_losses, val_accs = train(
        model, loader, X_test_t, y_test_t, k_epochs=30, lr=1e-3
    )

    # eval
    probs, metrics = evaluate(model, X_test_t, y_test, threshold=0.5)

    # plots
    plot_training_curves(train_losses, val_losses, val_accs)
    plot_confusion_matrix(y_test, probs, threshold=0.5)
    threshold_sweep(probs, y_test)

    # fairness audit
    print("\nRunning fairness audit...")
    auditor = FairnessAuditor(
        X_demo    = demo_test,
        y_test    = y_test,
        threshold = 0.5,
        gap       = 0.05
    )
    auditor.add_model("Two-Layer Net", probs)
    auditor.run_audit()
    auditor.summary()
    auditor.plot_audit(
        save_path=os.path.join(graph_dir, "fairness_audit.png")
    )

    # save artifacts
    torch.save(model.state_dict(),
               os.path.join(output_dir, "nn_base.pth"))
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(output_dir, "input_dim.txt"), "w") as f:
        f.write(str(X_train_t.shape[1]))

    demo_test.to_csv(os.path.join(output_dir, "demo_test.csv"), index=False)
    pd.Series(y_test).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False, header=True
    )

    print("\nAll artifacts saved:")