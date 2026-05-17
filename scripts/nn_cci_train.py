import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, f1_score,
                             accuracy_score)
from sklearn.calibration import calibration_curve

from src.nn_model_1_layer import OneLayerNet
from src.load_data import load_data, split_and_scale
from metrics.fairness_audit import FairnessAuditor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "nn_cci")
graph_dir    = os.path.join(PROJECT_ROOT, "graphs",  "nn_cci")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(graph_dir,  exist_ok=True)


def make_data_loader(X, y, batch_size=256):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size, shuffle=True
    )

def train(model, train_loader, pos_weight, k_epochs=30, lr=1e-3):
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []
    for epoch in range(k_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{k_epochs} — loss: {avg:.4f}")
    return history

def evaluate(model, X_test, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_test, dtype=torch.float32))
        ).squeeze().numpy()

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    accuracy    = accuracy_score(y_test, preds)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1          = f1_score(y_test, preds, zero_division=0)
    auc_roc     = roc_auc_score(y_test, probs)
    auc_pr      = average_precision_score(y_test, probs)

    print(f"\n{'='*45}")
    print(f"  Evaluation (threshold = {threshold})")
    print(f"{'='*45}")
    print(f"  Accuracy:             {accuracy:.4f}")
    print(f"  Precision (PPV):      {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-score:             {f1:.4f}")
    print(f"  AUROC:                {auc_roc:.4f}")
    print(f"  AUC-PR:               {auc_pr:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")

    return probs, {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc_roc':   auc_roc,
        'auc_pr':    auc_pr,
    }

# -- plots (unchanged from original, just save to graph_dir) --
def plot_loss(history):
    plt.figure(figsize=(7, 3))
    plt.plot(range(1, len(history)+1), history, color='steelblue', linewidth=1.5)
    plt.xlabel('Epoch'); plt.ylabel('Avg batch loss')
    plt.title('Neural Net + CCI — Training Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "training_loss.png"), dpi=150)
    plt.show()

def plot_confusion_matrix(y_test, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: No','Pred: Yes'])
    ax.set_yticklabels(['True: No','True: Yes'])
    ax.set_title(f'Neural Net + CCI — Confusion Matrix (τ={threshold})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=14,
                    fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "confusion_matrix.png"), dpi=150)
    plt.show()

def threshold_sweep(probs, y_test):
    thresholds = np.arange(0.05, 0.95, 0.01)
    sens, ppvs = [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            sens.append(0); ppvs.append(0); continue
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        sens.append(tp/(tp+fn+1e-9)); ppvs.append(tp/(tp+fp+1e-9))
    plt.figure(figsize=(8,4))
    plt.plot(thresholds, sens, label='Sensitivity', color='steelblue')
    plt.plot(thresholds, ppvs, label='PPV',         color='coral')
    plt.axvline(0.5, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Threshold τ'); plt.ylabel('Score')
    plt.title('Neural Net + CCI — Threshold Sweep')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "threshold_sweep.png"), dpi=150)
    plt.show()

def plot_calibration(probs, y_test):
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6,4))
    plt.plot(mean_pred, frac_pos, marker='o', color='steelblue', label='Model')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction actually readmitted')
    plt.title('Neural Net + CCI — Calibration')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "calibration.png"), dpi=150)
    plt.show()


if __name__ == '__main__':
    X, y, demo = load_data()

    X_train, X_test, y_train, y_test, scaler, feature_cols, demo_test = \
    split_and_scale(X, y, demo)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader  = make_data_loader(X_tr_t, y_tr_t)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model      = OneLayerNet(input_dim=X_tr_t.shape[1])

    print("\nTraining...")
    history = train(model, loader, pos_weight, k_epochs=30, lr=1e-3)
    probs, metrics = evaluate(model, X_test, y_test)

    plot_loss(history)
    plot_confusion_matrix(y_test, probs)
    threshold_sweep(probs, y_test)
    plot_calibration(probs, y_test)

    # Fairness audit
    auditor = FairnessAuditor(X_demo=demo_test, y_test=y_test,
                          threshold=0.5, gap=0.05)
    auditor.add_model("Neural Net + CCI", probs)
    auditor.run_audit()
    auditor.summary()
    auditor.plot_audit(
        save_path=os.path.join(graph_dir, "fairness_audit.png")
    )

    # Save artifacts
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    pd.Series(y_test).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False)
    demo_test.to_csv(
        os.path.join(output_dir, "demo_test.csv"), index=False)
    with open(os.path.join(output_dir, "input_dim.txt"), "w") as f:
        f.write(str(X_tr_t.shape[1]))
    print("\nAll artifacts saved.")