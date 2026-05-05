import sys
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'metrics'))
from fairness_audit import FairnessAuditor


# diabetes CCI codes where higher score means sicker patient and weights from Quan et al. (2005)
CCI_MAP = [
    ('250',  1, 'diabetes'),
    ('2504', 2, 'diabetes_complicated'),
    ('2505', 2, 'diabetes_complicated'),
    ('2506', 2, 'diabetes_complicated'),
    ('2507', 2, 'diabetes_complicated'),
    ('2508', 2, 'diabetes_complicated'),
    ('2509', 2, 'diabetes_complicated'),
    ('585',  2, 'renal'),
    ('428',  1, 'heart_failure'),
    ('440',  1, 'peripheral_vascular'),
]


def compute_cci(row):
    # sum weights across all diagnosis columns and count each condition once
    score = 0
    seen  = set()
    for code in row:
        if pd.isna(code) or code == '?':
            continue
        code = str(code).replace('.', '').strip()
        for prefix, weight, name in CCI_MAP:
            if name not in seen and code.startswith(prefix):
                score += weight
                seen.add(name)
    return score


def load_data():
    print("Loading dataset...")
    raw = fetch_ucirepo(id=296)
    X   = raw.data.features.copy()
    y   = raw.data.targets.copy()

    y = (y['readmitted'] == '<30').astype(int).values

    # compute CCI before dropping diagnosis columns
    diag_cols      = [c for c in ['diag_1', 'diag_2', 'diag_3'] if c in X.columns]
    X['cci_score'] = X[diag_cols].apply(compute_cci, axis=1)
    print(f"CCI score -- mean: {X['cci_score'].mean():.2f}  max: {X['cci_score'].max()}")

    cols_to_drop = [
        'weight', 'payer_code', 'medical_specialty',
        'encounter_id', 'patient_nbr',
        'diag_1', 'diag_2', 'diag_3',
    ]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # remove patients who died or went to hospice
    if 'discharge_disposition_id' in X.columns:
        dead_codes = [11, 13, 14, 19, 20, 21]
        mask = ~X['discharge_disposition_id'].isin(dead_codes)
        X    = X[mask]
        y    = y[mask]

    X = X.replace('?', np.nan)

    missing_frac = X.isnull().mean()
    X = X.drop(columns=missing_frac[missing_frac > 0.4].index)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # drop high cardinality categoricals
    high_cardinality = [c for c in cat_cols if X[c].nunique() > 20]
    X        = X.drop(columns=high_cardinality)
    cat_cols = [c for c in cat_cols if c not in high_cardinality]

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
    X = pd.get_dummies(X, columns=list(cat_cols), drop_first=True)

    print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Positive class (readmitted <30d): {y.mean()*100:.1f}%")

    return X.values, y


def load_data_with_demographics():
    raw  = fetch_ucirepo(id=296)
    X    = raw.data.features.copy()
    y    = raw.data.targets.copy()
    y    = (y['readmitted'] == '<30').astype(int).values

    demo = X[['race', 'gender', 'age']].copy()

    if 'discharge_disposition_id' in X.columns:
        dead_codes = [11, 13, 14, 19, 20, 21]
        mask = ~X['discharge_disposition_id'].isin(dead_codes)
        demo = demo[mask].reset_index(drop=True)
        y    = y[mask]

    return demo, y


def make_data_loader(X, y, batch_size=256):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True
    )


class ReadmissionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)        # raw logit
        )

    def forward(self, x):
        return self.pipeline(x)

# training loop
def train(model, train_loader, pos_weight, k_epochs=30, lr=1e-3):
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    for epoch in range(k_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{k_epochs} — loss: {avg_loss:.4f}")

    return train_loss



def evaluate(model, X_test, y_test, threshold=0.5):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).squeeze().numpy()

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    sensitivity     = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv             = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc_roc         = roc_auc_score(y_test, probs)
    auc_pr          = average_precision_score(y_test, probs)
    baseline_auc_pr = average_precision_score(y_test, np.zeros_like(y_test))

    print(f"\n--- Evaluation (threshold = {threshold}) ---")
    print(f"  Sensitivity (recall): {sensitivity:.3f}")
    print(f"  PPV (precision):      {ppv:.3f}")
    print(f"  AUC-ROC:              {auc_roc:.3f}")
    print(f"  AUC-PR:               {auc_pr:.3f}  (baseline: {baseline_auc_pr:.3f})")
    print(f"  Confusion matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")

   
    for t in [0.3, 0.4]:
        p = (probs >= t).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        s = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        v = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
        print(f"\n  At threshold {t}: sensitivity={s:.3f}  PPV={v:.3f}  "
              f"TP={tp2}  FP={fp2}  FN={fn2}")

    return probs, {'sensitivity': sensitivity, 'ppv': ppv,
                   'auc_roc': auc_roc, 'auc_pr': auc_pr}


def plot_confusion_matrix(y_test, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No', 'Pred: Yes'])
    ax.set_yticklabels(['True: No', 'True: Yes'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Neural Net + CCI — Confusion Matrix (τ={threshold})')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('nn_confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: nn_confusion_matrix.png")



def threshold_sweep(probs, y_test):
    thresholds    = np.arange(0.05, 0.95, 0.01)
    sensitivities = []
    ppvs          = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            sensitivities.append(0)
            ppvs.append(0)
            continue
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sensitivities.append(tp / (tp + fn + 1e-9))
        ppvs.append(tp / (tp + fp + 1e-9))

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, sensitivities, label='Sensitivity (recall)', color='steelblue')
    plt.plot(thresholds, ppvs,          label='PPV (precision)',      color='coral')
    plt.axvline(0.5, linestyle='--', color='gray', alpha=0.5, label='default τ=0.5')
    plt.xlabel('Classification threshold τ')
    plt.ylabel('Score')
    plt.title('Neural Net + CCI — Sensitivity vs PPV')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nn_threshold_sweep.png', dpi=150)
    plt.show()
    print("Saved: nn_threshold_sweep.png")


def plot_calibration(probs, y_test):
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure(figsize=(6, 4))
    plt.plot(mean_pred, fraction_pos, marker='o', color='steelblue', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction actually readmitted')
    plt.title('Neural Net + CCI — Calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nn_calibration.png', dpi=150)
    plt.show()
    print("Saved: nn_calibration.png")


def plot_loss(train_loss):
    plt.figure(figsize=(7, 3))
    plt.plot(range(1, len(train_loss) + 1), train_loss,
             color='steelblue', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Avg batch loss')
    plt.title('Neural Net + CCI — training loss')
    plt.tight_layout()
    plt.savefig('nn_training_loss.png', dpi=150)
    plt.show()
    print("Saved: nn_training_loss.png")


def plot_fairness_audit(auditor):
    # plots AUROC by demographic subgroup and saves as png
    labels = [l for l in auditor.SUBGROUPS if any(
        l in auditor.results.get(m, {}) for m in auditor.models
    )]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(16, 12))
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        subgroups = list(dict.fromkeys(
            sg for m in auditor.models
            for sg in auditor.results.get(m, {}).get(label, pd.DataFrame()).index
        ))
        if not subgroups:
            continue

        x     = np.arange(len(subgroups))
        width = 0.8 / len(auditor.models)

        for i, (name, color) in enumerate(zip(auditor.models, colors)):
            df     = auditor.results.get(name, {}).get(label, pd.DataFrame())
            aurocs = [df.loc[sg, "auroc"] if sg in df.index else np.nan
                      for sg in subgroups]
            offset = (i - len(auditor.models) / 2 + 0.5) * width
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
    plt.savefig('nn_fairness_audit.png', dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: nn_fairness_audit.png")


if __name__ == '__main__':

    X, y = load_data()

    
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    train_loader = make_data_loader(X_train_t, y_train_t)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\npos_weight: {pos_weight:.2f}")

    model = ReadmissionNet(input_dim=X_train_t.shape[1])

    print("\nTraining...")
    train_loss = train(model, train_loader, pos_weight, k_epochs=30, lr=1e-3)

    probs, metrics = evaluate(model, X_test, y_test, threshold=0.5)

    plot_loss(train_loss)
    plot_confusion_matrix(y_test, probs, threshold=0.5)
    threshold_sweep(probs, y_test)
    plot_calibration(probs, y_test)

    # fairness audit
    print("\nRunning fairness audit...")
    demo, y_demo = load_data_with_demographics()
    demo_test    = demo.iloc[idx_test].reset_index(drop=True)

    auditor = FairnessAuditor(
        X_demo    = demo_test,
        y_test    = y_test,
        threshold = 0.5,
        gap       = 0.05
    )
    auditor.add_model("Neural Net + CCI", probs)
    auditor.run_audit()
    auditor.summary()
    plot_fairness_audit(auditor)

    # save model weights
    torch.save(model.state_dict(), 'neural_net_cci.pth')
    print("\nModel saved: neural_net_cci.pth")