import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from ucimlrepo import fetch_ucirepo

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


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


# data loader
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
            nn.Linear(64, 1)        \
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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc         = roc_auc_score(y_test, probs)

    print(f"\n--- Evaluation (threshold = {threshold}) ---")
    print(f"  Sensitivity (recall): {sensitivity:.3f}")
    print(f"  PPV (precision):      {ppv:.3f}")
    print(f"  AUC-ROC:              {auc:.3f}")
    print(f"  Confusion matrix:")
    print(f"    TN={tn:6d}  FP={fp:6d}")
    print(f"    FN={fn:6d}  TP={tp:6d}")

    return probs, {'sensitivity': sensitivity, 'ppv': ppv, 'auc': auc}


if __name__ == '__main__':

    X, y = load_data()

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

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

    torch.save(model.state_dict(), 'neural_net_cci.pth')
    print("\nModel saved: neural_net_cci.pth")