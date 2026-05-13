import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

df = pd.read_csv("data/diabetic_data.csv")
df.replace("?", pd.NA, inplace=True)
df.dropna(axis=1, thresh=int(0.6 * len(df)), inplace=True)
for col in df.columns:
    if df[col].isna().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns="readmitted").values
y = (df["readmitted"] != 0).astype(int).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
)

input_dim = X_train_t.shape[1]

model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 30
train_losses, val_losses, val_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    scheduler.step()

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
            f"Epoch {epoch+1:>3}/{EPOCHS} | "
            f"Train Loss: {train_losses[-1]:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {acc:.4f}"
        )

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, preds, target_names=["No Readmit", "Readmitted"]))

auc = roc_auc_score(y_test, probs)
print(f"ROC-AUC Score: {auc:.4f}")

torch.save(model.state_dict(), "neural_net_cci.pth")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses,   label="Val Loss")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend()

axes[1].plot(val_accs, color="green", label="Val Accuracy")
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Readmit", "Readmitted"])
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
