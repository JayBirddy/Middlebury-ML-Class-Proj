# Trains the full pipeline and saves everything the standalone auditor requires:
# model.pth, scaler.pkl, feature_cols.pkl, input_dim.txt, demo_test.csv, y_test.csv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from models.neural_network import ReadmissionMLP

# set up output directory for training artifacts to be saved
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "nn")
os.makedirs(output_dir, exist_ok=True)

# data pipeline
print("Loading data...")
df = pd.read_csv("data/diabetic_data.csv", na_values=["?"])

# drop unusable columns
cols_to_drop = [
    "encounter_id", "patient_nbr", "weight",
    "payer_code", "medical_specialty", "examide", "citoglipton",
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# remove expired / hospice patients (can't be readmitted)
df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]

# binary target
df["target"] = (df["readmitted"] == "<30").astype(int)
df.drop(columns=["readmitted"], inplace=True)

# age ordinal
age_map = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}
df["age"] = df["age"].map(age_map)

# ICD-9 grouping
def icd9_group(code):
    if pd.isnull(code): return "Other"
    code = str(code)
    if code.startswith("V") or code.startswith("E"): return "External"
    try:
        c = float(code)
        if 390 <= c <= 459 or c == 785: return "Circulatory"
        if 460 <= c <= 519 or c == 786: return "Respiratory"
        if 520 <= c <= 579 or c == 787: return "Digestive"
        if c == 250:                    return "Diabetes"
        if 800 <= c <= 999:             return "Injury"
        if 710 <= c <= 739:             return "Musculoskeletal"
        if 580 <= c <= 629 or c == 788: return "Genitourinary"
        if 140 <= c <= 239:             return "Neoplasms"
        return "Other"
    except ValueError:
        return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = df[col].apply(icd9_group)

# drug features
drug_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone"
]
change_map = {"No": 0, "Steady": 0, "Up": 1, "Down": 1}
df["num_drug_changes"] = df[drug_cols].apply(
    lambda col: col.map(change_map).fillna(0)
).sum(axis=1)

drug_ordinal = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
for col in drug_cols:
    if col in df.columns:
        df[col] = df[col].map(drug_ordinal).fillna(0).astype(int)

# save demographic columns before one-hot encoding
demo_cols = ["race", "gender", "age"]
df_demo = df[demo_cols].copy()

# one-hot encode nominal categoricals
nominal_cats = [
    "race", "gender", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed"
]
df = pd.get_dummies(df, columns=nominal_cats, drop_first=False, dtype=int)
print(f"Dataset shape after encoding: {df.shape}")


# split and scale features
X = df.drop(columns=["target"]).reset_index(drop=True)
y = df["target"].reset_index(drop=True)
df_demo = df_demo.reset_index(drop=True)

X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
    X, y, df_demo,
    test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# save scaler and feature column order
with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Saved: scaler.pkl")

feature_cols = list(X_train.columns)
with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Saved: feature_cols.pkl  ({len(feature_cols)} features)")

# save demo_test for standalone auditor to load directly
demo_test.reset_index(drop=True).to_csv(os.path.join(output_dir, "demo_test.csv"), index=False)
y_test.reset_index(drop=True).to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
print("Saved: demo_test.csv, y_test.csv")


# train neural network
X_tr_t = torch.tensor(X_train_sc, dtype=torch.float32)
X_te_t = torch.tensor(X_test_sc,  dtype=torch.float32)
y_tr_t = torch.tensor(y_train.values, dtype=torch.float32)

pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
)
input_dim = X_train_sc.shape[1]
model     = ReadmissionMLP(input_dim)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True
)

n = 50

print("\nTraining...")
for epoch in range(n):
    model.train()
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(X_te_t)).numpy()
        print(f"  Epoch {epoch+1}/{n}  AUROC={roc_auc_score(y_test, p):.4f}")

model.eval()
with torch.no_grad():
    final_probs = torch.sigmoid(model(X_te_t)).numpy()
print(f"Final AUROC: {roc_auc_score(y_test, final_probs):.4f}")


# save model weights and input dimension for auditor
torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
print("Saved: model.pth")

with open(os.path.join(output_dir, "input_dim.txt"), "w") as f:
    f.write(str(input_dim))
print(f"Saved: input_dim.txt  (input_dim={input_dim})")