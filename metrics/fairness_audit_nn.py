# equity analysis through metric analysis on test set predictions and the dataset's original demographic column 
# compute AUROC, recall, precision, and positive prediction rate broken out by each demographic subgroup 
# flag where performance gaps exceed a threshold (default: 0.5) as well as a gap threshold for AUROC (default 0.05)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir   = os.path.join(PROJECT_ROOT, "outputs", "nn")
os.makedirs(output_dir, exist_ok=True)

import argparse
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from models.neural_network import ReadmissionMLP
from fairness_audit import FairnessAuditor

# argument parsing
parser = argparse.ArgumentParser(
    description="Standalone fairness audit for readmission prediction model."
)
parser.add_argument(
    "--model",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "model.pth"),
    help="Path to the .pth weights file (default: model.pth)"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Decision threshold for binary predictions (default: 0.5)"
)
parser.add_argument(
    "--gap",
    type=float,
    default=0.05,
    help="AUROC gap threshold to flag as a disparity (default: 0.05)"
)
parser.add_argument(
    "--scaler",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "scaler.pkl"),
    help="Path to the scaler pickle file (default: scaler.pkl)"
)
parser.add_argument(
    "--features",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "feature_cols.pkl"),
    help="Path to the feature columns pickle file (default: feature_cols.pkl)"
)
parser.add_argument(
    "--demo",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "demo_test.csv"),
    help="Path to demographic test split CSV (default: demo_test.csv)"
)
parser.add_argument(
    "--labels",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "y_test.csv"),
    help="Path to ground truth labels CSV (default: y_test.csv)"
)
parser.add_argument(
    "--input_dim",
    default=os.path.join(PROJECT_ROOT, "outputs", "nn", "input_dim.txt"),
    help="Path to ground truth labels CSV (default: input_dim.txt)"
)
args = parser.parse_args()


# load all saved artifacts
def load_artifact(path, loader_fn, label):
    """helper: load a file and exit clearly if it's missing."""
    try:
        return loader_fn(path)
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find {label} at '{path}'.")
        print("        Run train_and_save.py first to generate all artifacts.")
        sys.exit(1)


print("Loading saved artifacts...")

# feature column list
feature_cols = load_artifact(
    args.features,
    lambda p: pickle.load(open(p, "rb")),
    "feature_cols.pkl"
)
print(f"  Feature columns loaded: {len(feature_cols)} features")

# scaler
scaler = load_artifact(
    args.scaler,
    lambda p: pickle.load(open(p, "rb")),
    "scaler.pkl"
)
print("Scaler loaded")

# input dimension
input_dim = load_artifact(
    args.input_dim,
    lambda p: int(open(p).read().strip()),
    "input_dim.txt"
)
print(f"Input dim: {input_dim}")

# model weights
# instantiate the architecture first, then load weights into it.
# map_location="cpu" ensures weights load even if the model was trained on GPU
model = ReadmissionMLP(input_dim)
state = load_artifact(
    args.model,
    lambda p: torch.load(p, map_location="cpu"),
    args.model
)
model.load_state_dict(state)
model.eval()   # important: disables Dropout and sets BatchNorm to eval mode.
               # Without this, predictions are stochastic and wrong.
print(f"  Model weights loaded from '{args.model}'")

# demographic test split
demo_test = load_artifact(
    args.demo,
    lambda p: pd.read_csv(p),
    "demo_test.csv"
)
print(f"  Demo test loaded: {demo_test.shape}")

# ground truth labels
y_test = load_artifact(
    args.labels,
    lambda p: pd.read_csv(p).squeeze(),   # squeeze turns single-col DF into Series
    "y_test.csv"
)
print(f"  Labels loaded: {len(y_test)} rows, positive rate={y_test.mean():.3f}")

# reconstruct features for the test set
print("\nRebuilding test feature matrix from raw data...")

df_raw = pd.read_csv("data/diabetic_data.csv", na_values=["?"])

# apply identical preprocessing steps
cols_to_drop = [
    "encounter_id", "patient_nbr", "weight",
    "payer_code", "medical_specialty", "examide", "citoglipton",
]
df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns], inplace=True)
df_raw = df_raw[~df_raw["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
df_raw["target"] = (df_raw["readmitted"] == "<30").astype(int)
df_raw.drop(columns=["readmitted"], inplace=True)

age_map = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}
df_raw["age"] = df_raw["age"].map(age_map)

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
    df_raw[col] = df_raw[col].apply(icd9_group)

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
df_raw["num_drug_changes"] = df_raw[drug_cols].apply(
    lambda col: col.map(change_map).fillna(0)
).sum(axis=1)
drug_ordinal = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
for col in drug_cols:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].map(drug_ordinal).fillna(0).astype(int)

nominal_cats = [
    "race", "gender", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed"
]
df_raw = pd.get_dummies(df_raw, columns=nominal_cats, drop_first=False, dtype=int)

# reconstruct the exact test split
# use the same random_state=42 and stratify to reproduce the identical split.
from sklearn.model_selection import train_test_split

X_full = df_raw.drop(columns=["target"]).reset_index(drop=True)
y_full = df_raw["target"].reset_index(drop=True)

_, X_test_raw, _, _ = train_test_split(
    X_full, y_full,
    test_size=0.2, random_state=42, stratify=y_full
)

# align columns to training feature list
X_test_aligned = X_test_raw.reindex(columns=feature_cols, fill_value=0)

# apply the saved scaler (transform only — never fit)
X_test_sc = scaler.transform(X_test_aligned)
print(f"  Test feature matrix: {X_test_sc.shape}")


# run inference
X_te_t = torch.tensor(X_test_sc, dtype=torch.float32)

with torch.no_grad():   # no gradient computation needed at inference time
    logits    = model(X_te_t)
    nn_probs  = torch.sigmoid(logits).numpy()

from sklearn.metrics import roc_auc_score
print(f"\nModel AUROC on test set: {roc_auc_score(y_test, nn_probs):.4f}")


print(f"\nRunning fairness audit (threshold={args.threshold}, gap={args.gap})...")

auditor = FairnessAuditor(
    X_demo=demo_test.reset_index(drop=True),
    y_test=y_test.reset_index(drop=True),
    threshold=args.threshold,
    gap=args.gap,
)

auditor.add_model("Neural Network", nn_probs)

auditor.run_audit()
auditor.summary()
auditor.plot_audit()