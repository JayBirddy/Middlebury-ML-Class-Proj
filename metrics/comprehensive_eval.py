import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from models.neural_network import ReadmissionMLP
from metrics.fairness_audit import FairnessAuditor
from metrics.threshold_analysis import ThresholdAnalyzer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load all models ──────────────────────────────────────────────────────────
def load_nn():
    input_dim = int(open(os.path.join(PROJECT_ROOT, "outputs/nn/input_dim.txt")).read())
    model = ReadmissionMLP(input_dim)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "outputs/nn/model.pth"), map_location="cpu"
    ))
    model.eval()
    scaler       = pickle.load(open(os.path.join(PROJECT_ROOT, "outputs/nn/scaler.pkl"), "rb"))
    feature_cols = pickle.load(open(os.path.join(PROJECT_ROOT, "outputs/nn/feature_cols.pkl"), "rb"))
    return model, scaler, feature_cols

def load_logreg():
    model        = pickle.load(open(os.path.join(PROJECT_ROOT, "outputs/sto_logreg/model.pkl"), "rb"))
    scaler       = pickle.load(open(os.path.join(PROJECT_ROOT, "outputs/sto_logreg/scaler.pkl"), "rb"))
    feature_cols = pickle.load(open(os.path.join(PROJECT_ROOT, "outputs/sto_logreg/feature_cols.pkl"), "rb"))
    return model, scaler, feature_cols

# ── Load shared test data ─────────────────────────────────────────────────────
# Both models were trained on the same split so they share the same test set
demo_test = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs/nn/demo_test.csv"))
y_test    = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs/nn/y_test.csv")).squeeze()

# ── Get probabilities from each model ────────────────────────────────────────
nn_model, nn_scaler, nn_cols       = load_nn()
lr_model, lr_scaler, lr_cols       = load_logreg()

# ... reconstruct X_test, scale, run inference ...
# nn_probs = torch.sigmoid(nn_model(X_te_t)).numpy()
# lr_probs = lr_model.predict_proba(X_test_sc)[:, 1]

# ── Run all metrics ───────────────────────────────────────────────────────────
auditor = FairnessAuditor(demo_test, y_test)
auditor.add_model("Neural Network",  nn_probs)
auditor.add_model("Stochastic LR",   lr_probs)
auditor.run()
auditor.summary()
auditor.plot()

analyzer = ThresholdAnalyzer(y_test)
analyzer.add_model("Neural Network", nn_probs)
analyzer.add_model("Stochastic LR",  lr_probs)
analyzer.run_threshold_sweep(beta=2.0)
analyzer.plot_threshold_sweep()
analyzer.plot_calibration()
analyzer.plot_clinical_cost(fn_cost_ratio=5)
analyzer.optimal_thresholds_report(fn_cost_ratio=5)