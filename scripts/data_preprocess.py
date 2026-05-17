import numpy as np
import pandas as pd

# CCI weights from Quan et al. (2011)
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

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}

# high-percentage missing cols. DROPPED
HIGH_MISSING_COLS = [
    'weight',             # 97% missing
    'max_glu_serum',      # 95% missing
    'A1Cresult',          # 83% missing
    'medical_specialty',  # 49% missing
    'payer_code',         # 40% missing
]

# ID columns don't help predict readmission rate
ID_COLS = ['encounter_id', 'patient_nbr']

# Zero-variance drug columns — same value for every row; doesn't help with prediction
ZERO_VARIANCE_COLS = ['citoglipton', 'examide']

# Diagnosis columns — dropped after CCI is computed from them
DIAG_COLS = ['diag_1', 'diag_2', 'diag_3']

# Discharge codes where readmission is impossible
DEAD_CODES = [11, 13, 14, 19, 20, 21]


# feature engineering: compute Charlson Comorbidity Index (CCI) score from diagnosis codes
def compute_cci(row):
    """Sum CCI weights across diagnosis columns, counting each condition once."""
    score, seen = 0, set()
    for code in row:
        if pd.isna(code) or code == '?':
            continue
        code = str(code).replace('.', '').strip()
        for prefix, weight, name in CCI_MAP:
            if name not in seen and code.startswith(prefix):
                score += weight
                seen.add(name)
    return score


def add_cci(X):
    """Compute CCI score from diagnosis columns and add as a new feature."""
    diag_cols      = [c for c in DIAG_COLS if c in X.columns]
    X['cci_score'] = X[diag_cols].apply(compute_cci, axis=1)
    print(f"CCI score -- mean: {X['cci_score'].mean():.2f}  max: {X['cci_score'].max()}")
    return X


# demographics
def extract_demographics(X):
    """
    Save race, gender, age before encoding.
    Age mapped to ordinal int so FairnessAuditor.AGE_LABELS works correctly.
    """
    demo_cols = [c for c in ['race', 'gender', 'age'] if c in X.columns]
    demo      = X[demo_cols].copy()
    if 'age' in demo.columns:
        demo['age'] = demo['age'].map(AGE_MAP)
    return demo


# cleaning
def replace_question_marks(X):
    """Replace '?' placeholder with NaN so pandas missing-value tools work."""
    return X.replace('?', np.nan)


def drop_all_diag_missing(X, y, demo):
    """
    ADDED: Drop rows where ALL THREE diagnosis columns are simultaneously missing.
    These rows have no diagnostic information at all. Rows with at least one
    valid diagnosis are retained — more surgical than dropping the whole column.
    """
    if not all(c in X.columns for c in DIAG_COLS):
        return X, y, demo

    all_missing = (
        (X['diag_1'] == '?') | X['diag_1'].isna()
    ) & (
        (X['diag_2'] == '?') | X['diag_2'].isna()
    ) & (
        (X['diag_3'] == '?') | X['diag_3'].isna()
    )
    mask  = ~all_missing
    n_dropped = all_missing.sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with all three diagnosis columns missing.")
    return (
        X[mask].reset_index(drop=True),
        y[mask],
        demo[mask].reset_index(drop=True)
    )


def drop_columns(X):
    """
    Drop high-missing columns (explicitly named), ID columns,
    zero-variance columns, and raw diagnosis columns.
    """
    to_drop = HIGH_MISSING_COLS + ID_COLS + ZERO_VARIANCE_COLS + DIAG_COLS
    dropped = [c for c in to_drop if c in X.columns]
    print(f"Dropping columns: {dropped}")
    return X.drop(columns=dropped)


def remove_nonreadmit_patients(X, y, demo):
    """
    Remove patients who cannot be readmitted:
    expired, hospice, or long-term care discharge codes.
    """
    if 'discharge_disposition_id' not in X.columns:
        return X, y, demo
    mask = ~X['discharge_disposition_id'].isin(DEAD_CODES)
    n_dropped = (~mask).sum()
    print(f"Dropped {n_dropped} non-readmittable patient encounters.")
    return (
        X[mask].reset_index(drop=True),
        y[mask],
        demo[mask].reset_index(drop=True)
    )


def drop_invalid_demographics(X, y, demo):
    """
    ADDED: Drop rows with invalid/unknown gender or race.
    These columns feed directly into the fairness audit — imputing unknown
    group membership would corrupt subgroup-level metric computation.
    """
    mask = pd.Series([True] * len(X), index=X.index)

    if 'gender' in X.columns:
        invalid_gender = X['gender'] == 'Unknown/Invalid'
        n = invalid_gender.sum()
        if n > 0:
            print(f"Dropped {n} rows with Unknown/Invalid gender.")
        mask = mask & ~invalid_gender

    if 'race' in X.columns:
        invalid_race = (X['race'] == '?') | X['race'].isna()
        n = invalid_race.sum()
        if n > 0:
            print(f"Dropped {n} rows with unknown race.")
        mask = mask & ~invalid_race

    return (
        X[mask].reset_index(drop=True),
        y[mask],
        demo[mask].reset_index(drop=True)
    )

def impute_and_encode(X):
    """
    Drop remaining high-missing columns (>40% threshold catches anything
    not already named above), impute, drop high-cardinality categoricals,
    and one-hot encode.
    """
    # Generic threshold drop for anything not already explicitly handled
    missing_frac = X.isnull().mean()
    remaining_high_missing = missing_frac[missing_frac > 0.4].index.tolist()
    if remaining_high_missing:
        print(f"Threshold-dropping additional high-missing cols: {remaining_high_missing}")
        X = X.drop(columns=remaining_high_missing)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # Drop high-cardinality categoricals (>20 unique values)
    high_card = [c for c in cat_cols if X[c].nunique() > 20]
    if high_card:
        print(f"Dropping high-cardinality categoricals: {high_card}")
    X        = X.drop(columns=high_card)
    cat_cols = [c for c in cat_cols if c not in high_card]

    # Impute
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # One-hot encode
    X = pd.get_dummies(X, columns=list(cat_cols), drop_first=True)

    return X