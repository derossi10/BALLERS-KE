"""
=============================================================================
BALLERS-KE | src/models/train_ranking.py
=============================================================================
AI-Based Soccer Player Performance Analytics and Scouting System
Technical University of Kenya — SCIT Project B

PURPOSE
-------
Trains 8 position-specific ML models (XGBoost Regressor + Random Forest
Classifier for each of FWD, MID, DEF, GK).  Evaluates every model against
the proposal thresholds, saves all .pkl files, writes players_ranked.csv
with Predicted_Score and Predicted_Tier columns, and prints a top-10 summary
per position.

MODELS TRAINED (8 total)
    FWD — XGBoost Regressor  |  Random Forest Classifier
    MID — XGBoost Regressor  |  Random Forest Classifier
    DEF — XGBoost Regressor  |  Random Forest Classifier
    GK  — XGBoost Regressor  |  Random Forest Classifier

PROPOSAL THRESHOLDS
    Accuracy  >= 85%   (classification tier assignment)
    Spearman  >= 0.80  (ranking correlation vs. actual Performance_Score)
    MAE       <= 10    (mean absolute error of predicted score)
    RMSE      <= 12    (root mean squared error of predicted score)

USAGE
-----
From the project root directory run:
    python src/models/train_ranking.py

REQUIREMENTS
    pip install xgboost scikit-learn scipy joblib pandas numpy
=============================================================================
"""

# =============================================================================
# STEP 1 - IMPORTS
# =============================================================================
# Standard library
import os
import warnings
import time

# Data handling
import pandas as pd
import numpy as np
import joblib

# scikit-learn: model selection and cross-validation
from sklearn.model_selection import KFold, StratifiedKFold

# scikit-learn: the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# scikit-learn: evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

# scikit-learn: label encoding for the tier strings
from sklearn.preprocessing import LabelEncoder

# XGBoost - primary gradient-boosted regressor (proposal spec)
# XGBoost is chosen because it handles missing values natively, is fast on
# tabular data, and consistently outperforms plain decision trees.
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    # Graceful fallback so the script still runs on machines where xgboost
    # is not yet installed.  On your project machine: pip install xgboost
    from sklearn.ensemble import GradientBoostingRegressor as _GBR

    class XGBRegressor(_GBR):
        """Thin wrapper that keeps the same API as the real XGBRegressor."""
        def __init__(self, **kwargs):
            n_estimators = kwargs.pop("n_estimators", 300)
            max_depth     = kwargs.pop("max_depth",     5)
            learning_rate = kwargs.pop("learning_rate", 0.05)
            subsample      = kwargs.pop("subsample",      0.8)
            kwargs.pop("colsample_bytree", None)
            kwargs.pop("reg_alpha",        None)
            kwargs.pop("reg_lambda",       None)
            kwargs.pop("random_state",     None)
            super().__init__(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
            )

    XGBOOST_AVAILABLE = False
    warnings.warn(
        "\n[BALLERS-KE] xgboost not installed - using sklearn "
        "GradientBoostingRegressor as a temporary fallback.\n"
        "Install the real XGBoost with:  pip install xgboost\n",
        stacklevel=2,
    )

# scipy: Spearman rank correlation
# Spearman measures whether the RANK ORDER of players is preserved -
# more meaningful than Pearson for scouting where rank matters most.
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# =============================================================================
# STEP 2 - CONFIGURATION
# =============================================================================

DATA_PATH   = "data/processed/players_features.csv"
OUTPUT_PATH = "data/processed/players_ranked.csv"
MODEL_DIR   = "models"

# Proposal performance thresholds - used to print PASS / FAIL next to metrics
THRESHOLDS = {
    "accuracy": 0.85,   # >= 85%  classification accuracy
    "spearman": 0.80,   # >= 0.80 Spearman rho
    "mae":      10.0,   # <= 10   mean absolute error
    "rmse":     12.0,   # <= 12   root mean squared error
}

N_FOLDS = 5    # 5-fold cross-validation as stated in the proposal
SEED    = 42   # random seed for reproducibility

# Tier ordering from lowest to highest performance
TIER_ORDER = ["Developing", "Average", "Good", "Elite"]


# =============================================================================
# STEP 3 - POSITION-SPECIFIC FEATURE COLUMN MAPPING
# =============================================================================
# Each position group ONLY sees the features relevant to that role.
# Feeding GK reflexes into a striker model would introduce noise and hurt
# accuracy.  Position-stratified models avoid this problem entirely.

POSITION_FEATURES = {
    # FORWARDS - attacking output: goal-scoring efficiency, shot quality, speed
    "FWD": [
        "Finishing_Efficiency",    # Goals / shots on target ratio (scaled 0-100)
        "xG_Proxy",                # Expected goals derived from shot quality
        "Shot_Accuracy_Index",     # On-target shots as % of total shots
        "Goal_Contribution_Rate",  # (Goals + Assists) per 90 min normalised
        "Pace_Score",              # Composite of Acceleration + SprintSpeed
    ],

    # MIDFIELDERS - ball distribution, creativity, carrying ability, work rate
    "MID": [
        "Passing_Index",           # Short + long passing composite
        "Vision_Score",            # Key passes, through-balls, chances created
        "Dribble_Contribution",    # Successful dribbles and take-ons per 90
        "PCI",                     # Progressive Carry Index - advances ball fwd
        "Work_Rate_Index",         # Stamina, pressing intensity, distance covered
    ],

    # DEFENDERS - defensive duelling, tackling, aerial play, man-marking
    "DEF": [
        "Defensive_Duel_Index",    # Ground duel success rate
        "Tackle_Success_Rate",     # Standing + sliding tackle accuracy
        "Aerial_Dominance",        # Headers won / total aerial duels
        "Marking_Intensity",       # Interceptions + blocks per 90 composite
    ],

    # GOALKEEPERS - shot-stopping, handling, distribution, positioning
    "GK": [
        "Reflex_Score",            # Reaction saves - GKReflexes based
        "Handling_Index",          # Claim success rate - GKHandling based
        "Distribution_Score",      # GKKicking + short distribution
        "GK_Positioning_Score",    # Positional awareness - GKPositioning based
    ],
}

REGRESSION_TARGET    = "Performance_Score"    # continuous 0-100
CLASSIFICATION_TARGET = "Performance_Tier"   # Developing/Average/Good/Elite


# =============================================================================
# STEP 4 - HELPER: threshold check flag
# =============================================================================

def _flag(value, threshold, higher_is_better=True):
    """Return a short PASS/FAIL string based on threshold direction."""
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return "PASS" if ok else "FAIL"


# =============================================================================
# STEP 5 - HELPER: load and validate the dataset
# =============================================================================

def load_data(path):
    """
    Load players_features.csv and perform basic sanity checks.

    Steps
    -----
    1. Read CSV into a DataFrame.
    2. Confirm all required columns exist.
    3. Drop rows where targets are missing (defensive programming).
    4. Report per-position row counts.
    """
    print("=" * 70)
    print("BALLERS-KE | Player Ranking Model Training")
    print("=" * 70)
    print(f"\n[1/6] Loading data from:  {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Cannot find '{path}'.\n"
            "Run this script from the PROJECT ROOT:\n"
            "    python src/models/train_ranking.py\n"
        )

    df = pd.read_csv(path)

    # Verify all required columns are present
    required = (
        ["PositionGroup", REGRESSION_TARGET, CLASSIFICATION_TARGET]
        + [col for cols in POSITION_FEATURES.values() for col in cols]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in CSV: {missing}")

    before = len(df)
    df = df.dropna(subset=[REGRESSION_TARGET, CLASSIFICATION_TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"  Warning: dropped {dropped} rows with missing targets.")

    print(f"  Dataset shape  : {df.shape[0]:,} players x {df.shape[1]} columns")
    print("  Position counts:")
    for pos, n in df["PositionGroup"].value_counts().items():
        print(f"    {pos:<6} {n:>5,} players")

    return df


# =============================================================================
# STEP 6 - HELPER: prepare one position group's X and y arrays
# =============================================================================

def prepare_position_data(df, position):
    """
    Filter the DataFrame to one position and return clean feature/target arrays.

    Steps
    -----
    1. Select rows where PositionGroup == position.
    2. Extract position-specific feature columns (X).
    3. Extract Performance_Score (y_reg) and encoded Performance_Tier (y_clf).
    4. Impute any remaining NaN values with column medians.
    5. Return X, y_reg, y_clf, the fitted LabelEncoder, and the row indices.

    Why LabelEncoder?
    -----------------
    Random Forest needs integer class labels.  We fit the encoder in the fixed
    TIER_ORDER so that class IDs are stable and human-readable in all output:
        Developing=0  Average=1  Good=2  Elite=3
    """
    subset   = df[df["PositionGroup"] == position].copy()
    features = POSITION_FEATURES[position]

    # Feature matrix
    X = subset[features].copy()

    # Regression target
    y_reg = subset[REGRESSION_TARGET].values

    # Classification target - encode tier strings as integers
    le = LabelEncoder()
    le.fit(TIER_ORDER)
    y_clf = le.transform(subset[CLASSIFICATION_TARGET].values)

    # Impute NaN values with column medians (affects ~48 FWD rows)
    X = X.fillna(X.median(numeric_only=True))

    return X, y_reg, y_clf, le, subset.index.tolist()


# =============================================================================
# STEP 7 - HELPER: build the XGBoost Regressor
# =============================================================================

def build_regressor():
    """
    Instantiate the XGBoost Regressor with hyperparameters that:
      - Prevent overfitting on smaller position subsets (max_depth=5,
        subsample=0.8, colsample_bytree=0.8)
      - Apply L1+L2 regularisation to keep leaf weights small
      - Run in reasonable time on a standard laptop (n_estimators=300)

    How XGBoost works (briefly)
    ---------------------------
    It builds an ensemble of shallow decision trees where each new tree
    corrects the residual errors of the previous trees (gradient boosting).
    The output is a continuous predicted score in [0, 100].
    """
    return XGBRegressor(
        n_estimators=300,       # number of boosting rounds / trees
        max_depth=5,            # max tree depth - controls model complexity
        learning_rate=0.05,     # shrinkage factor applied to each tree (eta)
        subsample=0.8,          # fraction of training rows sampled per tree
        colsample_bytree=0.8,   # fraction of features sampled per tree
        reg_alpha=0.1,          # L1 regularisation on leaf weights (sparsity)
        reg_lambda=1.0,         # L2 regularisation on leaf weights (smoothness)
        random_state=SEED,
    )


# =============================================================================
# STEP 8 - HELPER: build the Random Forest Classifier
# =============================================================================

def build_classifier():
    """
    Instantiate the Random Forest Classifier.

    How Random Forest works (briefly)
    ----------------------------------
    Grows many independent decision trees on random feature/row subsets and
    takes a majority vote across all trees.  It is robust to overfitting and
    handles class imbalance well via class_weight='balanced'.

    class_weight='balanced'
    -----------------------
    The tier distribution is heavily skewed (Average >> Elite).  Balanced
    weighting makes the model penalise misclassification of rare classes
    (Elite, Developing) more heavily - which is exactly what scouts care about.
    """
    return RandomForestClassifier(
        n_estimators=300,          # number of trees in the forest
        max_depth=10,              # cap tree depth to prevent overfitting
        min_samples_leaf=5,        # each leaf must contain >= 5 samples
        class_weight="balanced",   # compensate for tier class imbalance
        n_jobs=-1,                 # use all available CPU cores
        random_state=SEED,
    )


# =============================================================================
# STEP 9 - CORE: train and evaluate both models for one position group
# =============================================================================

def train_position(df, position, model_dir):
    """
    Full training pipeline for one position group.  Returns a results dict.

    Detailed steps
    --------------
    A) Prepare data          - filter, impute, encode labels
    B) Set CV strategy       - KFold for regressor (continuous target),
                               StratifiedKFold for classifier (preserves
                               tier proportions in each fold)
    C) Cross-validate        - collect fold-by-fold metrics
    D) Final fit             - retrain on 100% of position data so the saved
                               model has seen every available player
    E) Compute final metrics - MAE, RMSE, Spearman, Accuracy
    F) Save .pkl files       - regressor, classifier, label encoder
    G) Return metrics dict   - used to build the evaluation summary table

    Train/test split note
    ---------------------
    We use cross-validation rather than a fixed 80/20 split.  This gives a
    more reliable error estimate on our modest position subset sizes.
    The final model is then re-fit on 100% of position data (no data wasted).
    """
    print(f"\n{'─'*60}")
    print(f"  POSITION GROUP: {position}")
    print(f"{'─'*60}")

    # A) Prepare data
    X, y_reg, y_clf, le, idx = prepare_position_data(df, position)
    n        = len(X)
    features = POSITION_FEATURES[position]
    print(f"  Players in group : {n:,}")
    print(f"  Feature columns  : {features}")

    # B) Cross-validation strategies
    # Standard KFold for regression (no need to stratify a continuous target)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # StratifiedKFold for classification ensures each fold has roughly the
    # same proportion of tiers - critical given the heavy class imbalance
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # C-1) Cross-validate the XGBoost Regressor
    print(f"\n  [Regressor - XGBoost]  {N_FOLDS}-fold CV ...")
    reg = build_regressor()
    t0  = time.time()

    fold_maes, fold_rmses, fold_spearmans = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr,  X_val  = X.iloc[tr_idx],   X.iloc[val_idx]
        y_tr,  y_val  = y_reg[tr_idx],    y_reg[val_idx]

        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_val)

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        rho, _ = spearmanr(y_val, y_pred)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_spearmans.append(rho)
        print(f"    Fold {fold}: MAE={mae:.3f}  RMSE={rmse:.3f}  Spearman={rho:.4f}")

    mean_mae      = float(np.mean(fold_maes))
    mean_rmse     = float(np.mean(fold_rmses))
    mean_spearman = float(np.mean(fold_spearmans))
    print(f"  -- CV averages --")
    print(f"    MAE      = {mean_mae:.3f}   [{_flag(mean_mae,  THRESHOLDS['mae'],  higher_is_better=False)}]")
    print(f"    RMSE     = {mean_rmse:.3f}   [{_flag(mean_rmse, THRESHOLDS['rmse'], higher_is_better=False)}]")
    print(f"    Spearman = {mean_spearman:.4f}   [{_flag(mean_spearman, THRESHOLDS['spearman'])}]")
    print(f"    Time     = {time.time()-t0:.1f}s")

    # C-2) Cross-validate the Random Forest Classifier
    print(f"\n  [Classifier - Random Forest]  {N_FOLDS}-fold stratified CV ...")
    clf = build_classifier()
    t0  = time.time()

    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_clf), 1):
        X_tr,  X_val  = X.iloc[tr_idx],   X.iloc[val_idx]
        y_tr,  y_val  = y_clf[tr_idx],    y_clf[val_idx]

        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, clf.predict(X_val))
        fold_accs.append(acc)
        print(f"    Fold {fold}: Accuracy={acc:.4f}")

    mean_acc = float(np.mean(fold_accs))
    print(f"  -- CV averages --")
    print(f"    Accuracy = {mean_acc:.4f}   [{_flag(mean_acc, THRESHOLDS['accuracy'])}]")
    print(f"    Time     = {time.time()-t0:.1f}s")

    # D) Final fit on ALL position data (no data wasted)
    print(f"\n  [Final fit on full {position} subset ({n:,} players)] ...")
    reg.fit(X, y_reg)
    clf.fit(X, y_clf)

    # Predictions for the output CSV
    predicted_scores = reg.predict(X)
    predicted_tiers  = le.inverse_transform(clf.predict(X))

    # F) Save model .pkl files
    os.makedirs(model_dir, exist_ok=True)
    reg_path = os.path.join(model_dir, f"{position}_xgb_regressor.pkl")
    clf_path = os.path.join(model_dir, f"{position}_rf_classifier.pkl")
    le_path  = os.path.join(model_dir, f"{position}_label_encoder.pkl")

    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)
    joblib.dump(le,  le_path)

    print(f"  Saved: {reg_path}")
    print(f"  Saved: {clf_path}")
    print(f"  Saved: {le_path}")

    # G) Return all results
    return {
        "position":          position,
        "n_players":         n,
        "mean_mae":          mean_mae,
        "mean_rmse":         mean_rmse,
        "mean_spearman":     mean_spearman,
        "mean_accuracy":     mean_acc,
        "reg_path":          reg_path,
        "clf_path":          clf_path,
        "predicted_scores":  predicted_scores,
        "predicted_tiers":   predicted_tiers,
    }


# =============================================================================
# STEP 10 - EVALUATION SUMMARY TABLE
# =============================================================================

def print_evaluation_table(results):
    """
    Print a consolidated table comparing all 4 positions against the
    proposal thresholds.  Easy to copy into your Chapter 6 test plan.
    """
    xgb_label = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting (fallback)"

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION SUMMARY  (proposal thresholds in brackets)")
    print("=" * 70)
    print(
        f"  {'Position':<8}  {'Players':>7}  "
        f"{'MAE(<=10)':>10}  {'RMSE(<=12)':>11}  "
        f"{'Spearman(>=0.80)':>17}  {'Accuracy(>=0.85)':>17}"
    )
    print("  " + "-" * 74)

    for r in results:
        print(
            f"  {r['position']:<8}  {r['n_players']:>7,}  "
            f"  {r['mean_mae']:>6.3f} [{_flag(r['mean_mae'],      THRESHOLDS['mae'],      higher_is_better=False):<4}]  "
            f"  {r['mean_rmse']:>6.3f} [{_flag(r['mean_rmse'],     THRESHOLDS['rmse'],     higher_is_better=False):<4}]  "
            f"  {r['mean_spearman']:>7.4f} [{_flag(r['mean_spearman'], THRESHOLDS['spearman']):<4}]  "
            f"  {r['mean_accuracy']:>7.4f} [{_flag(r['mean_accuracy'], THRESHOLDS['accuracy']):<4}]"
        )

    print()
    print(f"  Regressor  : {xgb_label}")
    print(f"  Classifier : Random Forest  (class_weight='balanced')")
    print(f"  CV folds   : {N_FOLDS}-fold  (Stratified for classifier)")
    print(f"  Seed       : {SEED}")


# =============================================================================
# STEP 11 - TOP-10 SUMMARY REPORT per position
# =============================================================================

def summary_report(df_ranked):
    """
    Print the top-10 ranked players per position sorted by Predicted_Score.
    This directly supports the scouting use-case described in the proposal -
    coaches and scouts can instantly see who the system ranks highest.
    """
    print("\n" + "=" * 70)
    print("  TOP-10 PLAYERS PER POSITION  (sorted by Predicted_Score desc)")
    print("=" * 70)

    display_cols = []
    for col in ["Name", "Club", "Age", "Predicted_Score",
                "Predicted_Tier", "Performance_Score"]:
        if col in df_ranked.columns:
            display_cols.append(col)

    for position in ["FWD", "MID", "DEF", "GK"]:
        subset = (
            df_ranked[df_ranked["PositionGroup"] == position]
            .sort_values("Predicted_Score", ascending=False)
            .head(10)
            .copy()
            .reset_index(drop=True)
        )
        subset.index += 1  # rank starts at 1

        # Round numeric columns
        for col in ["Predicted_Score", "Performance_Score"]:
            if col in subset.columns:
                subset[col] = subset[col].round(2)

        print(f"\n  -- {position} " + "-" * 50)
        print(subset[display_cols].to_string())


# =============================================================================
# STEP 12 - MAIN PIPELINE
# =============================================================================

def main():
    """
    Orchestrates the full training pipeline:

    Step 1  Load data from players_features.csv
    Step 2  Create output directories (models/, data/processed/)
    Step 3  Train + evaluate all 4 position groups (8 models total)
    Step 4  Write players_ranked.csv with Predicted_Score & Predicted_Tier
    Step 5  Print evaluation summary table
    Step 6  Print top-10 player rankings per position
    """

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\n[2/6] Model directory : {MODEL_DIR}/")
    print(f"      Output CSV      : {OUTPUT_PATH}")

    # 3. Initialise new output columns (will be filled per-position below)
    df["Predicted_Score"] = np.nan
    df["Predicted_Tier"]  = ""

    # 4. Train all four position groups
    print("\n[3/6] Training position-specific models ...")
    all_results = []

    for position in ["FWD", "MID", "DEF", "GK"]:
        result = train_position(df, position, MODEL_DIR)
        all_results.append(result)

        # Write predictions back into the full DataFrame at the correct rows
        pos_mask = df["PositionGroup"] == position
        df.loc[pos_mask, "Predicted_Score"] = result["predicted_scores"]
        df.loc[pos_mask, "Predicted_Tier"]  = result["predicted_tiers"]

    # 5. Save ranked CSV
    print(f"\n[4/6] Saving ranked CSV -> {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  {len(df):,} players saved with Predicted_Score & Predicted_Tier")
    print(f"  Predicted_Score range: "
          f"[{df['Predicted_Score'].min():.2f}, {df['Predicted_Score'].max():.2f}]")
    print("  Predicted_Tier counts:")
    for tier in TIER_ORDER:
        print(f"    {tier:<12} {(df['Predicted_Tier']==tier).sum():>5,}")

    # 6. Evaluation table
    print("\n[5/6] Evaluation summary ...")
    print_evaluation_table(all_results)

    # 7. Top-10 report
    print("\n[6/6] Generating top-10 rankings ...")
    summary_report(df)

    print("\n" + "=" * 70)
    print("  BALLERS-KE training complete.")
    print(f"  Models saved  : {MODEL_DIR}/")
    print(f"  Ranked CSV    : {OUTPUT_PATH}")
    print("=" * 70 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()