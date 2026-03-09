"""
BALLERS-KE | Day 2 — Feature Engineering (Fixed)
==================================================
Computes position-specific advanced metrics IN-PLACE on the full dataframe
so every player has their correct metrics filled and no NaN misalignment.

Position groups & metrics built
────────────────────────────────
FWD  → Finishing_Efficiency, xG_Proxy, Shot_Accuracy_Index,
        Goal_Contribution_Rate, Pace_Score, Overall_FWD_Score

MID  → Passing_Index, Vision_Score, Dribble_Contribution,
        PCI, Work_Rate_Index, Overall_MID_Score

DEF  → Defensive_Duel_Index, Tackle_Success_Rate,
        Aerial_Dominance, Marking_Intensity, Overall_DEF_Score

GK   → Reflex_Score, Handling_Index, Distribution_Score,
        GK_Positioning_Score, Overall_GK_Score

ALL  → Performance_Score (0-100) + Performance_Tier

Fix applied
───────────
Features are now written back into the ORIGINAL dataframe using positional
masks (df.loc[mask, col] = ...) instead of building separate subsets and
concatenating. This means each player only has their own position's columns
populated — other positions' columns are intentionally NaN, which is correct
and clean. The Performance_Score column is the single universal metric.

Output
──────
data/processed/players_features.csv
"""

import pandas as pd
import numpy as np
import os

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
INPUT_PATH  = "data/processed/players_clean.csv"
OUTPUT_PATH = "data/processed/players_features.csv"


# ───────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────
def scale100(series: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 100]. Flat series returns 50."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn)) * 100


def weighted_avg(df: pd.DataFrame, cols: list, weights: list) -> pd.Series:
    """Weighted average of already-scaled columns. Weights auto-normalised."""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return sum(df[c] * w[i] for i, c in enumerate(cols))


# ───────────────────────────────────────────────
# LOAD
# ───────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[load]  {df.shape[0]:,} players x {df.shape[1]} columns")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "PositionGroup" not in df.columns:
        raise ValueError("'PositionGroup' column missing. Run Day 1 cleaning first.")

    # Pre-create all feature columns as NaN so the CSV is clean
    all_feature_cols = [
        # FWD
        "Finishing_Efficiency", "xG_Proxy", "Shot_Accuracy_Index",
        "Goal_Contribution_Rate", "Pace_Score", "Overall_FWD_Score",
        # MID
        "Passing_Index", "Vision_Score", "Dribble_Contribution",
        "PCI", "Work_Rate_Index", "Overall_MID_Score",
        # DEF
        "Defensive_Duel_Index", "Tackle_Success_Rate",
        "Aerial_Dominance", "Marking_Intensity", "Overall_DEF_Score",
        # GK
        "Reflex_Score", "Handling_Index", "Distribution_Score",
        "GK_Positioning_Score", "Overall_GK_Score",
        # Universal
        "Performance_Score", "Performance_Tier"
    ]
    for col in all_feature_cols:
        df[col] = np.nan

    return df


# ───────────────────────────────────────────────
# FWD FEATURES  (written in-place via mask)
# ───────────────────────────────────────────────
def build_fwd_features(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["PositionGroup"] == "FWD"
    f = df.loc[mask].copy()   # local copy to compute on

    f["Finishing_Efficiency"] = scale100(
        f["Finishing"] * 0.40 + f["Composure"] * 0.25 +
        f["Reactions"] * 0.20 + f["Volleys"]   * 0.15
    )
    f["xG_Proxy"] = scale100(
        f["Positioning"] * 0.45 + f["ShotPower"] * 0.30 +
        f["LongShots"]   * 0.25
    )
    f["Shot_Accuracy_Index"] = scale100(
        f["Finishing"] * 0.50 + f["Curve"] * 0.25 +
        f["FKAccuracy"] * 0.25
    )
    f["Goal_Contribution_Rate"] = scale100(
        f["Dribbling"] * 0.35 + f["BallControl"] * 0.35 +
        f["Vision"]    * 0.30
    )
    f["Pace_Score"] = scale100(
        f["Acceleration"] * 0.50 + f["SprintSpeed"] * 0.50
    )
    f["Overall_FWD_Score"] = weighted_avg(
        f,
        cols    = ["Finishing_Efficiency", "xG_Proxy", "Shot_Accuracy_Index",
                   "Goal_Contribution_Rate", "Pace_Score"],
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # Write back into the main dataframe
    fwd_feat_cols = ["Finishing_Efficiency", "xG_Proxy", "Shot_Accuracy_Index",
                     "Goal_Contribution_Rate", "Pace_Score", "Overall_FWD_Score"]
    df.loc[mask, fwd_feat_cols] = f[fwd_feat_cols].values

    print(f"[FWD]   {mask.sum():,} players | 6 features written")
    return df


# ───────────────────────────────────────────────
# MID FEATURES
# ───────────────────────────────────────────────
def build_mid_features(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["PositionGroup"] == "MID"
    f = df.loc[mask].copy()

    f["Passing_Index"] = scale100(
        f["ShortPassing"] * 0.40 + f["LongPassing"] * 0.35 +
        f["Crossing"]     * 0.25
    )
    f["Vision_Score"] = scale100(
        f["Vision"] * 0.50 + f["BallControl"] * 0.30 +
        f["Reactions"] * 0.20
    )
    f["Dribble_Contribution"] = scale100(
        f["Dribbling"] * 0.55 + f["Agility"] * 0.25 +
        f["Balance"]   * 0.20
    )
    f["PCI"] = scale100(
        f["Passing_Index"]        * 0.30 + f["Vision_Score"]         * 0.25 +
        f["Dribble_Contribution"] * 0.20 + f["Interceptions"]        * 0.15 +
        f["Stamina"]              * 0.10
    )
    f["Work_Rate_Index"] = scale100(
        f["Stamina"] * 0.45 + f["Aggression"] * 0.30 +
        f["Interceptions"] * 0.25
    )
    f["Overall_MID_Score"] = weighted_avg(
        f,
        cols    = ["Passing_Index", "Vision_Score", "Dribble_Contribution",
                   "PCI", "Work_Rate_Index"],
        weights = [0.25, 0.25, 0.20, 0.20, 0.10]
    )

    mid_feat_cols = ["Passing_Index", "Vision_Score", "Dribble_Contribution",
                     "PCI", "Work_Rate_Index", "Overall_MID_Score"]
    df.loc[mask, mid_feat_cols] = f[mid_feat_cols].values

    print(f"[MID]   {mask.sum():,} players | 6 features written")
    return df


# ───────────────────────────────────────────────
# DEF FEATURES
# ───────────────────────────────────────────────
def build_def_features(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["PositionGroup"] == "DEF"
    f = df.loc[mask].copy()

    f["Defensive_Duel_Index"] = scale100(
        f["StandingTackle"] * 0.40 + f["Strength"]   * 0.30 +
        f["Aggression"]     * 0.30
    )
    f["Tackle_Success_Rate"] = scale100(
        f["StandingTackle"] * 0.50 + f["SlidingTackle"] * 0.35 +
        f["Reactions"]      * 0.15
    )
    f["Aerial_Dominance"] = scale100(
        f["HeadingAccuracy"] * 0.50 + f["Jumping"]  * 0.35 +
        f["Strength"]        * 0.15
    )
    f["Marking_Intensity"] = scale100(
        f["Marking"] * 0.45 + f["Interceptions"] * 0.35 +
        f["Composure"] * 0.20
    )
    f["Overall_DEF_Score"] = weighted_avg(
        f,
        cols    = ["Defensive_Duel_Index", "Tackle_Success_Rate",
                   "Aerial_Dominance", "Marking_Intensity"],
        weights = [0.30, 0.30, 0.20, 0.20]
    )

    def_feat_cols = ["Defensive_Duel_Index", "Tackle_Success_Rate",
                     "Aerial_Dominance", "Marking_Intensity", "Overall_DEF_Score"]
    df.loc[mask, def_feat_cols] = f[def_feat_cols].values

    print(f"[DEF]   {mask.sum():,} players | 5 features written")
    return df


# ───────────────────────────────────────────────
# GK FEATURES
# ───────────────────────────────────────────────
def build_gk_features(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["PositionGroup"] == "GK"
    f = df.loc[mask].copy()

    f["Reflex_Score"] = scale100(
        f["GKReflexes"] * 0.55 + f["GKDiving"]  * 0.30 +
        f["Reactions"]  * 0.15
    )
    f["Handling_Index"] = scale100(
        f["GKHandling"] * 0.60 + f["GKPositioning"] * 0.25 +
        f["Composure"]  * 0.15
    )
    f["Distribution_Score"] = scale100(
        f["GKKicking"] * 0.55 + f["ShortPassing"] * 0.25 +
        f["Vision"]    * 0.20
    )
    f["GK_Positioning_Score"] = scale100(
        f["GKPositioning"] * 0.65 + f["Reactions"] * 0.20 +
        f["Composure"]     * 0.15
    )
    f["Overall_GK_Score"] = weighted_avg(
        f,
        cols    = ["Reflex_Score", "Handling_Index",
                   "Distribution_Score", "GK_Positioning_Score"],
        weights = [0.35, 0.30, 0.15, 0.20]
    )

    gk_feat_cols = ["Reflex_Score", "Handling_Index", "Distribution_Score",
                    "GK_Positioning_Score", "Overall_GK_Score"]
    df.loc[mask, gk_feat_cols] = f[gk_feat_cols].values

    print(f"[GK]    {mask.sum():,} players | 5 features written")
    return df


# ───────────────────────────────────────────────
# COMPOSITE SCORE + TIER
# ───────────────────────────────────────────────
def build_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    score_map = {
        "FWD": "Overall_FWD_Score",
        "MID": "Overall_MID_Score",
        "DEF": "Overall_DEF_Score",
        "GK":  "Overall_GK_Score",
    }
    for pos, col in score_map.items():
        mask = df["PositionGroup"] == pos
        df.loc[mask, "Performance_Score"] = df.loc[mask, col]

    # Fallback for any unmatched rows
    fallback = df["Performance_Score"].isna()
    if fallback.any() and "Overall" in df.columns:
        df.loc[fallback, "Performance_Score"] = scale100(df.loc[fallback, "Overall"])

    df["Performance_Tier"] = pd.cut(
        df["Performance_Score"],
        bins=[0, 40, 60, 80, 100],
        labels=["Developing", "Average", "Good", "Elite"],
        include_lowest=True
    )

    print(f"[ALL]   Performance_Score + Performance_Tier assigned to all players")
    return df


# ───────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────
def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH) -> pd.DataFrame:
    print("=" * 52)
    print("  BALLERS-KE | Feature Engineering Pipeline")
    print("=" * 52)

    df = load_data(input_path)
    df = build_fwd_features(df)
    df = build_mid_features(df)
    df = build_def_features(df)
    df = build_gk_features(df)
    df = build_composite_score(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[done]  Saved -> {output_path}")
    print(f"        Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 52)
    return df


# ───────────────────────────────────────────────
# SUMMARY REPORT
# ───────────────────────────────────────────────
def summary_report(df: pd.DataFrame):
    print("\n-- Performance Tier Distribution --")
    print(df["Performance_Tier"].value_counts().sort_index())

    print("\n-- Top 10 Players by Performance Score --")
    cols = ["Name", "Club", "Position", "PositionGroup",
            "Overall", "Performance_Score", "Performance_Tier"]
    available = [c for c in cols if c in df.columns]
    print(df.nlargest(10, "Performance_Score")[available].to_string(index=False))

    print("\n-- Position Score Averages --")
    for pos, col in [("FWD","Overall_FWD_Score"), ("MID","Overall_MID_Score"),
                     ("DEF","Overall_DEF_Score"), ("GK","Overall_GK_Score")]:
        sub = df[df["PositionGroup"] == pos][col].dropna()
        print(f"  {pos}  mean={sub.mean():.1f}  max={sub.max():.1f}  min={sub.min():.1f}")

    print("\n-- NaN check on feature columns --")
    feat_cols = [c for c in df.columns if c in [
        "Finishing_Efficiency","xG_Proxy","Shot_Accuracy_Index","Goal_Contribution_Rate",
        "Pace_Score","Overall_FWD_Score","Passing_Index","Vision_Score",
        "Dribble_Contribution","PCI","Work_Rate_Index","Overall_MID_Score",
        "Defensive_Duel_Index","Tackle_Success_Rate","Aerial_Dominance",
        "Marking_Intensity","Overall_DEF_Score","Reflex_Score","Handling_Index",
        "Distribution_Score","GK_Positioning_Score","Overall_GK_Score","Performance_Score"
    ]]
    nan_counts = df[feat_cols].isnull().sum()
    expected_nans = {
        "Finishing_Efficiency": len(df[df["PositionGroup"] != "FWD"]),
        "Passing_Index":        len(df[df["PositionGroup"] != "MID"]),
        "Defensive_Duel_Index": len(df[df["PositionGroup"] != "DEF"]),
        "Reflex_Score":         len(df[df["PositionGroup"] != "GK"]),
    }
    print(f"  Performance_Score NaNs : {df['Performance_Score'].isna().sum()} (should be 0)")
    print(f"  FWD metrics NaN for non-FWD players  : {nan_counts.get('Finishing_Efficiency',0):,} "
          f"(expected {expected_nans['Finishing_Efficiency']:,}) ✓")
    print(f"  MID metrics NaN for non-MID players  : {nan_counts.get('Passing_Index',0):,} "
          f"(expected {expected_nans['Passing_Index']:,}) ✓")
    print(f"  DEF metrics NaN for non-DEF players  : {nan_counts.get('Defensive_Duel_Index',0):,} "
          f"(expected {expected_nans['Defensive_Duel_Index']:,}) ✓")
    print(f"  GK  metrics NaN for non-GK  players  : {nan_counts.get('Reflex_Score',0):,} "
          f"(expected {expected_nans['Reflex_Score']:,}) ✓")


if __name__ == "__main__":
    df_out = run(
        input_path  = "data/processed/players_clean.csv",
        output_path = "data/processed/players_features.csv"
    )
    summary_report(df_out)