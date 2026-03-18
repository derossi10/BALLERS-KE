"""
utils.py — BALLERS-KE Dashboard Helper Functions
Handles data loading, metric computation, and scouting report generation.
Data source: players_ranked.csv (18,207 FIFA players with ML-predicted scores)
"""

import pandas as pd
import numpy as np
import os

# ── Path resolution ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RANKED_CSV = os.path.join(DATA_DIR, "players_ranked.csv")

# ── Tier colour palette (used across all pages) ────────────────────────────────
TIER_COLORS = {
    "Elite":      "#00E5A0",   # bright teal-green
    "Good":       "#3B9EFF",   # electric blue
    "Average":    "#FFB547",   # amber
    "Developing": "#FF6B6B",   # coral red
}

POSITION_COLORS = {
    "FWD": "#FF6B6B",
    "MID": "#3B9EFF",
    "DEF": "#00E5A0",
    "GK":  "#FFB547",
}

# Position-specific radar attributes
RADAR_ATTRS = {
    "FWD": {
        "labels":  ["Finishing","Dribbling","Pace","Shot Power","Ball Control",
                    "Positioning","Acceleration","Volleys"],
        "columns": ["Finishing","Dribbling","Pace_Score","ShotPower","BallControl",
                    "Positioning","Acceleration","Volleys"],
    },
    "MID": {
        "labels":  ["Short Pass","Vision","Long Pass","Dribbling","Ball Control",
                    "Stamina","Interceptions","Crossing"],
        "columns": ["ShortPassing","Vision","LongPassing","Dribbling","BallControl",
                    "Stamina","Interceptions","Crossing"],
    },
    "DEF": {
        "labels":  ["Standing Tackle","Sliding Tackle","Marking","Interceptions",
                    "Heading","Strength","Aggression","Reactions"],
        "columns": ["StandingTackle","SlidingTackle","Marking","Interceptions",
                    "HeadingAccuracy","Strength","Aggression","Reactions"],
    },
    "GK": {
        "labels":  ["Diving","Handling","Kicking","Reflexes","GK Positioning",
                    "Reactions","Jumping","Composure"],
        "columns": ["GKDiving","GKHandling","GKKicking","GKReflexes","GKPositioning",
                    "Reactions","Jumping","Composure"],
    },
}

# ── Data loading ───────────────────────────────────────────────────────────────
@pd.api.extensions.register_dataframe_accessor("bke")   # no-op, just for clarity
class _Noop: pass

def load_data() -> pd.DataFrame:
    """Load and lightly clean the ranked player dataset."""
    df = pd.read_csv(RANKED_CSV, encoding="utf-8-sig")

    # Ensure numeric types
    num_cols = [
        "Age","Overall","Potential","Performance_Score","Predicted_Score",
        "Finishing","Dribbling","ShotPower","BallControl","Positioning",
        "Acceleration","SprintSpeed","Volleys","ShortPassing","Vision",
        "LongPassing","Stamina","Interceptions","Crossing","HeadingAccuracy",
        "Strength","Aggression","Reactions","StandingTackle","SlidingTackle",
        "Marking","GKDiving","GKHandling","GKKicking","GKReflexes",
        "GKPositioning","Jumping","Composure","Balance","Agility",
        "Overall_FWD_Score","Overall_MID_Score","Overall_DEF_Score","Overall_GK_Score",
        "Finishing_Efficiency","xG_Proxy","Shot_Accuracy_Index","Goal_Contribution_Rate",
        "Pace_Score","Passing_Index","Vision_Score","Dribble_Contribution","PCI",
        "Work_Rate_Index","Defensive_Duel_Index","Tackle_Success_Rate","Aerial_Dominance",
        "Marking_Intensity","Reflex_Score","Handling_Index","Distribution_Score",
        "GK_Positioning_Score",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive Pace_Score if missing
    if "Pace_Score" not in df.columns:
        df["Pace_Score"] = (df["Acceleration"].fillna(0) + df["SprintSpeed"].fillna(0)) / 2

    # Clean tier ordering
    tier_order = ["Developing", "Average", "Good", "Elite"]
    for tcol in ["Performance_Tier", "Predicted_Tier"]:
        if tcol in df.columns:
            df[tcol] = pd.Categorical(df[tcol], categories=tier_order, ordered=True)

    return df


def get_position_score(row: pd.Series) -> float:
    """Return the position-specific score for a single player row."""
    pos = row.get("PositionGroup", "")
    col_map = {
        "FWD": "Overall_FWD_Score",
        "MID": "Overall_MID_Score",
        "DEF": "Overall_DEF_Score",
        "GK":  "Overall_GK_Score",
    }
    col = col_map.get(pos, "Performance_Score")
    val = row.get(col, np.nan)
    return val if pd.notna(val) else row.get("Performance_Score", 0)


def enrich_with_position_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a unified 'Score' column using the position-specific model score."""
    df = df.copy()
    df["Score"] = df.apply(get_position_score, axis=1)
    df["Score"] = df["Score"].fillna(df["Performance_Score"])
    return df


def format_value(val) -> str:
    """Convert raw value (e.g. 1.1e8) to human-readable string (e.g. €110M)."""
    try:
        v = float(val)
        if v >= 1_000_000:
            return f"€{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"€{v/1_000:.0f}K"
        return f"€{v:.0f}"
    except Exception:
        return str(val)


# ── Radar data builder ─────────────────────────────────────────────────────────
def get_radar_values(row: pd.Series, position: str) -> tuple[list[str], list[float]]:
    """Return (labels, values) for a radar chart given a player row."""
    config = RADAR_ATTRS.get(position, RADAR_ATTRS["MID"])
    labels  = config["labels"]
    columns = config["columns"]
    values  = [float(row.get(c, 0) or 0) for c in columns]
    return labels, values


# ── Scouting report generator ──────────────────────────────────────────────────
def generate_scouting_report(row: pd.Series) -> dict:
    """
    Produce a structured scouting report dict with strengths, weaknesses,
    and a recommendation for the given player.
    """
    name     = row.get("Name", "Unknown")
    pos      = row.get("PositionGroup", "MID")
    age      = int(row.get("Age", 0) or 0)
    overall  = int(row.get("Overall", 0) or 0)
    potential= int(row.get("Potential", 0) or 0)
    score    = float(row.get("Predicted_Score", 0) or 0)
    tier     = row.get("Predicted_Tier", "Average")
    club     = row.get("Club", "Unknown")
    nation   = row.get("Nationality", "Unknown")
    foot     = row.get("Preferred Foot", "Right")
    work     = row.get("Work Rate", "Medium/ Medium")
    growth   = potential - overall

    # ── Strengths ─────────────────────────────────────────────────────────────
    strengths = []

    if pos == "FWD":
        if (row.get("Finishing") or 0) >= 80:
            strengths.append(f"Clinical finisher — Finishing rating of {int(row.get('Finishing',0))} places him among the elite strikers.")
        if (row.get("Pace_Score") or 0) >= 80:
            strengths.append(f"Exceptional pace ({int(row.get('Pace_Score',0))}) makes him a constant threat in behind defensive lines.")
        if (row.get("Dribbling") or 0) >= 82:
            strengths.append(f"Outstanding dribbling ability ({int(row.get('Dribbling',0))}) enables effective ball progression in tight spaces.")
        if (row.get("Finishing_Efficiency") or 0) >= 75:
            strengths.append("High Finishing Efficiency index — converts chances at an above-average rate relative to xG.")
        if (row.get("Goal_Contribution_Rate") or 0) >= 70:
            strengths.append("Strong goal contribution rate indicating consistent direct involvement in attacking play.")

    elif pos == "MID":
        if (row.get("ShortPassing") or 0) >= 82:
            strengths.append(f"Elite short passing ({int(row.get('ShortPassing',0))}) — excellent ball recycler and link-up player.")
        if (row.get("Vision") or 0) >= 82:
            strengths.append(f"High vision ({int(row.get('Vision',0))}) — consistently finds runners and opens defensive blocks.")
        if (row.get("LongPassing") or 0) >= 80:
            strengths.append(f"Strong long-range distribution ({int(row.get('LongPassing',0))}) to switch play and release forwards.")
        if (row.get("PCI") or 0) >= 75:
            strengths.append("High Playmaker Contribution Index — a creative hub who drives attacking sequences.")
        if (row.get("Stamina") or 0) >= 80:
            strengths.append(f"Excellent engine with Stamina of {int(row.get('Stamina',0))} — covers the pitch effectively for the full 90.")

    elif pos == "DEF":
        if (row.get("StandingTackle") or 0) >= 80:
            strengths.append(f"Commanding tackler ({int(row.get('StandingTackle',0))}) — wins duels cleanly and reduces opponent transitions.")
        if (row.get("Marking") or 0) >= 78:
            strengths.append(f"Disciplined marker ({int(row.get('Marking',0))}) who limits space for opposing forwards.")
        if (row.get("HeadingAccuracy") or 0) >= 78:
            strengths.append(f"Aerially dominant with heading accuracy of {int(row.get('HeadingAccuracy',0))} — reliable at set-pieces.")
        if (row.get("Defensive_Duel_Index") or 0) >= 70:
            strengths.append("High Defensive Duel Index — wins a significant proportion of 1v1 challenges.")
        if (row.get("ShortPassing") or 0) >= 75:
            strengths.append("Comfortable on the ball — contributes to build-up play from deep positions.")

    elif pos == "GK":
        if (row.get("GKReflexes") or 0) >= 80:
            strengths.append(f"Lightning reflexes ({int(row.get('GKReflexes',0))}) — capable of saving shots in close-range situations.")
        if (row.get("GKDiving") or 0) >= 78:
            strengths.append(f"Good diving ability ({int(row.get('GKDiving',0))}) — covers the corners of the goal effectively.")
        if (row.get("GKHandling") or 0) >= 78:
            strengths.append(f"Safe hands ({int(row.get('GKHandling',0))}) — commanding in catching crosses and set-pieces.")
        if (row.get("GKPositioning") or 0) >= 78:
            strengths.append("Strong positional sense reduces angles for opposition shooters.")
        if (row.get("GKKicking") or 0) >= 72:
            strengths.append("Reliable distribution with the ball at feet — supports team's build-up play.")

    if not strengths:
        strengths.append(f"Solid performer at {tier} tier with a predicted score of {score:.1f}.")

    # ── Weaknesses ────────────────────────────────────────────────────────────
    weaknesses = []

    if pos == "FWD":
        if (row.get("Finishing") or 0) < 65:
            weaknesses.append("Below-average finishing — needs improvement in composure inside the penalty area.")
        if (row.get("HeadingAccuracy") or 0) < 55:
            weaknesses.append("Limited aerial threat — struggles to contribute to set-piece attacking plays.")
        if (row.get("Strength") or 0) < 55:
            weaknesses.append("Lacks physical presence — can be bullied off the ball by strong defenders.")
    elif pos == "MID":
        if (row.get("Interceptions") or 0) < 55:
            weaknesses.append("Defensive contribution is limited — doesn't consistently win the ball back.")
        if (row.get("LongShots") or 0) < 55:
            weaknesses.append("Lacks long-range shooting threat, which limits tactical unpredictability.")
        if (row.get("Aggression") or 0) < 50:
            weaknesses.append("Low aggression may result in being overrun in physical midfield battles.")
    elif pos == "DEF":
        if (row.get("Pace_Score") or 0) < 55:
            weaknesses.append("Pace limitations make him vulnerable to quick forwards in one-on-one situations.")
        if (row.get("ShortPassing") or 0) < 60:
            weaknesses.append("Distribution under pressure needs work — can give the ball away cheaply.")
        if (row.get("Agility") or 0) < 55:
            weaknesses.append("Limited agility reduces effectiveness when tracking lateral movement.")
    elif pos == "GK":
        if (row.get("GKKicking") or 0) < 55:
            weaknesses.append("Distribution with the feet is below standard for modern goalkeeping demands.")
        if (row.get("GKHandling") or 0) < 65:
            weaknesses.append("Handling concerns — can be uncertain when claiming crosses into the box.")
        if (row.get("Composure") or 0) < 60:
            weaknesses.append("Composure under pressure could be an issue in high-stakes situations.")

    if not weaknesses:
        weaknesses.append("No major weaknesses identified at this performance level — well-rounded profile.")

    # ── Recommendation ────────────────────────────────────────────────────────
    if tier == "Elite":
        rec_level = "IMMEDIATE SIGNING TARGET"
        rec_text  = (
            f"{name} is an Elite-tier {pos} with a predicted performance score of {score:.1f}/100. "
            f"At {age} years old {'with ' + str(growth) + ' points of potential growth remaining' if growth > 0 else '— at peak performance'}, "
            f"this player represents exceptional value. Highly recommended for immediate acquisition."
        )
    elif tier == "Good":
        rec_level = "STRONG RECOMMENDATION"
        rec_text  = (
            f"{name} is a Good-tier {pos} (score: {score:.1f}/100) who offers reliable quality. "
            f"{'At ' + str(age) + ' with room to develop further, this is a sound medium-term investment.' if age < 27 else 'An experienced, consistent performer who can strengthen the squad immediately.'}"
        )
    elif tier == "Average":
        rec_level = "CONDITIONAL INTEREST"
        rec_text  = (
            f"{name} is a solid Average-tier {pos} (score: {score:.1f}/100). "
            f"{'Young enough to develop with the right coaching environment.' if age < 23 else 'A viable squad option depending on budget and positional needs.'} "
            f"Further evaluation recommended before committing to a transfer."
        )
    else:  # Developing
        rec_level = "MONITOR & DEVELOP"
        rec_text  = (
            f"{name} is currently in the Developing tier (score: {score:.1f}/100). "
            f"{'At {age}, high upside if placed in a strong development environment.' if age < 21 else 'Requires significant improvement before competing at a higher level.'} "
            f"Could be a low-cost development option."
        )

    return {
        "name":         name,
        "club":         club,
        "nationality":  nation,
        "position":     pos,
        "age":          age,
        "preferred_foot": foot,
        "work_rate":    work,
        "overall":      overall,
        "potential":    potential,
        "predicted_score": round(score, 1),
        "tier":         tier,
        "strengths":    strengths,
        "weaknesses":   weaknesses,
        "rec_level":    rec_level,
        "rec_text":     rec_text,
    }


# ── Key stats table builder ────────────────────────────────────────────────────
def get_key_stats(row: pd.Series) -> pd.DataFrame:
    """Return a two-column DataFrame of key stats for a player profile card."""
    pos = row.get("PositionGroup", "MID")

    base = {
        "Overall Rating": int(row.get("Overall", 0) or 0),
        "Potential":      int(row.get("Potential", 0) or 0),
        "Reactions":      int(row.get("Reactions", 0) or 0),
        "Composure":      int(row.get("Composure", 0) or 0),
        "Stamina":        int(row.get("Stamina", 0) or 0),
        "Strength":       int(row.get("Strength", 0) or 0),
        "Aggression":     int(row.get("Aggression", 0) or 0),
        "Balance":        int(row.get("Balance", 0) or 0),
    }

    pos_stats = {
        "FWD": {
            "Finishing":       int(row.get("Finishing", 0) or 0),
            "Dribbling":       int(row.get("Dribbling", 0) or 0),
            "Pace Score":      round(float(row.get("Pace_Score", 0) or 0), 1),
            "Shot Power":      int(row.get("ShotPower", 0) or 0),
            "Ball Control":    int(row.get("BallControl", 0) or 0),
            "Positioning":     int(row.get("Positioning", 0) or 0),
            "xG Proxy":        round(float(row.get("xG_Proxy", 0) or 0), 1),
            "Goal Contrib Rate": round(float(row.get("Goal_Contribution_Rate", 0) or 0), 1),
        },
        "MID": {
            "Short Passing":   int(row.get("ShortPassing", 0) or 0),
            "Vision":          int(row.get("Vision", 0) or 0),
            "Long Passing":    int(row.get("LongPassing", 0) or 0),
            "Dribbling":       int(row.get("Dribbling", 0) or 0),
            "Interceptions":   int(row.get("Interceptions", 0) or 0),
            "Crossing":        int(row.get("Crossing", 0) or 0),
            "PCI":             round(float(row.get("PCI", 0) or 0), 1),
            "Work Rate Index": round(float(row.get("Work_Rate_Index", 0) or 0), 1),
        },
        "DEF": {
            "Standing Tackle": int(row.get("StandingTackle", 0) or 0),
            "Sliding Tackle":  int(row.get("SlidingTackle", 0) or 0),
            "Marking":         int(row.get("Marking", 0) or 0),
            "Interceptions":   int(row.get("Interceptions", 0) or 0),
            "Heading Acc.":    int(row.get("HeadingAccuracy", 0) or 0),
            "Defensive Duel Index": round(float(row.get("Defensive_Duel_Index", 0) or 0), 1),
            "Tackle Success":  round(float(row.get("Tackle_Success_Rate", 0) or 0), 1),
            "Aerial Dominance": round(float(row.get("Aerial_Dominance", 0) or 0), 1),
        },
        "GK": {
            "GK Diving":       int(row.get("GKDiving", 0) or 0),
            "GK Handling":     int(row.get("GKHandling", 0) or 0),
            "GK Kicking":      int(row.get("GKKicking", 0) or 0),
            "GK Reflexes":     int(row.get("GKReflexes", 0) or 0),
            "GK Positioning":  int(row.get("GKPositioning", 0) or 0),
            "Reflex Score":    round(float(row.get("Reflex_Score", 0) or 0), 1),
            "Handling Index":  round(float(row.get("Handling_Index", 0) or 0), 1),
            "Distribution":    round(float(row.get("Distribution_Score", 0) or 0), 1),
        },
    }

    merged = {**base, **pos_stats.get(pos, {})}
    return pd.DataFrame({"Attribute": list(merged.keys()), "Value": list(merged.values())})
