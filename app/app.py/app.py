
import streamlit as st
import sys
import os

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="BALLERS-KE | Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark analytics theme ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background-color: #0D1117; color: #E6EDF3; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #8B949E !important;
    font-size: 0.9rem;
    padding: 6px 0;
    cursor: pointer;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #E6EDF3 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #00E5A0; }
[data-testid="metric-container"] label { color: #8B949E !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-size: 1.9rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #00E5A0 !important; }

/* ── Dataframes ── */
.stDataFrame { border: 1px solid #21262D; border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] thead th {
    background: #161B22 !important;
    color: #8B949E !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stDataFrame"] tbody tr:hover td { background: #21262D !important; }

/* ── Buttons & selectbox ── */
.stSelectbox > div > div { background: #161B22 !important; border-color: #21262D !important; color: #E6EDF3 !important; border-radius: 8px !important; }
.stButton > button { background: #238636 !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 8px 20px !important; transition: background 0.2s !important; }
.stButton > button:hover { background: #2EA043 !important; }

/* ── Slider ── */
.stSlider > div > div > div > div { background: #00E5A0 !important; }

/* ── Section divider ── */
hr { border-color: #21262D !important; }

/* ── Custom card class ── */
.bke-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.bke-card:hover { border-color: #30363D; }

/* ── Badge ── */
.badge-elite    { background:#00E5A020; color:#00E5A0; border:1px solid #00E5A0; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.badge-good     { background:#3B9EFF20; color:#3B9EFF; border:1px solid #3B9EFF; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.badge-average  { background:#FFB54720; color:#FFB547; border:1px solid #FFB547; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.badge-developing { background:#FF6B6B20; color:#FF6B6B; border:1px solid #FF6B6B; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/* ── Page title ── */
.page-title { font-size:1.7rem; font-weight:700; color:#E6EDF3; margin-bottom:4px; }
.page-subtitle { font-size:0.9rem; color:#8B949E; margin-bottom:24px; }
</style>
""", unsafe_allow_html=True)

# ── Import page modules ────────────────────────────────────────────────────────
from pages import (
    home,
    player_rankings,
    player_profile,
    player_comparison,
    team_analysis,
    talent_discovery,
    scouting_report,
    model_insights,
)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.2rem;">⚽</div>
        <div style="font-size:1.3rem; font-weight:800; color:#E6EDF3; letter-spacing:0.04em;">BALLERS-KE</div>
        <div style="font-size:0.72rem; color:#8B949E; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">
            Football Analytics Platform
        </div>
    </div>
    <hr style="border-color:#21262D; margin-bottom:16px;">
    """, unsafe_allow_html=True)

    PAGE_OPTIONS = {
        "🏠  Home":                 "Home",
        "📊  Player Rankings":      "Player Rankings",
        "👤  Player Profile":       "Player Profile",
        "⚔️   Player Comparison":   "Player Comparison",
        "🏟️   Team Analysis":       "Team Analysis",
        "🌟  Talent Discovery":     "Talent Discovery",
        "📋  AI Scouting Report":   "AI Scouting Report",
        "🤖  Model Insights":       "Model Insights",
    }

    selected_label = st.radio(
        "Navigation",
        list(PAGE_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    selected_page = PAGE_OPTIONS[selected_label]

    st.markdown("""
    <hr style="border-color:#21262D; margin-top:20px;">
    <div style="font-size:0.7rem; color:#484F58; text-align:center; padding-top:8px;">
        BALLERS-KE v1.0 · TUK Final Year Project<br>
        Dataset: FIFA 19 · 18,207 players
    </div>
    """, unsafe_allow_html=True)

# ── Route to page ──────────────────────────────────────────────────────────────
if selected_page == "Home":
    home.render()
elif selected_page == "Player Rankings":
    player_rankings.render()
elif selected_page == "Player Profile":
    player_profile.render()
elif selected_page == "Player Comparison":
    player_comparison.render()
elif selected_page == "Team Analysis":
    team_analysis.render()
elif selected_page == "Talent Discovery":
    talent_discovery.render()
elif selected_page == "AI Scouting Report":
    scouting_report.render()
elif selected_page == "Model Insights":
    model_insights.render()
