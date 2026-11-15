"""
Neurovia - Prodromal Health Monitoring
Patient-facing Streamlit prototype using Isolation Forest.

Important:
- This is an experimental monitoring prototype, not a diagnostic tool.
- Isolation Forest scores are anomaly scores, not disease probabilities.
- The model expects features in this exact order:
  sleep_duration, sleep_quality, mood, fatigue, tremor
"""

import base64
import calendar
import html
import inspect
import pickle
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Neurovia | Health Monitor",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "submissions.db"
MODEL_PATH = "isolation_forest_model.pkl"
SCALER_PATH = "feature_scaler.pkl"

FEATURE_COLUMNS = [
    "sleep_duration",
    "sleep_quality",
    "mood",
    "fatigue",
    "tremor"
]

FEATURE_LABELS = {
    "sleep_duration": "Sleep Problems",
    "sleep_quality": "Daytime Sleepiness",
    "mood": "Mood & Anxiety",
    "fatigue": "Fatigue",
    "tremor": "Tremor"
}

# Display-only helpers (do not affect the model or database)
SHORT_LABELS = {
    "sleep_duration": "Sleep",
    "sleep_quality": "Sleepiness",
    "mood": "Mood",
    "fatigue": "Fatigue",
    "tremor": "Tremor"
}

FEATURE_DESCRIPTIONS = {
    "sleep_duration": "Reported difficulties or disturbances during sleep.",
    "sleep_quality": "How sleepy or alert you feel during the day.",
    "mood": "Reported mood and anxiety difficulties.",
    "fatigue": "Reported tiredness and lack of energy.",
    "tremor": "Reported tremor severity."
}

PAGES = ["Overview", "Daily Check-in", "My Health Trends", "About"]

# Healthcare palette
C_BG = "#F5F8F6"
C_PRIMARY = "#245B52"
C_SECONDARY = "#A8C7B8"
C_ACCENT = "#DCEFE7"
C_DARK = "#263936"
C_WARN = "#E9B96E"
C_CORAL = "#E98C83"

SYMPTOM_COLORS = {
    "sleep_duration": C_WARN,
    "sleep_quality": C_SECONDARY,
    "mood": C_CORAL,
    "fatigue": C_PRIMARY,
    "tremor": "#6FA394"
}

# Text colour that stays readable on each bubble colour
SYMPTOM_TEXT = {
    "sleep_duration": C_DARK,
    "sleep_quality": C_DARK,
    "mood": C_DARK,
    "fatigue": "#FFFFFF",
    "tremor": "#FFFFFF"
}

KEY_SUPPORT = "key" in inspect.signature(st.container).parameters


# ============================================================================
# STYLING
# ============================================================================

def _icon_url(paths):
    """Build a CSS mask image from simple stroke-based SVG paths."""

    return (
        "url(\"data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        "fill='none' stroke='black' stroke-width='2' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        + paths +
        "</svg>\")"
    )


ICONS = {
    "grid": _icon_url(
        "<rect x='3' y='3' width='7' height='7' rx='2'/>"
        "<rect x='14' y='3' width='7' height='7' rx='2'/>"
        "<rect x='3' y='14' width='7' height='7' rx='2'/>"
        "<rect x='14' y='14' width='7' height='7' rx='2'/>"
    ),
    "check": _icon_url(
        "<path d='M9 11l3 3L22 4'/>"
        "<path d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 "
        "2-2h11'/>"
    ),
    "trend": _icon_url(
        "<polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/>"
    ),
    "info": _icon_url(
        "<circle cx='12' cy='12' r='10'/>"
        "<line x1='12' y1='16' x2='12' y2='12'/>"
        "<line x1='12' y1='8' x2='12.01' y2='8'/>"
    ),
    "moon": _icon_url(
        "<path d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/>"
    ),
    "zap": _icon_url(
        "<polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/>"
    ),
    "move": _icon_url(
        "<polyline points='5 9 2 12 5 15'/>"
        "<polyline points='9 5 12 2 15 5'/>"
        "<polyline points='15 19 12 22 9 19'/>"
        "<polyline points='19 9 22 12 19 15'/>"
        "<line x1='2' y1='12' x2='22' y2='12'/>"
        "<line x1='12' y1='2' x2='12' y2='22'/>"
    ),
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #F5F8F6;
    --primary: #245B52;
    --secondary: #A8C7B8;
    --accent: #DCEFE7;
    --dark: #263936;
    --warn: #E9B96E;
    --coral: #E98C83;
    --ink: #1B3530;
    --muted: #5E726C;
    --line: #E4ECE8;
    --shadow: 0 1px 2px rgba(38,57,54,.04), 0 18px 38px -20px rgba(38,57,54,.22);
}

/* ---------- Base ---------- */
.stApp {
    background: var(--bg);
    color: var(--muted);
    font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif !important;
}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif !important;
}
.stMarkdown p, .stMarkdown li { color: var(--muted); }
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton, [data-testid="stAppDeployButton"], #MainMenu, footer {
    display: none !important;
}
.block-container {
    max-width: 1240px;
    padding: 2.4rem 2.6rem 3rem;
}
@media (max-width: 700px) {
    .block-container { padding: 1.4rem 1rem 2.5rem; }
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 17.5rem !important;
    min-width: 17.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 1.2rem 1.1rem 1.5rem;
}
.brand { display: flex; align-items: center; gap: 12px; margin: .2rem 0 1.7rem; }
.brand-mark {
    position: relative; width: 40px; height: 40px; flex: none;
    border-radius: 13px; background: var(--dark);
}
.brand-mark i { position: absolute; border-radius: 50%; display: block; }
.brand-mark i:nth-child(1) { width: 16px; height: 16px; left: 7px; top: 7px; background: var(--warn); }
.brand-mark i:nth-child(2) { width: 14px; height: 14px; left: 17px; top: 14px; background: var(--coral); opacity: .92; }
.brand-mark i:nth-child(3) { width: 10px; height: 10px; left: 8px; top: 22px; background: var(--accent); }
.brand-name { font-weight: 800; font-size: 1.4rem; color: var(--primary); letter-spacing: -.03em; line-height: 1; }
.brand-sub { color: #7E918A; font-size: .78rem; margin-top: 5px; }
.side-label { font-weight: 800; color: var(--ink); font-size: .88rem; margin: 1.8rem 0 .2rem; }
.side-note { color: #8A9C96; font-size: .76rem; line-height: 1.55; }
.side-foot {
    margin-top: 1.6rem; padding: 14px 16px; border-radius: 16px;
    background: #F3F8F5; color: #6B7F78; font-size: .75rem; line-height: 1.55;
}

/* Navigation (radio restyled as a vertical menu; works with old and new Streamlit markup) */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > div { width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex; align-items: center; gap: 12px; width: 100%; box-sizing: border-box;
    padding: 11px 14px; border-radius: 16px; cursor: pointer;
    color: #4F655F; transition: background .2s ease, color .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child:not([data-testid]):not(:last-child) {
    display: none !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]) {
    display: none !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div { flex: 1; }
section[data-testid="stSidebar"] div[role="radiogroup"] label::before {
    content: ""; width: 18px; height: 18px; flex: none;
    background-color: currentColor;
    -webkit-mask-image: var(--ico); mask-image: var(--ico);
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center; mask-position: center;
    -webkit-mask-size: contain; mask-size: contain;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    margin: 0; color: inherit !important; font-weight: 600; font-size: .94rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #EEF5F1; color: var(--primary);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--dark); color: #FFFFFF;
    box-shadow: 0 10px 22px -12px rgba(38,57,54,.8);
}
/*NAVICONS*/

/* ---------- Typography blocks ---------- */
.page-head { margin: 0 0 1.5rem; }
.page-title {
    font-size: clamp(2rem, 3.4vw, 3rem); font-weight: 800; color: var(--ink);
    letter-spacing: -.035em; line-height: 1.08;
}
.page-sub { color: var(--muted); font-size: 1.02rem; margin-top: .6rem; max-width: 44rem; line-height: 1.55; }
.section-title { font-size: 1.15rem; font-weight: 800; color: var(--ink); letter-spacing: -.01em; margin: .3rem 0 .9rem; }
.card-title { font-size: 1.12rem; font-weight: 800; color: var(--ink); letter-spacing: -.01em; }
.card-sub { color: #7A8D86; font-size: .86rem; margin-top: .25rem; line-height: 1.5; }

/* ---------- Welcome ---------- */
.welcome {
    position: relative; overflow: hidden; border-radius: 28px;
    padding: 30px 34px; margin-bottom: 1.6rem;
    background: linear-gradient(120deg, #DCEFE7 0%, #E8F4EE 60%, #F0F7F3 100%);
}
.welcome::after {
    content: ""; position: absolute; right: -50px; top: -70px; width: 270px; height: 270px; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,.95), rgba(168,199,184,.6) 60%, rgba(168,199,184,0) 73%);
}
.welcome::before {
    content: ""; position: absolute; right: 190px; bottom: -58px; width: 140px; height: 140px; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,.85), rgba(233,185,110,.6) 60%, rgba(233,185,110,0) 74%);
}
.welcome-title { font-size: clamp(1.5rem, 2.4vw, 2.1rem); font-weight: 800; color: var(--primary); letter-spacing: -.025em; position: relative; z-index: 1; }
.welcome-text { max-width: 36rem; color: #4B6660; margin-top: .55rem; line-height: 1.6; position: relative; z-index: 1; }

/* ---------- Stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 1.5rem; }
.stat { background: #FFFFFF; border: 1px solid var(--line); border-radius: 22px; padding: 20px 22px; }
.stat-label { color: var(--muted); font-size: .86rem; font-weight: 600; }
.stat-value { font-size: 2.05rem; font-weight: 800; color: var(--ink); letter-spacing: -.03em; line-height: 1.12; margin-top: .3rem; }
.stat-note { color: #8A9C96; font-size: .8rem; margin-top: .4rem; }

/* ---------- Keyed Streamlit containers used as cards ---------- */
div[class*="st-key-card_"] {
    background: #FFFFFF; border: 1px solid var(--line); border-radius: 26px;
    padding: 24px 26px; box-shadow: var(--shadow);
}
div[class*="st-key-dark_"] {
    background: var(--dark); border-radius: 30px; padding: 26px 26px 22px;
    min-height: 500px; box-shadow: var(--shadow);
}
div[class*="st-key-dark_"] button {
    background: rgba(255,255,255,.08) !important; color: #E9F2EE !important;
    border: 1px solid rgba(255,255,255,.18) !important; min-height: 2.3rem;
}
div[class*="st-key-dark_"] button:hover { background: rgba(255,255,255,.18) !important; transform: none; }
div[class*="st-key-dark_"] button p { color: inherit !important; }
.st-key-hdr_new { display: flex; justify-content: flex-end; padding-top: .5rem; }

/* ---------- Observation panel (bubbles) ---------- */
.panel-sage {
    background: linear-gradient(135deg, #DCEFE7 0%, #CDE4D9 100%);
    border-radius: 30px; padding: 28px 30px; box-shadow: var(--shadow); min-height: 500px; margin-bottom: .5rem;
}
.panel-grid { display: grid; grid-template-columns: minmax(230px, .85fr) minmax(280px, 1.35fr); gap: 18px; align-items: center; }
@media (max-width: 820px) { .panel-grid { grid-template-columns: 1fr; } }
.panel-title { font-size: 1.3rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; }
.panel-sub { color: #55706A; font-size: .86rem; margin-top: .3rem; }
.lg { margin-top: 1.3rem; display: grid; gap: 9px; }
.lg-row { display: flex; align-items: center; gap: 10px; font-size: .86rem; color: #3E5852; }
.lg-pill { width: 26px; height: 11px; border-radius: 999px; flex: none; box-shadow: inset 0 -2px 3px rgba(0,0,0,.12); }
.lg-name { flex: 1; font-weight: 600; }
.lg-val { font-weight: 800; color: var(--ink); }
.note {
    margin-top: 1.1rem; padding: 14px 18px; border-radius: 18px;
    background: rgba(255,255,255,.62); color: #3E5852; font-size: .82rem; line-height: 1.55;
}
.note b { color: var(--ink); }
.bubble-wrap { position: relative; width: 100%; }
.bubble-wrap img { width: 100%; display: block; }
.bub { position: absolute; transform: translate(-50%, -50%); text-align: center; line-height: 1.05; font-weight: 800; font-size: clamp(.9rem, 1.7vw, 1.35rem); pointer-events: none; }
.bub small { display: block; font-size: .62rem; font-weight: 600; opacity: .8; letter-spacing: .02em; }
.bub.score { font-size: clamp(1rem, 2vw, 1.65rem); color: #FFFFFF; }

/* ---------- Calendar ---------- */
.cal-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.cal-title { color: #FFFFFF; font-weight: 800; font-size: 1.15rem; letter-spacing: -.01em; }
.cal-month { color: #B9D0C7; font-size: .9rem; font-weight: 600; }
.cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 7px 4px; margin-top: 18px; text-align: center; }
.cal-dow { color: #8FA9A0; font-size: .74rem; font-weight: 700; padding-bottom: 4px; }
.cal-day {
    display: flex; align-items: center; justify-content: center; width: 100%; max-width: 38px;
    aspect-ratio: 1; margin: 0 auto; border-radius: 50%; color: #D2E2DB; font-size: .86rem; font-weight: 600;
}
.cal-day.done { background: var(--warn); color: #2B2A1E; }
.cal-day.today { outline: 1.5px solid #FFFFFF; outline-offset: 2px; }
.cal-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 18px 0 6px; color: #A9C0B7; font-size: .76rem; }
.cal-legend span { display: inline-flex; align-items: center; gap: 6px; }
.cal-legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.cal-count { color: #D2E2DB; font-size: .84rem; margin-top: 10px; }

/* ---------- Activity ---------- */
.act-top { display: flex; align-items: center; gap: 20px; margin-top: 18px; flex-wrap: wrap; }
.ring {
    --p: 0; width: 128px; height: 128px; border-radius: 50%; flex: none; display: grid; place-items: center;
    background: conic-gradient(var(--primary) calc(var(--p) * 1%), #E3EFE9 0);
}
.ring-in { width: 96px; height: 96px; border-radius: 50%; background: #FFFFFF; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring-in b { font-size: 1.8rem; font-weight: 800; color: var(--ink); line-height: 1; }
.ring-in span { font-size: .7rem; color: #7A8D86; margin-top: 3px; }
.act-facts { display: grid; gap: 12px; }
.act-fact b { display: block; font-size: 1.35rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; line-height: 1.1; }
.act-fact span { font-size: .8rem; color: #7A8D86; }
.strip { display: grid; grid-template-columns: repeat(14, 1fr); gap: 5px; margin-top: 20px; }
.strip i { height: 34px; border-radius: 9px; background: var(--accent); display: block; }
.strip i.on { background: var(--primary); }
.strip-cap { display: flex; justify-content: space-between; color: #8A9C96; font-size: .72rem; margin-top: 7px; }

/* ---------- Recent observations ---------- */
.obs-list { display: grid; gap: 10px; margin: 16px 0 .5rem; }
.obs-row { display: flex; align-items: center; gap: 14px; padding: 12px 16px; border-radius: 18px; background: #F4F8F6; flex-wrap: wrap; }
.obs-date { width: 46px; height: 46px; border-radius: 50%; background: var(--accent); display: flex; flex-direction: column; align-items: center; justify-content: center; flex: none; color: var(--primary); }
.obs-date b { font-size: 1rem; line-height: 1; font-weight: 800; }
.obs-date span { font-size: .62rem; font-weight: 700; text-transform: uppercase; }
.obs-main { flex: 1; min-width: 150px; }
.obs-when { font-weight: 700; color: var(--ink); font-size: .9rem; }
.obs-sum { color: #7A8D86; font-size: .8rem; margin-top: 2px; }
.obs-bars { display: flex; align-items: flex-end; gap: 4px; height: 30px; }
.obs-bars i { width: 7px; border-radius: 4px; display: block; }
.obs-score { background: #FFFFFF; border-radius: 999px; padding: 6px 14px; font-weight: 800; color: var(--primary); font-size: .85rem; }

/* ---------- Check-in page ---------- */
.sec-head { display: flex; align-items: center; gap: 14px; margin-bottom: .6rem; }
.ico-chip { width: 44px; height: 44px; border-radius: 15px; background: var(--accent); color: var(--primary); display: flex; align-items: center; justify-content: center; flex: none; }
.ico { display: block; width: 20px; height: 20px; background-color: currentColor; -webkit-mask-image: var(--ico); mask-image: var(--ico); -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center; mask-position: center; -webkit-mask-size: contain; mask-size: contain; }
.sec-title { font-weight: 800; color: var(--ink); font-size: 1.1rem; letter-spacing: -.01em; }
.sec-desc { color: #7A8D86; font-size: .85rem; margin-top: 2px; }
.anchors { display: flex; justify-content: space-between; gap: 12px; color: #7A8D86; font-size: .78rem; margin: -.5rem 0 .7rem; }
.guide { background: var(--dark); border-radius: 28px; padding: 26px; color: #D6E5DF; }
.guide-title { color: #FFFFFF; font-weight: 800; font-size: 1.1rem; }
.guide-text { font-size: .86rem; line-height: 1.6; margin-top: .6rem; color: #C3D6CE; }
.scale { display: flex; align-items: flex-end; justify-content: space-between; margin-top: 1.3rem; padding: 0 4px; }
.scale div { text-align: center; color: #FFFFFF; font-weight: 800; font-size: .8rem; }
.scale i { display: block; border-radius: 50%; margin: 0 auto 6px; }
.scale-cap { display: flex; justify-content: space-between; margin-top: .8rem; font-size: .74rem; color: #93AEA4; }
.idcard { margin-top: 14px; background: #FFFFFF; border: 1px solid var(--line); border-radius: 22px; padding: 18px 20px; }
.idcard b { display: block; color: var(--ink); font-weight: 800; font-size: 1.05rem; margin-top: 4px; word-break: break-all; }
.idcard span { color: #7A8D86; font-size: .8rem; }
.disclaimer {
    background: #FFF8E9; border: 1px solid #F1E2BD; border-radius: 18px;
    padding: 14px 18px; color: #75613B; font-size: .88rem; line-height: 1.55; margin-bottom: 1rem;
}

/* ---------- About ---------- */
.about-hero { background: linear-gradient(120deg, #DCEFE7, #EAF5EF); border-radius: 28px; padding: 30px 34px; margin-bottom: 1.3rem; color: #3E5A54; line-height: 1.65; font-size: 1.02rem; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-bottom: 1.5rem; }
.grid-feat { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 1.5rem; }
.info-card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 24px; padding: 22px 24px; color: var(--muted); line-height: 1.6; font-size: .93rem; }
.info-card b.t { display: block; color: var(--ink); font-size: 1.02rem; margin-bottom: 6px; font-weight: 800; }
.step-n { width: 34px; height: 34px; border-radius: 50%; background: var(--primary); color: #FFFFFF; font-weight: 800; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
.dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; margin-bottom: 12px; }
.dark-info { background: var(--dark); border-radius: 28px; padding: 28px 30px; color: #CFE0D9; margin-bottom: 1.5rem; line-height: 1.65; font-size: .95rem; }
.dark-info .t { color: #FFFFFF; font-size: 1.15rem; font-weight: 800; margin-bottom: .6rem; }
.dark-info .bl { display: flex; gap: 12px; align-items: flex-start; margin-top: 10px; }
.dark-info .bl i { width: 9px; height: 9px; border-radius: 50%; background: var(--warn); flex: none; margin-top: 8px; }
.dark-info .fine { margin-top: 1rem; color: #FFFFFF; font-weight: 700; }
.info-card .bl { display: flex; gap: 12px; align-items: flex-start; margin-top: 8px; }
.info-card .bl i { width: 9px; height: 9px; border-radius: 50%; background: var(--secondary); flex: none; margin-top: 8px; }

/* ---------- Native widgets ---------- */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 999px; font-weight: 700; padding: .55rem 1.3rem;
    transition: background-color .2s ease, color .2s ease, border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
.stButton > button { background: #FFFFFF; color: var(--primary); border: 1px solid #D3E3DC; }
.stButton > button:hover { border-color: var(--primary); color: var(--primary); transform: translateY(-1px); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--dark); color: #FFFFFF; border: none;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--primary); color: #FFFFFF;
}
.stButton > button p, .stFormSubmitButton > button p { color: inherit !important; }
.stFormSubmitButton, div[data-testid="stFormSubmitButton"] { width: 100%; }
.stFormSubmitButton > button {
    width: 100%; min-height: 3.4rem; background: var(--primary); color: #FFFFFF; border: none; font-size: 1.02rem;
    box-shadow: 0 14px 26px -14px rgba(36,91,82,.8);
}
.stFormSubmitButton > button:hover { background: #1B4A42; color: #FFFFFF; transform: translateY(-1px); }
div[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
div[data-testid="stForm"] div[class*="st-key-card_"] { margin-bottom: 1rem; }
div[data-testid="stSlider"] label p { font-weight: 700; color: var(--ink); }
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background-color: var(--primary); box-shadow: 0 0 0 5px rgba(36,91,82,.14); }
div[data-testid="stSliderThumbValue"], div[data-testid="stThumbValue"] { color: var(--primary); font-weight: 800; }
.stTextInput input, div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    border-radius: 14px; background: #FFFFFF; border-color: #DAE6E0;
}
.stTextInput input:focus { border-color: var(--primary); box-shadow: 0 0 0 1px var(--primary); }
span[data-baseweb="tag"], span[data-tag] { background: var(--accent) !important; color: var(--primary) !important; border-radius: 999px !important; }
span[data-baseweb="tag"] span, span[data-tag] span, span[data-tag] button { color: var(--primary) !important; }
div[data-testid="stAlertContainer"] { border-radius: 16px; }
div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
"""


def inject_styles():
    """Inject the dashboard stylesheet (display only)."""

    nav_icons = ["grid", "check", "trend", "info"]
    icon_css = "".join(
        "section[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] > "
        f":nth-child({index + 1}) {{ --ico: {ICONS[name]}; }}\n"
        for index, name in enumerate(nav_icons)
    )
    icon_css += "".join(
        f".i-{name} {{ --ico: {ICONS[name]}; }}\n"
        for name in ["moon", "zap", "move"]
    )
    css = BASE_CSS.replace("/*NAVICONS*/", icon_css)

    # Without a configured theme, Streamlit paints sliders in its default red.
    # Shift that hue to the brand green. Skipped when .streamlit/config.toml
    # already sets the theme's primaryColor.
    if not st.get_option("theme.primaryColor"):
        css += (
            "div[data-testid=\"stSlider\"] div[role=\"group\"], "
            "div[data-testid=\"stSlider\"] div[data-baseweb=\"slider\"] "
            "{ filter: hue-rotate(172deg) saturate(.5) brightness(.62); }"
        )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_styles()


# ============================================================================
# SMALL UI HELPERS
# ============================================================================

esc = html.escape


def html_block(markup):
    """Render an HTML snippet. Lines are collapsed so Markdown never
    mistakes indented HTML for a code block."""

    st.markdown(
        " ".join(
            line.strip() for line in markup.splitlines() if line.strip()
        ),
        unsafe_allow_html=True
    )


def card(key, dark=False):
    """A styled Streamlit container. Falls back to a bordered container
    on older Streamlit versions that do not support container keys."""

    name = f"{'dark' if dark else 'card'}_{key}"

    if KEY_SUPPORT:
        return st.container(key=name)

    return st.container(border=True)


def go_to(page_name):
    """Navigation callback using session state."""

    st.session_state["page"] = page_name


def show_chart(fig):
    """Display a Plotly chart across the container width."""

    config = {"displayModeBar": False}

    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=config)
    except Exception:
        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            config=config
        )


def show_table(df):
    """Display a dataframe across the container width."""

    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def num(value, default=0):
    """Safe numeric conversion for display."""

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_score(value):
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"


def page_header(title, subtitle):
    html_block(f"""
        <div class="page-head">
            <div class="page-title">{title}</div>
            <div class="page-sub">{subtitle}</div>
        </div>
    """)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load the trained Isolation Forest and scaler."""

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found."
        )

    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(
            f"Scaler file '{SCALER_PATH}' was not found."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


try:
    iso_forest, scaler = load_models()

    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(FEATURE_COLUMNS):
            st.error(
                "The scaler expects a different number of input features. "
                "Please check the training pipeline."
            )
            st.stop()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Place isolation_forest_model.pkl and feature_scaler.pkl "
        "in the same folder as app.py."
    )
    st.stop()


# ============================================================================
# DATABASE
# ============================================================================

def get_connection():
    """Create a SQLite connection."""

    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    """Create the submissions table if it does not exist."""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                date TEXT,
                sleep_duration REAL,
                sleep_quality REAL,
                mood REAL,
                fatigue REAL,
                tremor REAL,
                score REAL,
                risk TEXT
            )
        """)


def save_submission(user_id, features, score):
    """Save a completed check-in."""

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO submissions (
                user_id,
                date,
                sleep_duration,
                sleep_quality,
                mood,
                fatigue,
                tremor,
                score,
                risk
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            features["sleep_duration"],
            features["sleep_quality"],
            features["mood"],
            features["fatigue"],
            features["tremor"],
            score,
            "Experimental model output"
        ))


def get_history(user_id):
    """Retrieve check-ins for a particular ID."""

    with get_connection() as conn:
        history = pd.read_sql_query("""
            SELECT *
            FROM submissions
            WHERE user_id = ?
            ORDER BY date DESC
        """, conn, params=(user_id,))

    return history


try:
    init_db()
except sqlite3.Error as error:
    st.error(f"Database could not be initialized: {error}")
    st.stop()


# ============================================================================
# MODEL SCORING
# ============================================================================

def score_features(features_array):
    """
    Generate an Isolation Forest anomaly score.

    A lower score_samples value indicates a more unusual
    observation relative to the patterns learned by the model.

    This is NOT a Parkinson's disease probability.
    """

    normalized = scaler.transform(features_array)
    score = float(iso_forest.score_samples(normalized)[0])

    return score


def get_score_explanation(score):
    """Return a non-clinical explanation of the model score."""

    return (
        "The model has generated an experimental anomaly score. "
        "Compare this score with your previous readings to observe "
        "changes over time. It does not estimate your probability "
        "of having Parkinson's disease."
    )


# ============================================================================
# DATA HELPERS (read-only views of the existing database)
# ============================================================================

def prepare_history(user_id):
    """History for one ID with parsed dates, newest first."""

    history = get_history(user_id)

    if history.empty:
        return history

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    return (
        history.dropna(subset=["date"])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def style_fig(fig, height=360):
    """Shared Plotly styling."""

    font = "Manrope, Segoe UI, sans-serif"

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=font, color="#4E635D", size=13),
        margin=dict(l=8, r=8, t=12, b=8),
        hoverlabel=dict(
            bgcolor=C_DARK,
            bordercolor=C_DARK,
            font=dict(family=font, color="#FFFFFF")
        ),
        legend=dict(orientation="h", y=-0.22, x=0)
    )
    fig.update_xaxes(showgrid=False, linecolor="#E4ECE8", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#E9F0EC", zeroline=False, automargin=True)

    return fig


def make_score_trend(history):
    """Create a chronological anomaly-score trend."""

    chart_data = history.copy()

    chart_data["date"] = pd.to_datetime(
        chart_data["date"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["date", "score"]
    ).sort_values("date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_data["date"],
        y=chart_data["score"],
        mode="lines+markers",
        line=dict(color=C_PRIMARY, width=3),
        marker=dict(
            size=10,
            color="#FFFFFF",
            line=dict(color=C_PRIMARY, width=3)
        ),
        name="Model score",
        hovertemplate=(
            "%{x|%d %b %Y, %I:%M %p}<br>"
            "Score: %{y:.4f}<extra></extra>"
        )
    ))

    if len(chart_data) > 1:
        fig.add_hline(
            y=float(chart_data["score"].mean()),
            line_dash="dot",
            line_color=C_CORAL,
            annotation_text="Your average in this range",
            annotation_position="bottom right",
            annotation_font=dict(color=C_CORAL, size=12)
        )

    style_fig(fig, 340)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Anomaly score", title_standoff=10)

    return fig


def make_symptom_trend(history, columns=None):
    """Plot the selected symptom features over time."""

    columns = columns or FEATURE_COLUMNS

    chart_data = history.copy()

    chart_data["date"] = pd.to_datetime(
        chart_data["date"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["date"]
    ).sort_values("date")

    fig = go.Figure()

    for column in columns:
        fig.add_trace(go.Scatter(
            x=chart_data["date"],
            y=chart_data[column],
            mode="lines+markers",
            name=FEATURE_LABELS[column],
            line=dict(color=SYMPTOM_COLORS[column], width=3),
            marker=dict(size=8),
            hovertemplate="%{y:.0f} / 4<extra>" + FEATURE_LABELS[column] + "</extra>"
        ))

    style_fig(fig, 380)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(
        title_text="Reported severity",
        range=[-0.3, 4.3],
        tickvals=[0, 1, 2, 3, 4]
    )

    return fig


def make_symptom_heatmap(history, columns):
    """Day-by-day view of reported values for the selected symptoms."""

    data = history.sort_values("date")
    labels = data["date"].dt.strftime("%d %b, %H:%M").tolist()

    fig = go.Figure(go.Heatmap(
        z=[data[c].astype(float).tolist() for c in columns],
        x=labels,
        y=[FEATURE_LABELS[c] for c in columns],
        zmin=0,
        zmax=4,
        xgap=4,
        ygap=4,
        colorscale=[
            [0.0, "#EAF5EF"],
            [0.5, C_WARN],
            [1.0, C_CORAL]
        ],
        colorbar=dict(
            title="0-4",
            thickness=10,
            tickvals=[0, 1, 2, 3, 4]
        ),
        hovertemplate="%{y}<br>%{x}<br>%{z:.0f} / 4<extra></extra>"
    ))

    style_fig(fig, 90 + 56 * len(columns))
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(autorange="reversed")

    return fig


# ============================================================================
# DASHBOARD COMPONENTS
# ============================================================================

def make_bubble_cluster(features, score):
    """Bubble visual: five symptom bubbles (size = reported 0-4 value) around
    a fixed-size model-score bubble. The score bubble is deliberately NOT
    sized by the score, because the score is not a probability."""

    width, height = 500, 440
    score_pos = (250, 205, 86)

    positions = {
        "sleep_duration": (97, 112),
        "sleep_quality": (105, 300),
        "mood": (403, 105),
        "fatigue": (398, 300),
        "tremor": (250, 362)
    }

    circles = [(score_pos[0], score_pos[1], score_pos[2], C_DARK)]
    labels = (
        f"<div class='bub score' style='left:{score_pos[0] / width * 100:.2f}%;"
        f"top:{score_pos[1] / height * 100:.2f}%'>"
        f"<small>Model score</small>{fmt_score(score)}</div>"
    )

    for column in FEATURE_COLUMNS:
        cx, cy = positions[column]
        value = num(features[column])
        radius = 21 + 14 * value
        circles.append((cx, cy, radius, SYMPTOM_COLORS[column]))
        labels += (
            f"<div class='bub' title='{esc(FEATURE_LABELS[column])}' "
            f"style='left:{cx / width * 100:.2f}%;top:{cy / height * 100:.2f}%;"
            f"color:{SYMPTOM_TEXT[column]}'>{value:.0f}</div>"
        )

    fills = "".join(
        f"<circle cx='{x}' cy='{y}' r='{r}' fill='{c}'/>"
        for x, y, r, c in circles
    )
    shine = "".join(
        f"<circle cx='{x}' cy='{y}' r='{r}' fill='url(#shine)'/>"
        for x, y, r, c in circles
    )
    rings = "".join(
        f"<circle cx='{x}' cy='{y}' r='{r}' fill='none' stroke='white' "
        "stroke-opacity='.55' stroke-width='2'/>"
        for x, y, r, c in circles
    )

    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>"
        "<defs>"
        "<filter id='goo' x='-20%' y='-20%' width='140%' height='140%'>"
        "<feGaussianBlur in='SourceGraphic' stdDeviation='7' result='b'/>"
        "<feColorMatrix in='b' mode='matrix' "
        "values='1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 19 -8'/>"
        "</filter>"
        "<radialGradient id='shine' cx='32%' cy='28%' r='75%'>"
        "<stop offset='0' stop-color='white' stop-opacity='.55'/>"
        "<stop offset='.55' stop-color='white' stop-opacity='0'/>"
        "<stop offset='1' stop-color='black' stop-opacity='.10'/>"
        "</radialGradient>"
        "</defs>"
        f"<g filter='url(#goo)'>{fills}</g>{shine}{rings}"
        "</svg>"
    )

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return (
        "<div class='bubble-wrap'>"
        f"<img alt='Bubble chart of your reported symptoms' "
        f"src='data:image/svg+xml;base64,{encoded}'/>"
        f"{labels}</div>"
    )


def observation_panel(features, score, when_text, heading="Latest observation"):
    """Main visual panel for one recorded observation."""

    legend = (
        "<div class='lg-row'><span class='lg-pill' "
        f"style='background:{C_DARK}'></span>"
        "<span class='lg-name'>Isolation Forest score</span>"
        f"<span class='lg-val'>{fmt_score(score)}</span></div>"
    )

    for column in FEATURE_COLUMNS:
        legend += (
            "<div class='lg-row'><span class='lg-pill' "
            f"style='background:{SYMPTOM_COLORS[column]}'></span>"
            f"<span class='lg-name'>{FEATURE_LABELS[column]}</span>"
            f"<span class='lg-val'>{num(features[column]):.0f} / 4</span></div>"
        )

    html_block(f"""
        <div class="panel-sage">
            <div class="panel-grid">
                <div>
                    <div class="panel-title">{esc(heading)}</div>
                    <div class="panel-sub">Check-in recorded on {esc(when_text)}</div>
                    <div class="lg">{legend}</div>
                </div>
                {make_bubble_cluster(features, score)}
            </div>
            <div class="note">
                <b>How to read the score.</b>
                {esc(get_score_explanation(score))}
                Lower values mean the entry looks more unusual compared
                with the model's learned patterns; higher values mean
                more typical. It is an anomaly score, not a disease
                probability. Bubble size shows the 0-4 value you
                reported.
            </div>
        </div>
    """)


def stat_cards(cards):
    """Row of simple statistic cards: (label, value, note)."""

    items = "".join(
        f"<div class='stat'><div class='stat-label'>{esc(label)}</div>"
        f"<div class='stat-value'>{esc(str(value))}</div>"
        f"<div class='stat-note'>{esc(note)}</div></div>"
        for label, value, note in cards
    )

    html_block(f"<div class='stat-grid'>{items}</div>")


def shift_month(delta):
    month = st.session_state.cal_month + delta
    year = st.session_state.cal_year

    if month < 1:
        month, year = 12, year - 1
    elif month > 12:
        month, year = 1, year + 1

    st.session_state.cal_month = month
    st.session_state.cal_year = year


def reset_month():
    today = date.today()
    st.session_state.cal_month = today.month
    st.session_state.cal_year = today.year


def calendar_card(history):
    """Monthly check-in calendar built from real database records."""

    year = st.session_state.cal_year
    month = st.session_state.cal_month
    today = date.today()
    checkin_days = set(history["date"].dt.date)

    cells = "".join(
        f"<div class='cal-dow'>{d}</div>"
        for d in ["M", "T", "W", "T", "F", "S", "S"]
    )

    month_days = 0

    for week in calendar.monthcalendar(year, month):
        for day in week:
            if day == 0:
                cells += "<div></div>"
                continue

            current = date(year, month, day)
            classes = "cal-day"

            if current in checkin_days:
                classes += " done"
                month_days += 1

            if current == today:
                classes += " today"

            cells += f"<div class='{classes}'>{day}</div>"

    fallback_style = (
        "" if KEY_SUPPORT
        else "background:#263936;border-radius:26px;padding:22px;"
    )

    with card("calendar", dark=True):
        html_block(f"""
            <div style="{fallback_style}">
                <div class="cal-head">
                    <div class="cal-title">Check-in calendar</div>
                    <div class="cal-month">{calendar.month_name[month]} {year}</div>
                </div>
                <div class="cal-grid">{cells}</div>
                <div class="cal-legend">
                    <span><i style="outline:1.5px solid #fff;outline-offset:1px"></i>Today</span>
                    <span><i style="background:{C_WARN}"></i>Check-in recorded</span>
                    <span><i style="background:rgba(255,255,255,.18)"></i>No check-in</span>
                </div>
                <div class="cal-count">
                    {month_days} day{'s' if month_days != 1 else ''} with a check-in in {calendar.month_name[month]}
                </div>
            </div>
        """)

        nav1, nav2, nav3 = st.columns(3)

        with nav1:
            st.button("Previous", key="cal_prev", on_click=shift_month, args=(-1,))
        with nav2:
            st.button("Today", key="cal_today", on_click=reset_month)
        with nav3:
            st.button("Next", key="cal_next", on_click=shift_month, args=(1,))


def activity_card(history):
    """Recent monitoring activity derived from existing records."""

    today = date.today()
    window = [today - timedelta(days=offset) for offset in range(13, -1, -1)]
    checkin_days = set(history["date"].dt.date)

    days_with = sum(1 for day in window if day in checkin_days)
    last_seven = int((history["date"].dt.date >= today - timedelta(days=6)).sum())
    percent = days_with / 14 * 100

    strip = "".join(
        f"<i class=\"{'on' if day in checkin_days else ''}\" "
        f"title=\"{day:%d %b}\"></i>"
        for day in window
    )

    html_block(f"""
        <div class="card-title">Monitoring activity</div>
        <div class="card-sub">Recorded check-ins and recent activity.</div>
        <div class="act-top">
            <div class="ring" style="--p:{percent:.0f}">
                <div class="ring-in"><b>{days_with}</b><span>of 14 days</span></div>
            </div>
            <div class="act-facts">
                <div class="act-fact"><b>{len(history)}</b><span>check-ins recorded in total</span></div>
                <div class="act-fact"><b>{last_seven}</b><span>in the last 7 days</span></div>
            </div>
        </div>
        <div class="strip">{strip}</div>
        <div class="strip-cap"><span>14 days ago</span><span>Today</span></div>
    """)


def recent_card(history):
    """Most recent observations."""

    rows = ""

    for _, row in history.head(5).iterrows():
        stamp = row["date"]
        summary = ", ".join(
            f"{SHORT_LABELS[c]} {num(row[c]):.0f}" for c in FEATURE_COLUMNS
        )
        bars = "".join(
            f"<i style='height:{max(8, num(row[c]) / 4 * 100):.0f}%;"
            f"background:{SYMPTOM_COLORS[c]}'></i>"
            for c in FEATURE_COLUMNS
        )
        rows += (
            "<div class='obs-row'>"
            f"<div class='obs-date'><b>{stamp:%d}</b><span>{stamp:%b}</span></div>"
            "<div class='obs-main'>"
            f"<div class='obs-when'>{stamp:%d %B %Y, %I:%M %p}</div>"
            f"<div class='obs-sum'>{esc(summary)}</div></div>"
            f"<div class='obs-bars'>{bars}</div>"
            f"<div class='obs-score'>{fmt_score(row['score'])}</div>"
            "</div>"
        )

    html_block(f"""
        <div class="card-title">Recent observations</div>
        <div class="card-sub">Your latest check-ins with the model score and the values you reported.</div>
        <div class="obs-list">{rows}</div>
    """)

    st.button(
        "See all in My Health Trends",
        key="recent_to_trends",
        on_click=go_to,
        args=("My Health Trends",)
    )


# ============================================================================
# SESSION STATE
# ============================================================================

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "patient_001"

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

if "cal_month" not in st.session_state:
    st.session_state.cal_month = date.today().month
    st.session_state.cal_year = date.today().year


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    html_block("""
        <div class="brand">
            <div class="brand-mark"><i></i><i></i><i></i></div>
            <div>
                <div class="brand-name">Neurovia</div>
                <div class="brand-sub">Daily Health Monitoring</div>
            </div>
        </div>
    """)

    page = st.radio(
        "Navigation",
        PAGES,
        key="page",
        label_visibility="collapsed"
    )

    html_block('<div class="side-label">Your tracking ID</div>')

    patient_id = st.text_input(
        "Patient ID",
        value=st.session_state.patient_id,
        key="sidebar_patient_id",
        help=(
            "This ID groups your records in the prototype. "
            "It is not a password or secure login."
        )
    )

    if patient_id.strip():
        st.session_state.patient_id = patient_id.strip()

    html_block("""
        <div class="side-note">
            Prototype only. Your tracking ID does not authenticate
            your identity or protect your records.
        </div>
        <div class="side-foot">
            Powered by an experimental Isolation Forest model. This is not a
            diagnostic tool and does not replace advice from a healthcare
            professional.
        </div>
    """)


# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":

    head_left, head_right = st.columns([3, 1])

    with head_left:
        page_header(
            "Your health, one day at a time",
            "Monitor your daily observations and understand patterns over time."
        )

    with head_right:
        st.button(
            "New check-in",
            key="hdr_new",
            type="primary",
            on_click=go_to,
            args=("Daily Check-in",)
        )

    html_block("""
        <div class="welcome">
            <div class="welcome-title">A little check-in can go a long way.</div>
            <div class="welcome-text">
                Take a moment to record how you have been feeling. Your
                entries help you review patterns in your daily observations
                over time.
            </div>
        </div>
    """)

    if not st.session_state.patient_id:
        st.warning("Enter your tracking ID in the sidebar to continue.")

    else:
        try:
            history = prepare_history(st.session_state.patient_id)

            if history.empty:

                html_block(
                    '<div class="section-title">Your monitoring journey starts here</div>'
                )

                st.info(
                    "You have not recorded a check-in yet. "
                    "Complete your first daily check-in to start "
                    "building your personal history."
                )

                st.button(
                    "Complete your first check-in",
                    type="primary",
                    on_click=go_to,
                    args=("Daily Check-in",)
                )

            else:
                latest = history.iloc[0]
                latest_date = latest["date"]
                first_date = history["date"].min()

                html_block('<div class="section-title">Your latest update</div>')

                stat_cards([
                    (
                        "Total check-ins",
                        len(history),
                        f"Since {first_date:%d %b %Y}"
                    ),
                    (
                        "Latest model score",
                        fmt_score(latest["score"]),
                        "Experimental anomaly score"
                    ),
                    (
                        "Last recorded",
                        f"{latest_date:%d %b}",
                        f"{latest_date:%Y, %I:%M %p}"
                    )
                ])

                col_main, col_cal = st.columns([1.75, 1])

                with col_main:
                    features = {c: latest[c] for c in FEATURE_COLUMNS}
                    observation_panel(
                        features,
                        latest["score"],
                        f"{latest_date:%d %B %Y, %I:%M %p}"
                    )

                with col_cal:
                    calendar_card(history)

                col_act, col_recent = st.columns([1, 1.55])

                with col_act:
                    with card("activity"):
                        activity_card(history)

                with col_recent:
                    with card("recent"):
                        recent_card(history)

        except sqlite3.Error as error:
            st.error(f"Unable to retrieve your records: {error}")


# ============================================================================
# PAGE 2: DAILY CHECK-IN
# ============================================================================

elif page == "Daily Check-in":

    page_header(
        "Daily check-in",
        "Answer a few questions about how you have been feeling. "
        "There are no right or wrong answers."
    )

    html_block("""
        <div class="disclaimer">
            <strong>Before you begin:</strong>
            This check-in is an experimental project feature.
            It does not diagnose Parkinson's disease or replace
            an assessment by a healthcare professional.
        </div>
    """)

    def section_header(icon, title, description):
        html_block(f"""
            <div class="sec-head">
                <div class="ico-chip"><i class="ico i-{icon}"></i></div>
                <div>
                    <div class="sec-title">{title}</div>
                    <div class="sec-desc">{description}</div>
                </div>
            </div>
        """)

    def scale_slider(column, default, help_text, low, high):
        value = st.slider(
            FEATURE_LABELS[column],
            min_value=0,
            max_value=4,
            value=default,
            help=help_text
        )
        html_block(
            f"<div class='anchors'><span>0 = {low}</span>"
            f"<span>4 = {high}</span></div>"
        )
        return value

    if not st.session_state.patient_id:
        st.warning("Enter a tracking ID in the sidebar before submitting.")

    else:

        form_col, guide_col = st.columns([2.15, 1])

        with form_col:

            with st.form("daily_checkin_form"):

                with card("sleep"):
                    section_header(
                        "moon",
                        "Sleep & Rest",
                        "Think about your sleep and alertness during the day."
                    )

                    sleep_duration = scale_slider(
                        "sleep_duration", 1,
                        "0 = No problems, 4 = Severe sleep disturbances.",
                        "No problems", "Severe sleep disturbances"
                    )

                    sleep_quality = scale_slider(
                        "sleep_quality", 1,
                        "0 = Alert during the day, "
                        "4 = Severe daytime sleepiness.",
                        "Alert during the day", "Severe daytime sleepiness"
                    )

                with card("mood"):
                    section_header(
                        "zap",
                        "Mood & Energy",
                        "Reflect on your emotional state and energy levels."
                    )

                    mood = scale_slider(
                        "mood", 1,
                        "0 = No noticeable problems, "
                        "4 = Severe mood or anxiety difficulties.",
                        "No noticeable problems",
                        "Severe mood or anxiety difficulties"
                    )

                    fatigue = scale_slider(
                        "fatigue", 1,
                        "0 = Normal energy, "
                        "4 = Severe fatigue or lack of energy.",
                        "Normal energy", "Severe fatigue or lack of energy"
                    )

                with card("motor"):
                    section_header(
                        "move",
                        "Motor Symptoms",
                        "Report any tremor you have noticed."
                    )

                    tremor = scale_slider(
                        "tremor", 0,
                        "0 = No tremor, 4 = Severe tremor.",
                        "No tremor", "Severe tremor"
                    )

                submitted = st.form_submit_button(
                    "Save My Check-in",
                    type="primary"
                )

        with guide_col:

            scale_bubbles = "".join(
                f"<div><i style='width:{12 + 6 * n}px;height:{12 + 6 * n}px;"
                f"background:{color}'></i>{n}</div>"
                for n, color in enumerate(
                    [C_ACCENT, C_SECONDARY, C_WARN, "#EDA27A", C_CORAL]
                )
            )

            html_block(f"""
                <div class="guide">
                    <div class="guide-title">How to rate each item</div>
                    <div class="guide-text">
                        Scale: 0 means no or minimal symptoms; 4 means severe
                        symptoms. These are simplified prototype input scales,
                        not a validated clinical questionnaire.
                    </div>
                    <div class="scale">{scale_bubbles}</div>
                    <div class="scale-cap"><span>No or minimal</span><span>Severe</span></div>
                </div>
                <div class="idcard">
                    <span>This check-in will be saved under</span>
                    <b>{esc(st.session_state.patient_id)}</b>
                </div>
            """)

        if submitted:

            features = {
                "sleep_duration": sleep_duration,
                "sleep_quality": sleep_quality,
                "mood": mood,
                "fatigue": fatigue,
                "tremor": tremor
            }

            try:

                feature_array = np.array([[
                    features[column]
                    for column in FEATURE_COLUMNS
                ]], dtype=float)

                if not np.isfinite(feature_array).all():
                    st.error("Please enter valid values for all fields.")

                else:

                    score = score_features(feature_array)

                    save_submission(
                        st.session_state.patient_id,
                        features,
                        score
                    )

                    st.session_state.latest_result = {
                        "score": score,
                        "features": features,
                        "date": datetime.now().strftime(
                            "%d %B %Y, %I:%M %p"
                        )
                    }

                    st.success("Your check-in has been saved.")

            except (ValueError, TypeError) as error:
                st.error(f"Unable to process the check-in: {error}")

            except sqlite3.Error as error:
                st.error(f"Unable to save your check-in: {error}")

    # Display result after successful submission.

    result = st.session_state.latest_result

    if result is not None:

        html_block('<div class="section-title" style="margin-top:1.6rem">Your check-in summary</div>')

        observation_panel(
            result["features"],
            result["score"],
            result["date"],
            heading="Experimental anomaly score"
        )

        html_block("""
            <div class="disclaimer" style="margin-top:1rem">
                <strong>Remember:</strong>
                A model score is not a diagnosis, a disease probability,
                or a measure of your overall health. Do not make medical
                decisions based on this result. If you have health concerns,
                speak with a qualified healthcare professional.
            </div>
        """)

        st.button(
            "See my trends",
            key="result_to_trends",
            on_click=go_to,
            args=("My Health Trends",)
        )


# ============================================================================
# PAGE 3: HEALTH TRENDS
# ============================================================================

elif page == "My Health Trends":

    page_header(
        "My health trends",
        "Review your previous check-ins and observe how your "
        "reported symptoms and experimental model scores change."
    )

    user_id = st.session_state.patient_id

    if not user_id:

        st.warning("Enter your tracking ID in the sidebar.")

    else:

        try:

            history = prepare_history(user_id)

            if history.empty:

                st.info(
                    "Your history is empty. Complete a daily check-in "
                    "to start tracking your observations."
                )

                st.button(
                    "Complete a check-in",
                    type="primary",
                    on_click=go_to,
                    args=("Daily Check-in",)
                )

            else:

                first_date = history["date"].min()
                last_date = history["date"].max()
                key_stamp = f"{user_id}_{first_date:%Y%m%d%H%M%S}_{last_date:%Y%m%d%H%M%S}"

                # ---------------- Filters ----------------
                with card("filters"):

                    filter_left, filter_right = st.columns([1, 1.6])

                    with filter_left:
                        picked = st.date_input(
                            "Date range",
                            value=(first_date.date(), last_date.date()),
                            min_value=first_date.date(),
                            max_value=max(last_date.date(), date.today()),
                            key=f"range_{key_stamp}"
                        )

                    with filter_right:
                        chosen = st.multiselect(
                            "Symptoms to show",
                            options=[FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
                            default=[FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
                            key="trend_symptoms"
                        )

                if isinstance(picked, (tuple, list)):
                    start = picked[0]
                    end = picked[1] if len(picked) > 1 else picked[0]
                else:
                    start = end = picked

                days = history["date"].dt.date

                filtered = history[
                    (days >= start) & (days <= end)
                ].sort_values("date")

                selected_columns = [
                    c for c in FEATURE_COLUMNS
                    if FEATURE_LABELS[c] in chosen
                ]

                st.write("")

                stat_cards([
                    ("Total check-ins", len(history), "All recorded entries"),
                    ("First check-in", f"{first_date:%d %b %Y}", f"{first_date:%I:%M %p}"),
                    ("Latest check-in", f"{last_date:%d %b %Y}", f"{last_date:%I:%M %p}"),
                    ("In selected range", len(filtered), f"{start:%d %b} to {end:%d %b %Y}")
                ])

                if filtered.empty:

                    st.info("No check-ins fall within the selected dates.")

                else:

                    with card("score"):
                        html_block("""
                            <div class="card-title">Model score history</div>
                            <div class="card-sub">Experimental Isolation Forest scores across your check-ins.</div>
                        """)
                        show_chart(make_score_trend(filtered))
                        st.caption(
                            "For Isolation Forest score_samples, lower values "
                            "indicate more unusual observations relative to "
                            "the model's learned reference patterns. "
                            "Scores are not disease probabilities."
                        )

                    st.write("")

                    with card("symptoms"):
                        html_block("""
                            <div class="card-title">Symptom trends</div>
                            <div class="card-sub">Use the selector above to focus on one or more symptoms.</div>
                        """)

                        if not selected_columns:
                            st.info("Select at least one symptom to see its trend.")
                        else:
                            show_chart(make_symptom_trend(filtered, selected_columns))
                            show_chart(make_symptom_heatmap(filtered, selected_columns))
                            st.caption(
                                "These charts show the values you reported in "
                                "each check-in. They do not establish disease "
                                "progression."
                            )

                    st.write("")

                    with card("history"):
                        html_block("""
                            <div class="card-title">Previous check-ins</div>
                            <div class="card-sub">Your recorded entries, newest first.</div>
                        """)

                        display_history = filtered.sort_values(
                            "date",
                            ascending=False
                        ).copy()

                        display_history["date"] = display_history["date"].dt.strftime(
                            "%d %b %Y, %I:%M %p"
                        )

                        display_history = display_history.rename(
                            columns={
                                "date": "Date",
                                "sleep_duration": "Sleep Problems",
                                "sleep_quality": "Daytime Sleepiness",
                                "mood": "Mood & Anxiety",
                                "fatigue": "Fatigue",
                                "tremor": "Tremor",
                                "score": "Model Score"
                            }
                        )

                        display_history["Model Score"] = display_history[
                            "Model Score"
                        ].round(4)

                        show_table(
                            display_history[
                                [
                                    "Date",
                                    "Sleep Problems",
                                    "Daytime Sleepiness",
                                    "Mood & Anxiety",
                                    "Fatigue",
                                    "Tremor",
                                    "Model Score"
                                ]
                            ]
                        )

        except sqlite3.Error as error:
            st.error(f"Unable to load your history: {error}")


# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

elif page == "About":

    page_header("About Neurovia", "How the prototype works, and what it can and cannot tell you.")

    html_block("""
        <div class="about-hero">
            Neurovia is a student-developed health-monitoring prototype
            designed to record selected daily observations and explore
            unusual patterns using machine learning.
        </div>
    """)

    html_block('<div class="section-title">How does it work?</div>')

    html_block("""
        <div class="info-card" style="margin-bottom:1rem">
            The application uses <b>Isolation Forest</b>, an unsupervised
            machine-learning algorithm. Instead of predicting a disease, the
            algorithm identifies observations that appear different from the
            patterns it learned during training. The process has three main
            steps:
        </div>
        <div class="grid-3">
            <div class="info-card"><div class="step-n">1</div>
                <b class="t">Input</b>You enter values for five daily health observations.</div>
            <div class="info-card"><div class="step-n">2</div>
                <b class="t">Preprocessing</b>The saved feature scaler transforms the inputs.</div>
            <div class="info-card"><div class="step-n">3</div>
                <b class="t">Scoring</b>The Isolation Forest generates an anomaly score.</div>
        </div>
    """)

    html_block('<div class="section-title">What does the app monitor?</div>')

    feature_cards = "".join(
        f"<div class='info-card'><span class='dot' "
        f"style='background:{SYMPTOM_COLORS[c]}'></span>"
        f"<b class='t'>{FEATURE_LABELS[c]}</b>{FEATURE_DESCRIPTIONS[c]}</div>"
        for c in FEATURE_COLUMNS
    )

    html_block(f"<div class='grid-feat'>{feature_cards}</div>")

    html_block("""
        <div class="dark-info">
            <div class="t">Understanding the model score</div>
            The application displays the value returned by the Isolation
            Forest <b>score_samples()</b> method.
            <div class="bl"><i></i><span>A lower score indicates a more unusual observation.</span></div>
            <div class="bl"><i></i><span>A higher score indicates a more typical observation relative to the model's learned patterns.</span></div>
            <div class="bl"><i></i><span>The score is not a percentage or probability.</span></div>
            <div class="bl"><i></i><span>The score cannot independently confirm or rule out Parkinson's disease.</span></div>
            <div class="fine">No clinical risk thresholds are applied in this version.</div>
        </div>
    """)

    html_block("""
        <div class="grid-3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
            <div class="info-card">
                <b class="t">About the training data</b>
                The original project uses Parkinson's Progression Markers
                Initiative (PPMI) data as its research reference.
                <br><br>
                Model performance and interpretation depend on the training
                dataset, preprocessing, selected features, and validation.
                The current prototype should not be assumed to generalize
                to every individual or population.
            </div>
            <div class="info-card">
                <b class="t">Privacy and limitations</b>
                <div class="bl"><i></i><span>Check-ins are stored in a local SQLite database.</span></div>
                <div class="bl"><i></i><span>The application does not provide secure user authentication.</span></div>
                <div class="bl"><i></i><span>A tracking ID is only used to group records.</span></div>
                <div class="bl"><i></i><span>Anyone with access to the database or a known tracking ID may be able to access records.</span></div>
                <div class="bl"><i></i><span>The prototype has not been established as a clinically validated medical device.</span></div>
            </div>
        </div>
        <div class="disclaimer">
            <strong>Medical disclaimer</strong><br>
            Neurovia is an educational and experimental monitoring
            prototype. It is not intended to diagnose, treat, prevent,
            or predict Parkinson's disease. It does not replace medical
            advice, examination, or diagnosis by a qualified healthcare
            professional. Always consult a healthcare professional
            regarding symptoms or health concerns.
        </div>
    """)


# ============================================================================
# FOOTER
# ============================================================================

html_block("""
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
        margin-top:2.2rem;padding-top:1rem;border-top:1px solid #E4ECE8;
        color:#8A9C96;font-size:.8rem">
        <span>Neurovia | Experimental health-monitoring prototype</span>
        <span>Isolation Forest · Local SQLite storage · Not a diagnostic tool</span>
    </div>
""")
