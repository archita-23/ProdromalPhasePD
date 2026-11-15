"""
Parkinson's Prodromal Detection System
Streamlit app: linear, single-task assessment with a clinical report view.

Model, scoring thresholds and database schema are unchanged from the original.
Requires Streamlit 1.39 or newer (uses keyed containers/widgets for styling hooks).
Pair with .streamlit/config.toml (light theme, teal primary colour).
"""

import html
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Parkinson's Prodromal Detection",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# DESIGN SYSTEM
# One family (Public Sans), one canvas, one accent (teal, call-to-action only).
# Amber is reserved for notices. Motion: 150ms linear fades only.
# ============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --canvas: #F8F9FA;
    --ink: #1A1F2C;
    --muted: #4B5563;
    --rule: #D9DEE5;
    --control: #5B6475;
    --track: #E3E8EE;
    --cta: #0F766E;
    --cta-hover: #0B5C56;
    --focus: #1E3A8A;
    --amber-bg: #FEF3D7;
    --amber-line: #E0B252;
    --amber-ink: #4A2F00;
}

/* ---------- Canvas and type ---------- */
.stApp {
    background: var(--canvas);
    color: var(--ink);
}
.stApp, .stApp *:not([data-testid="stIconMaterial"]) {
    font-family: 'Public Sans', 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stStatusWidget"] { display: none; }
[data-testid="stHeaderActionElements"] { display: none; }

[data-testid="stMainBlockContainer"], .block-container {
    max-width: 46rem !important;
    padding: 3rem 1.5rem 6rem !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1.125rem;
    line-height: 1.6;
    color: var(--ink);
}
[data-testid="stMarkdownContainer"] ul { padding-left: 1.4rem; }
[data-testid="stMarkdownContainer"] li { margin-bottom: .5rem; }

.stApp h1 {
    font-size: 2.5rem !important;
    line-height: 1.2 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: var(--ink) !important;
    padding: 0 !important;
    margin: 0 0 1rem !important;
}
.stApp h2 {
    font-size: 1.75rem !important;
    line-height: 1.3 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: var(--ink) !important;
    padding: 0 !important;
    margin: 0 0 1rem !important;
}
.stApp h3 {
    font-size: 1.375rem !important;
    line-height: 1.4 !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    padding: 0 !important;
    margin: 0 0 .5rem !important;
}

.brand {
    font-size: 1rem !important;
    font-weight: 700;
    letter-spacing: 0;
    color: var(--ink);
    margin: 0 0 .5rem;
}
.hint {
    color: var(--muted) !important;
    margin: 0 0 1.25rem;
}
.stepline {
    font-size: 1rem !important;
    font-weight: 600;
    color: var(--muted) !important;
    margin: 2.5rem 0 .75rem;
}
.progress {
    display: flex;
    gap: 4px;
    margin-bottom: 2.5rem;
}
.progress span {
    flex: 1;
    height: 6px;
    background: var(--rule);
}
.progress span.done { background: #334155; }

hr.rule,
[data-testid="stMarkdownContainer"] hr {
    border: 0;
    border-top: 1px solid var(--rule);
    margin: 2.75rem 0;
}

/* ---------- Section navigation (plain text tabs) ---------- */
.st-key-nav div[role="radiogroup"] {
    gap: 0;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--rule);
}
.st-key-nav div[role="radiogroup"] > label {
    position: relative;
    min-height: 56px;
    margin: 0 0 -1px;
    padding: 0 1.75rem 0 0;
    display: flex;
    align-items: center;
    border-bottom: 4px solid transparent;
    cursor: pointer;
    transition: border-color 150ms linear;
}
.st-key-nav div[role="radiogroup"] > label > div:first-child {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    overflow: hidden;
}
.st-key-nav div[role="radiogroup"] > label p {
    font-size: 1.125rem;
    font-weight: 500;
    color: var(--muted);
}
.st-key-nav div[role="radiogroup"] > label:hover {
    border-bottom-color: var(--control);
}
.st-key-nav div[role="radiogroup"] > label:hover p { color: var(--ink); }
.st-key-nav div[role="radiogroup"] > label:has(input:checked) {
    border-bottom-color: var(--ink);
}
.st-key-nav div[role="radiogroup"] > label:has(input:checked) p {
    font-weight: 700;
    color: var(--ink);
}
.st-key-nav div[role="radiogroup"] > label:has(input:focus-visible) {
    outline: 4px solid var(--focus);
    outline-offset: 2px;
}

/* ---------- Rating scale (large radio targets) ---------- */
[class*="st-key-scale_"] div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: .75rem;
}
[class*="st-key-scale_"] div[role="radiogroup"] > label {
    flex: 1 1 8rem;
    min-height: 60px;
    margin: 0;
    padding: .5rem 1rem;
    display: flex;
    align-items: center;
    background: #FFFFFF;
    border: 2px solid var(--control);
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 150ms linear, border-color 150ms linear;
}
[class*="st-key-scale_"] div[role="radiogroup"] > label p {
    font-size: 1.125rem;
    line-height: 1.3;
    margin: 0;
}
[class*="st-key-scale_"] div[role="radiogroup"] > label:hover {
    background: #EEF1F4;
    border-color: var(--ink);
}
[class*="st-key-scale_"] div[role="radiogroup"] > label:has(input:checked) {
    background: #E3E8EE;
    border-color: var(--ink);
    box-shadow: inset 0 0 0 1px var(--ink);
}
[class*="st-key-scale_"] div[role="radiogroup"] > label:has(input:focus-visible) {
    outline: 4px solid var(--focus);
    outline-offset: 3px;
}

/* ---------- Buttons (56px minimum height) ---------- */
.stButton > button {
    width: 100%;
    min-height: 56px;
    padding: 0 1.75rem;
    border-radius: 6px;
    border: 2px solid var(--control);
    background: #FFFFFF;
    color: var(--ink);
    font-size: 1.125rem;
    font-weight: 600;
    transition: background-color 150ms linear, border-color 150ms linear;
}
.stButton > button p {
    color: inherit;
    font-size: inherit;
    font-weight: inherit;
    margin: 0;
}
.stButton > button:hover {
    background: #EEF1F4;
    border-color: var(--ink);
    color: var(--ink);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--cta);
    border-color: var(--cta);
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--cta-hover);
    border-color: var(--cta-hover);
    color: #FFFFFF;
}
.stButton > button:focus-visible {
    outline: 4px solid var(--focus);
    outline-offset: 3px;
    box-shadow: none;
}

/* ---------- Text inputs ---------- */
.stTextInput label p {
    font-size: 1.125rem;
    font-weight: 600;
}
.stTextInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="base-input"] {
    background: #FFFFFF;
    border-radius: 6px;
}
.stTextInput div[data-baseweb="input"] {
    border: 2px solid var(--control);
    min-height: 56px;
    transition: border-color 150ms linear;
}
.stTextInput div[data-baseweb="input"]:hover { border-color: var(--ink); }
.stTextInput div[data-baseweb="input"]:focus-within {
    border-color: var(--focus);
    outline: 4px solid var(--focus);
    outline-offset: 2px;
}
.stTextInput input {
    font-size: 1.125rem;
    min-height: 52px;
    color: var(--ink);
}

/* ---------- File upload zone (for voice or movement recordings) ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF;
    border: 2px dashed var(--control);
    border-radius: 6px;
    min-height: 140px;
    padding: 2rem;
    transition: border-color 150ms linear, background-color 150ms linear;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: #EEF1F4;
    border-color: var(--ink);
}
[data-testid="stFileUploaderDropzone"]:focus-within {
    border-style: solid;
    border-color: var(--focus);
    outline: 4px solid var(--focus);
    outline-offset: 2px;
}
[data-testid="stFileUploaderDropzone"] button {
    min-height: 56px;
    padding: 0 1.5rem;
    font-size: 1.125rem;
    font-weight: 600;
}

/* ---------- Report tables ---------- */
.table-wrap { overflow-x: auto; }
table.report {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
    line-height: 1.5;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
}
table.report thead th {
    text-align: left;
    font-weight: 600;
    font-size: .9375rem;
    color: var(--muted);
    padding: .5rem 1rem .75rem 0;
    border-bottom: 2px solid var(--ink);
    white-space: nowrap;
}
table.report tbody th {
    text-align: left;
    font-weight: 500;
    padding: 1rem 1rem 1rem 0;
    border-bottom: 1px solid var(--rule);
}
table.report td {
    padding: 1rem 1rem 1rem 0;
    border-bottom: 1px solid var(--rule);
    vertical-align: middle;
}
table.report td.num, table.report th.num { text-align: right; padding-right: 1.25rem; }
table.report td.scale { width: 38%; }
table.report.meta tbody th { width: 11rem; color: var(--muted); }
table.report.meta thead { display: none; }

.bar {
    height: 10px;
    min-width: 96px;
    background: var(--track);
}
.bar span {
    display: block;
    height: 100%;
    background: #334155;
}

/* ---------- Score meter ---------- */
.score-readout {
    font-size: 1rem !important;
    color: var(--muted) !important;
    margin: 1.75rem 0 .75rem;
}
.score-readout strong {
    font-size: 1.75rem;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    margin-left: .25rem;
}
.meter { position: relative; padding-top: 10px; }
.meter .zones { display: flex; height: 16px; }
.meter .z-high { background: #F2D9A3; }
.meter .z-mod  { background: #F8E9C6; }
.meter .z-norm { background: var(--track); }
.meter .marker {
    position: absolute;
    top: 0;
    width: 6px;
    height: 36px;
    margin-left: -3px;
    background: var(--ink);
}
.meter-labels { display: flex; margin-top: .6rem; }
.meter-labels span {
    font-size: .875rem;
    line-height: 1.4;
    color: var(--muted);
    padding-right: .5rem;
}

/* ---------- Result line and notices ---------- */
.result-line {
    font-size: 1.5rem !important;
    font-weight: 700;
    margin: 0 0 .5rem;
}
.note {
    background: var(--amber-bg);
    border: 1px solid var(--amber-line);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
    color: var(--amber-ink);
    font-size: 1.0625rem;
    line-height: 1.6;
}
.small {
    font-size: 1rem !important;
    color: var(--muted) !important;
}

/* ---------- Motion: one 150ms linear fade per step, nothing else ---------- */
@keyframes fade150 { from { opacity: 0; } to { opacity: 1; } }
[class*="st-key-stage_"] { animation: fade150 150ms linear; }
@media (prefers-reduced-motion: reduce) {
    [class*="st-key-stage_"] { animation: none; }
    .stApp *, .stApp *::before, .stApp *::after { transition: none !important; }
}
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (cached so they don't reload)
# ============================================================================


@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    try:
        with open('isolation_forest_model.pkl', 'rb') as f:
            iso_forest = pickle.load(f)
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return iso_forest, scaler
    except FileNotFoundError:
        st.error("Models not found. Run preprocessing first.")
        return None, None


iso_forest, scaler = load_models()

if iso_forest is None:
    st.stop()

# ============================================================================
# DATABASE SETUP (unchanged)
# ============================================================================


def init_db():
    """Create SQLite database for storing submissions"""
    conn = sqlite3.connect('submissions.db')
    c = conn.cursor()
    c.execute('''
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
    ''')
    conn.commit()
    conn.close()


def save_submission(user_id, features_dict, score, risk):
    """Save submission to database"""
    conn = sqlite3.connect('submissions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO submissions
        (user_id, date, sleep_duration, sleep_quality, mood, fatigue, tremor, score, risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        features_dict['sleep_duration'],
        features_dict['sleep_quality'],
        features_dict['mood'],
        features_dict['fatigue'],
        features_dict['tremor'],
        score,
        risk
    ))
    conn.commit()
    conn.close()


def get_history(user_id):
    """Get user's submission history"""
    conn = sqlite3.connect('submissions.db')
    df = pd.read_sql_query(
        'SELECT * FROM submissions WHERE user_id = ? ORDER BY date DESC',
        conn,
        params=(user_id,)
    )
    conn.close()
    return df


init_db()

# ============================================================================
# SCORING (unchanged thresholds)
# ============================================================================


def score_features(features_array):
    """Score features and categorize risk"""
    normalized = scaler.transform(features_array)
    score = iso_forest.score_samples(normalized)[0]

    if score < -0.5:
        risk = "High Risk"
    elif score < -0.2:
        risk = "Moderate Risk"
    else:
        risk = "Normal"

    return score, risk


# ============================================================================
# CONTENT
# ============================================================================

# Order matters: this is the column order the model was trained on.
FEATURE_ORDER = ["sleep_duration", "sleep_quality", "mood", "fatigue", "tremor"]

FEATURES = {
    "sleep_duration": {
        "label": "Sleep problems",
        "question": "How severe are your sleep problems?",
        "hint": "Rate any sleep disturbances.",
    },
    "sleep_quality": {
        "label": "Daytime sleepiness",
        "question": "How severe is your daytime sleepiness?",
        "hint": "Rate excessive sleepiness during the day.",
    },
    "fatigue": {
        "label": "Fatigue",
        "question": "How severe is your fatigue?",
        "hint": "Rate low energy levels and lack of motivation.",
    },
    "mood": {
        "label": "Mood and anxiety",
        "question": "How severe are your low mood or anxiety?",
        "hint": "Rate feelings of depression and anxiety.",
    },
    "tremor": {
        "label": "Tremor",
        "question": "How severe is your tremor?",
        "hint": "Rate how visible any tremor is.",
    },
}

SCALE_WORDS = {0: "None", 1: "Slight", 2: "Mild", 3: "Moderate", 4: "Severe"}

STEP_TITLES = ["Before you begin", "Sleep", "Energy and mood", "Movement", "Review your answers"]
STEP_QUESTIONS = {
    1: ["sleep_duration", "sleep_quality"],
    2: ["fatigue", "mood"],
    3: ["tremor"],
}
TOTAL_STEPS = len(STEP_TITLES)

LEVELS = {
    "High Risk": {
        "label": "High risk",
        "summary": "Your combination of ratings is unusual compared with the reference data and needs attention.",
        "alert": "Please arrange to speak with a healthcare provider about these results.",
        "steps": [
            "Consult with a healthcare provider.",
            "Monitor your symptoms closely.",
            "Track how your ratings change over the next few days.",
        ],
    },
    "Moderate Risk": {
        "label": "Moderate risk",
        "summary": "Some of your ratings show an atypical pattern compared with the reference data.",
        "alert": "Keep an eye on these results and repeat the check-in within a week.",
        "steps": [
            "Monitor your symptoms for changes.",
            "Keep track of your daily patterns.",
            "Follow up with another check-in in one week.",
        ],
    },
    "Normal": {
        "label": "Normal",
        "summary": "Your ratings fall within the typical range of the reference data.",
        "alert": None,
        "steps": [
            "Continue regular monitoring.",
            "Maintain a healthy lifestyle.",
            "Schedule routine check-ups.",
        ],
    },
}

# ============================================================================
# HTML HELPERS (compact, unindented so Markdown never treats them as code)
# ============================================================================

esc = html.escape


def rule():
    st.markdown('<hr class="rule">', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note" role="alert">{esc(text)}</div>', unsafe_allow_html=True)


def progress(step):
    n = step + 1
    segs = "".join(f'<span class="{"done" if i < n else ""}"></span>' for i in range(TOTAL_STEPS))
    st.markdown(
        f'<p class="stepline">Step {n} of {TOTAL_STEPS}: {esc(STEP_TITLES[step])}</p>'
        f'<div class="progress" role="progressbar" aria-label="Assessment progress" '
        f'aria-valuemin="1" aria-valuemax="{TOTAL_STEPS}" aria-valuenow="{n}">{segs}</div>',
        unsafe_allow_html=True,
    )


def ratings_table(values, average=False):
    """Symptom / rating / simple bar. `values` maps feature key -> 0-4 number."""
    rows = []
    for key in FEATURE_ORDER:
        v = float(values[key])
        label = FEATURES[key]["label"]
        text = f"{v:.1f} of 4" if average else f"{int(v)} of 4, {SCALE_WORDS[int(v)].lower()}"
        pct = max(0.0, min(v / 4 * 100, 100.0))
        rows.append(
            f'<tr><th scope="row">{esc(label)}</th><td class="num">{text}</td>'
            f'<td class="scale"><div class="bar" role="img" aria-label="{esc(label)}: {v:.1f} out of 4">'
            f'<span style="width:{pct:.0f}%"></span></div></td></tr>'
        )
    return (
        '<div class="table-wrap"><table class="report">'
        '<thead><tr><th scope="col">Symptom</th><th scope="col" class="num">Rating</th>'
        '<th scope="col">Scale, 0 to 4</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def meta_table(pairs):
    rows = "".join(f'<tr><th scope="row">{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in pairs)
    return (
        '<div class="table-wrap"><table class="report meta">'
        f'<thead><tr><th>Item</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table></div>'
    )


def score_meter(score):
    lo, hi = -1.0, 0.5
    pct = min(max((score - lo) / (hi - lo), 0.0), 1.0) * 100
    return (
        f'<p class="score-readout">Anomaly score<strong>{score:.3f}</strong></p>'
        f'<div class="meter" role="meter" aria-label="Anomaly score" aria-valuemin="{lo}" '
        f'aria-valuemax="{hi}" aria-valuenow="{score:.3f}">'
        '<div class="zones"><span class="z-high" style="width:33.33%"></span>'
        '<span class="z-mod" style="width:20%"></span>'
        '<span class="z-norm" style="width:46.67%"></span></div>'
        f'<div class="marker" style="left:{pct:.2f}%"></div></div>'
        '<div class="meter-labels">'
        '<span style="width:33.33%">High risk<br>below −0.5</span>'
        '<span style="width:20%">Moderate<br>−0.5 to −0.2</span>'
        '<span style="width:46.67%">Normal<br>above −0.2</span></div>'
    )


def thresholds_table():
    return (
        '<div class="table-wrap"><table class="report">'
        '<thead><tr><th scope="col">Result</th><th scope="col">Score</th><th scope="col">Meaning</th></tr></thead>'
        '<tbody>'
        '<tr><th scope="row">High risk</th><td>Below −0.5</td><td>Unusual pattern that needs medical attention.</td></tr>'
        '<tr><th scope="row">Moderate risk</th><td>−0.5 to −0.2</td><td>Atypical pattern to monitor.</td></tr>'
        '<tr><th scope="row">Normal</th><td>Above −0.2</td><td>Typical pattern in the reference data.</td></tr>'
        '</tbody></table></div>'
    )


# ============================================================================
# SESSION STATE
# ============================================================================

ss = st.session_state
ss.setdefault("nav", "Check-in")
ss.setdefault("step", 0)
ss.setdefault("answers", {})
ss.setdefault("user_id", "patient_001")
ss.setdefault("result", None)


def start_over():
    ss.step = 0
    ss.answers = {}
    ss.result = None


def show_history():
    ss.nav = "History"


# ============================================================================
# CHECK-IN: one task per screen
# ============================================================================


def scale_question(key):
    meta = FEATURES[key]
    st.markdown(f"### {meta['question']}")
    st.markdown(f'<p class="hint">{esc(meta["hint"])}</p>', unsafe_allow_html=True)
    choice = st.radio(
        meta["question"],
        options=list(SCALE_WORDS.keys()),
        index=ss.answers.get(key),
        format_func=lambda v: f"**{v}**  {SCALE_WORDS[v]}",
        horizontal=True,
        key=f"scale_{key}",
        label_visibility="collapsed",
    )
    ss.answers[key] = choice


def render_checkin():
    if ss.result is not None:
        render_report(ss.result)
        return

    step = ss.step
    message = None

    with st.container(key=f"stage_{step}"):
        progress(step)
        st.markdown(f"# {STEP_TITLES[step]}")

        if step == 0:
            st.markdown(
                "You will rate five symptoms from 0 (none) to 4 (severe), based on how you have "
                "felt recently. Take as much time as you need. Nothing is saved until the last step."
            )
            st.markdown(
                '<div class="table-wrap"><table class="report"><thead><tr>'
                + "".join(f'<th scope="col" class="num">{v}</th>' for v in SCALE_WORDS)
                + "</tr></thead><tbody><tr>"
                + "".join(f'<td class="num">{w}</td>' for w in SCALE_WORDS.values())
                + "</tr></tbody></table></div>",
                unsafe_allow_html=True,
            )
            rule()
            uid = st.text_input("Your ID", value=ss.user_id, key="uid_widget")
            st.markdown(
                '<p class="hint">Results are saved under this ID so you can follow changes over time. '
                "Use the same ID for every check-in.</p>",
                unsafe_allow_html=True,
            )
            ss.user_id = uid.strip()

        elif step in STEP_QUESTIONS:
            keys = STEP_QUESTIONS[step]
            for i, key in enumerate(keys):
                if i > 0:
                    rule()
                scale_question(key)

        else:  # review
            st.markdown("Check your ratings. Select Back to change any of them.")
            st.markdown(
                ratings_table({k: ss.answers[k] for k in FEATURE_ORDER}),
                unsafe_allow_html=True,
            )

        rule()

        left, right = st.columns(2)
        go_back = left.button("Back", key=f"back_{step}") if step > 0 else False
        next_label = "Generate report" if step == TOTAL_STEPS - 1 else "Continue"
        go_next = right.button(next_label, key=f"next_{step}", type="primary")

        if go_back:
            ss.step -= 1
            st.rerun()

        if go_next:
            if step == 0 and not ss.user_id:
                message = "Enter an ID to continue."
            elif step in STEP_QUESTIONS and any(ss.answers.get(k) is None for k in STEP_QUESTIONS[step]):
                message = "Choose one answer for each question to continue."
            elif step == TOTAL_STEPS - 1:
                features = np.array([[ss.answers[k] for k in FEATURE_ORDER]])
                score, risk = score_features(features)
                save_submission(ss.user_id, {k: ss.answers[k] for k in FEATURE_ORDER}, score, risk)
                ss.result = {
                    "score": float(score),
                    "risk": risk,
                    "answers": {k: ss.answers[k] for k in FEATURE_ORDER},
                    "user_id": ss.user_id,
                    "when": datetime.now(),
                }
                st.rerun()
            else:
                ss.step += 1
                st.rerun()

        if message:
            note(message)


# ============================================================================
# REPORT
# ============================================================================


def render_report(res):
    level = LEVELS[res["risk"]]

    with st.container(key="stage_report"):
        st.markdown("# Assessment report")
        st.markdown(
            meta_table([
                ("ID", res["user_id"]),
                ("Date", res["when"].strftime("%d %B %Y, %H:%M")),
                ("Method", "Isolation Forest anomaly detection"),
            ]),
            unsafe_allow_html=True,
        )

        rule()
        st.markdown("## Result")
        st.markdown(f'<p class="result-line">{esc(level["label"])}</p>', unsafe_allow_html=True)
        st.markdown(level["summary"])
        if level["alert"]:
            note(level["alert"])
        st.markdown(score_meter(res["score"]), unsafe_allow_html=True)

        rule()
        st.markdown("## Reported symptoms")
        st.markdown(ratings_table(res["answers"]), unsafe_allow_html=True)

        rule()
        st.markdown("## How to read the score")
        st.markdown(thresholds_table(), unsafe_allow_html=True)

        rule()
        st.markdown("## Suggested next steps")
        st.markdown("\n".join(f"- {s}" for s in level["steps"]))

        rule()
        st.markdown("## Limits of this report")
        st.markdown(
            "This is a monitoring tool, not a diagnostic tool. It compares your ratings with "
            "patterns in research data and cannot confirm or rule out Parkinson's disease. "
            "Always consult a healthcare provider for medical decisions."
        )

        rule()
        left, right = st.columns(2)
        left.button("View history", key="report_history", on_click=show_history)
        right.button("Start a new check-in", key="report_restart", type="primary", on_click=start_over)


# ============================================================================
# HISTORY
# ============================================================================


def render_history():
    with st.container(key="stage_history"):
        st.markdown("# History")
        user_id = st.text_input("Your ID", value=ss.user_id, key="hist_id").strip()

        if not user_id:
            return

        history = get_history(user_id)

        if len(history) == 0:
            rule()
            st.markdown("No check-ins found for this ID. Complete a check-in and it will appear here.")
            return

        rule()
        st.markdown("## Summary")
        st.markdown(
            meta_table([
                ("Check-ins", str(len(history))),
                ("Average score", f"{history['score'].mean():.3f}"),
                ("High-risk results", str(int((history['risk'] == 'High Risk').sum()))),
            ]),
            unsafe_allow_html=True,
        )

        rule()
        st.markdown("## Recent check-ins")
        rows = []
        for _, r in history.head(20).iterrows():
            rows.append(
                f'<tr><th scope="row">{esc(str(r["date"]))}</th>'
                f'<td>{esc(LEVELS.get(r["risk"], {"label": str(r["risk"])})["label"])}</td>'
                f'<td class="num">{r["score"]:.3f}</td>'
                + "".join(f'<td class="num">{int(r[k])}</td>' for k in FEATURE_ORDER)
                + "</tr>"
            )
        heads = "".join(f'<th scope="col" class="num">{esc(FEATURES[k]["label"])}</th>' for k in FEATURE_ORDER)
        st.markdown(
            '<div class="table-wrap"><table class="report"><thead><tr>'
            '<th scope="col">Date</th><th scope="col">Result</th><th scope="col" class="num">Score</th>'
            f'{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )

        rule()
        st.markdown("## Score over time")
        st.markdown(
            '<p class="hint">Lower scores are more unusual. Dotted lines mark the moderate and high-risk thresholds.</p>',
            unsafe_allow_html=True,
        )
        chart = history[['date', 'score']].sort_values('date')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart['date'], y=chart['score'],
            mode="lines+markers",
            line=dict(color="#1A1F2C", width=2),
            marker=dict(size=9, color="#1A1F2C"),
            hovertemplate="%{x}<br>Score %{y:.3f}<extra></extra>",
            showlegend=False,
        ))
        fig.add_hline(y=-0.2, line_dash="dot", line_color="#8A5A00",
                      annotation_text="Moderate threshold (−0.2)", annotation_position="top left",
                      annotation_font_color="#4A2F00")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="#8A5A00",
                      annotation_text="High-risk threshold (−0.5)", annotation_position="bottom left",
                      annotation_font_color="#4A2F00")
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Public Sans, Segoe UI, Arial, sans-serif", size=14, color="#1A1F2C"),
            xaxis=dict(type="category", title="Date", showgrid=False, linecolor="#5B6475"),
            yaxis=dict(title="Anomaly score", gridcolor="#D9DEE5", zeroline=False),
            transition=dict(duration=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        rule()
        st.markdown("## Average ratings")
        avg = {k: history[k].mean() for k in FEATURE_ORDER}
        st.markdown(ratings_table(avg, average=True), unsafe_allow_html=True)


# ============================================================================
# ABOUT
# ============================================================================


def render_about():
    with st.container(key="stage_about"):
        st.markdown("# About this system")
        st.markdown(
            "This system uses Isolation Forest, a machine learning method trained on data from the "
            "Parkinson's Progression Markers Initiative (PPMI), to detect unusual patterns in daily "
            "health ratings."
        )

        rule()
        st.markdown("## What is rated")
        st.markdown(
            "- **Sleep problems:** sleep disturbances, rated 0 to 4.\n"
            "- **Daytime sleepiness:** excessive sleepiness, rated 0 to 4.\n"
            "- **Mood and anxiety:** depression and anxiety level, rated 0 to 4.\n"
            "- **Fatigue:** energy levels and apathy, rated 0 to 4.\n"
            "- **Tremor:** visible tremor severity, rated 0 to 4."
        )

        rule()
        st.markdown("## How results are grouped")
        st.markdown(thresholds_table(), unsafe_allow_html=True)

        rule()
        st.markdown("## Important")
        st.markdown(
            "This is a monitoring tool, not a diagnostic tool. Always consult a healthcare provider "
            "for medical decisions."
        )

        rule()
        st.markdown("## Your data")
        st.markdown(
            "Check-in results are stored in a local database on this system and are not sent to "
            "other servers."
        )


# ============================================================================
# MAIN
# ============================================================================

st.markdown('<p class="brand">Parkinson’s Prodromal Detection System</p>', unsafe_allow_html=True)
page = st.radio(
    "Section",
    ["Check-in", "History", "About"],
    horizontal=True,
    key="nav",
    label_visibility="collapsed",
)

if page == "Check-in":
    render_checkin()
elif page == "History":
    render_history()
else:
    render_about()

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown(
    '<p class="small">Screening aid, not a diagnosis. Model: Isolation Forest. Reference data: PPMI.</p>',
    unsafe_allow_html=True,
)
