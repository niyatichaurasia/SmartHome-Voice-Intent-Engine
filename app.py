"""
VoiceIntent — Real-Time Speech Command Recognition
Clean rebuild. No navbar hacks. Impressive for recruiters and non-tech users.
"""

import streamlit as st
import numpy as np
import time
from pathlib import Path

st.set_page_config(
    page_title="VoiceIntent",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":       "#0d0f1a",
        "card":     "#141826",
        "card2":    "#1a2035",
        "border":   "#252d48",
        "border2":  "#2e3a5e",
        "accent":   "#818cf8",   # indigo-400 — modern, not neon
        "accent_bg":"#1e2240",
        "tx":       "#f1f2f9",   # near-white, readable
        "tx2":      "#8b92b8",   # muted — secondary info
        "tx3":      "#4a5070",   # very muted — timestamps, labels
        "ok":       "#4ade80",   # green
        "ok_bg":    "#0f2a1a",
        "err":      "#f87171",
        "err_bg":   "#2a0f0f",
        "swap":     "light",
        "swap_lbl": "☀",
        "mfcc":     "plasma",
        "af":       "invert(1) hue-rotate(180deg)",
    },
    "light": {
        "bg":       "#f8f9fc",
        "card":     "#ffffff",
        "card2":    "#f3f4f8",
        "border":   "#e2e4ee",
        "border2":  "#c8cce0",
        "accent":   "#4f46e5",   # indigo-600 — readable on white
        "accent_bg":"#eef2ff",
        "tx":       "#0f1117",   # near-black
        "tx2":      "#4b5280",   # readable muted
        "tx3":      "#9499be",   # labels
        "ok":       "#16a34a",
        "ok_bg":    "#f0fdf4",
        "err":      "#dc2626",
        "err_bg":   "#fef2f2",
        "swap":     "dark",
        "swap_lbl": "☽",
        "mfcc":     "viridis",
        "af":       "none",
    },
}

def T():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    return THEMES[st.session_state["theme"]]


# ── CSS — clean, no hacks ─────────────────────────────────────────────────────
def inject_css():
    t = T()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {{
    --bg:    {t['bg']};
    --card:  {t['card']};
    --card2: {t['card2']};
    --bd:    {t['border']};
    --bd2:   {t['border2']};
    --ac:    {t['accent']};
    --acbg:  {t['accent_bg']};
    --tx:    {t['tx']};
    --tx2:   {t['tx2']};
    --tx3:   {t['tx3']};
    --ok:    {t['ok']};
    --okbg:  {t['ok_bg']};
    --err:   {t['err']};
    --errbg: {t['err_bg']};
    --r:     10px;
    --font:  'Plus Jakarta Sans', system-ui, sans-serif;
    --mono:  'JetBrains Mono', monospace;
}}

/* ── Hard reset ── */
*, *::before, *::after {{
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
}}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .stApp {{
    background: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--tx) !important;
}}

/* Kill all Streamlit chrome */
[data-testid="stHeader"], #MainMenu,
footer, header, [data-testid="stToolbar"] {{
    display: none !important;
}}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* Remove padding from block container */
.block-container {{
    padding: 0 !important;
    max-width: 800px !important;
}}

/* Streamlit adds gap between elements — kill it */
[data-testid="stVerticalBlock"] > * + * {{
    margin-top: 0 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--bd2); border-radius: 2px; }}

/* ════════════════════════════════
   TEXT — strict hierarchy
   Every element has an explicit role
   ════════════════════════════════ */

/* All text defaults */
p, span, div, li, label, td, th, a, button {{
    font-family: var(--font) !important;
    color: var(--tx) !important;
}}

/* Streamlit markdown */
.stMarkdown p, [data-testid="stMarkdownContainer"] p {{
    font-size: 15px !important;
    color: var(--tx2) !important;
    line-height: 1.7 !important;
    margin: 0 0 4px !important;
}}
[data-testid="stCaptionContainer"] p {{
    font-size: 13px !important;
    color: var(--tx3) !important;
    margin: 0 !important;
}}
[data-testid="stSpinner"] p {{
    font-size: 14px !important;
    color: var(--tx2) !important;
}}
[data-testid="stAlert"] {{
    background: var(--card) !important;
    border-color: var(--bd) !important;
    border-radius: var(--r) !important;
}}
[data-testid="stAlert"] p {{
    font-size: 14px !important;
    color: var(--tx) !important;
}}
code {{
    font-family: var(--mono) !important;
    font-size: 12px !important;
    background: var(--card2) !important;
    color: var(--ac) !important;
    border: 1px solid var(--bd) !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
}}

/* ════════════════════════════════
   CUSTOM HTML COMPONENTS
   ════════════════════════════════ */

/* App shell — full page wrapper */
.vi-shell {{
    min-height: 100vh;
    background: var(--bg);
    padding: 0;
}}

/* ── App header ── */
.vi-header {{
    padding: 48px 0 0;
    text-align: center;
    margin-bottom: 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}}
.vi-app-name {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ac) !important;
    font-family: var(--mono) !important;
    margin-bottom: 16px;
    display: block;
}}
.vi-headline {{
    font-size: 3rem;
    font-weight: 800;
    color: var(--tx) !important;
    letter-spacing: -0.035em;
    line-height: 1.05;
    margin-bottom: 16px;
}}
.vi-subline {{
    font-size: 16px;
    color: var(--tx2) !important;
    line-height: 1.65;
    max-width: 520px;
    margin: 0 auto 20px;
    text-align: center;
}}

/* ── Status badge ── */
.vi-status {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    font-family: var(--mono) !important;
    letter-spacing: 0.02em;
    margin-bottom: 0;
}}
.vi-status.ok  {{
    background: var(--okbg);
    color: var(--ok) !important;
    border: 1px solid {t['ok']}30;
}}
.vi-status.err {{
    background: var(--errbg);
    color: var(--err) !important;
    border: 1px solid {t['err']}30;
}}
.vi-status.ld  {{
    background: var(--acbg);
    color: var(--ac) !important;
    border: 1px solid {t['accent']}30;
}}

/* ── Divider ── */
.vi-line {{
    height: 1px;
    background: var(--bd);
    margin: 32px 0;
    border: none;
}}
.vi-line-sm {{
    height: 1px;
    background: var(--bd);
    margin: 20px 0;
    border: none;
}}

/* ── Section label ── */
.vi-eyebrow {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--tx3) !important;
    font-family: var(--mono) !important;
    display: block;
    margin-bottom: 10px;
}}

/* ── Cards ── */
.vi-card {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 16px;
}}
.vi-card-sm {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}}

/* ── Input card with mode toggle ── */
.vi-input-card {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 16px;
}}
.vi-input-header {{
    display: flex;
    border-bottom: 1px solid var(--bd);
    background: var(--card2);
}}
.vi-mode-tab {{
    flex: 1;
    padding: 14px 20px;
    font-size: 14px;
    font-weight: 500;
    color: var(--tx2) !important;
    cursor: pointer;
    text-align: center;
    transition: background 0.15s, color 0.15s;
    user-select: none;
    border: none;
    background: transparent;
    font-family: var(--font) !important;
}}
.vi-mode-tab:hover {{ color: var(--tx) !important; }}
.vi-mode-tab.on {{
    background: var(--card);
    color: var(--tx) !important;
    font-weight: 600;
    box-shadow: inset 0 -2px 0 var(--ac);
}}
.vi-mode-tab:first-child {{ border-right: 1px solid var(--bd); }}
.vi-input-body {{ padding: 24px 28px 28px; }}

/* ── Result card ── */
.vi-result {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}}
.vi-result::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--ac);
    border-radius: 14px 14px 0 0;
}}
.vi-intent-tag {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ac) !important;
    font-family: var(--mono) !important;
    display: block;
    margin-bottom: 8px;
}}
.vi-intent-name {{
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--tx) !important;
    font-family: var(--mono) !important;
    line-height: 1.25;
    margin-bottom: 24px;
    word-break: break-word;
}}
.vi-conf-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}
.vi-conf-lbl {{
    font-size: 13px;
    font-weight: 500;
    color: var(--tx2) !important;
}}
.vi-conf-num {{
    font-size: 22px;
    font-weight: 800;
    color: var(--ok) !important;
    font-family: var(--mono) !important;
    letter-spacing: -0.02em;
}}
.vi-bar {{
    height: 5px;
    background: var(--card2);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 28px;
    border: 1px solid var(--bd);
}}
.vi-bar-fill {{
    height: 100%;
    border-radius: 3px;
    background: var(--ac);
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}}
.vi-top5-label {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--tx3) !important;
    font-family: var(--mono) !important;
    display: block;
    margin-bottom: 12px;
}}
.vi-pred-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2px;
}}
.vi-pred-name {{
    font-size: 12px;
    color: var(--tx2) !important;
    font-family: var(--mono) !important;
}}
.vi-pred-name.top {{ color: var(--tx) !important; font-weight: 600; }}
.vi-pred-pct {{
    font-size: 12px;
    color: var(--tx3) !important;
    font-family: var(--mono) !important;
}}
.vi-pred-pct.top {{ color: var(--tx2) !important; }}
.vi-mini-bar {{
    height: 2px;
    background: var(--bd);
    border-radius: 1px;
    overflow: hidden;
    margin-bottom: 8px;
}}
.vi-mini-fill {{ height: 100%; border-radius: 1px; background: var(--bd2); }}
.vi-mini-fill.top {{ background: var(--ac); opacity: 0.8; }}

/* ── Empty result panel ── */
.vi-empty {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 56px 24px;
    min-height: 280px;
    margin-bottom: 16px;
}}
.vi-empty-icon {{ font-size: 2.2rem; opacity: 0.4; margin-bottom: 14px; }}
.vi-empty-head {{
    font-size: 16px;
    font-weight: 600;
    color: var(--tx2) !important;
    margin-bottom: 6px;
}}
.vi-empty-sub {{
    font-size: 13px;
    color: var(--tx3) !important;
    line-height: 1.6;
}}

/* ── Metric row ── */
.vi-metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}}
.vi-metric {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}}
.vi-metric-val {{
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--tx) !important;
    font-family: var(--mono) !important;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
}}
.vi-metric-lbl {{
    font-size: 11px;
    color: var(--tx3) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}

/* ── Intent chips ── */
.vi-chips {{ line-height: 2.5; }}
.vi-chip {{
    display: inline-flex;
    align-items: center;
    background: var(--card2);
    border: 1px solid var(--bd);
    color: var(--tx2) !important;
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 11px;
    font-family: var(--mono) !important;
    margin: 3px;
}}

/* ── History rows ── */
.vi-hist-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--bd);
}}
.vi-hist-row:last-child {{ border-bottom: none; }}
.vi-hist-name {{
    font-size: 13px;
    font-weight: 600;
    color: var(--tx) !important;
    font-family: var(--mono) !important;
}}
.vi-hist-meta {{
    font-size: 11px;
    color: var(--tx3) !important;
    font-family: var(--mono) !important;
}}

/* ── Viz card ── */
.vi-viz {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 12px;
    padding: 18px 20px 10px;
    margin-bottom: 12px;
}}

/* ── Command list ── */
.vi-cmd {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--bd);
}}
.vi-cmd:last-child {{ border-bottom: none; }}
.vi-cmd-bullet {{
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--ac);
    opacity: 0.5;
    flex-shrink: 0;
}}
.vi-cmd-text {{ font-size: 14px; color: var(--tx2) !important; }}

/* ── About block ── */
.vi-about {{
    background: var(--card);
    border: 1px solid var(--bd);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 10px;
}}
.vi-about-label {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ac) !important;
    font-family: var(--mono) !important;
    display: block;
    margin-bottom: 10px;
}}
.vi-about-body {{
    font-family: var(--mono) !important;
    font-size: 13px;
    color: var(--tx2) !important;
    line-height: 1.85;
    white-space: pre-wrap;
    margin: 0;
}}

/* ════════════════════════════════════
   STREAMLIT BUTTONS — clean, one style
   ════════════════════════════════════ */
.stButton > button {{
    width: 100%;
    background: var(--card) !important;
    color: var(--tx) !important;
    border: 1px solid var(--bd) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: background 0.12s, border-color 0.12s !important;
    box-shadow: none !important;
    cursor: pointer !important;
    line-height: 1.4 !important;
    letter-spacing: 0 !important;
}}
.stButton > button:hover {{
    background: var(--card2) !important;
    border-color: var(--bd2) !important;
    color: var(--tx) !important;
    opacity: 1 !important;
    transform: none !important;
    box-shadow: none !important;
}}
.stButton > button p,
.stButton > button span,
.stButton > button div {{
    color: var(--tx) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: var(--font) !important;
}}

/* Primary CTA — accent style */
.vi-cta .stButton > button {{
    background: var(--ac) !important;
    color: #fff !important;
    border-color: var(--ac) !important;
    font-weight: 600 !important;
}}
.vi-cta .stButton > button:hover {{
    opacity: 0.9 !important;
    background: var(--ac) !important;
    border-color: var(--ac) !important;
}}
.vi-cta .stButton > button p,
.vi-cta .stButton > button span {{
    color: #fff !important;
    font-weight: 600 !important;
}}

/* Segmented control — identical height for ALL nav buttons */
.vi-seg .stButton > button,
.vi-seg-active .stButton > button {{
    border-radius: 6px !important;
    padding: 0 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    vertical-align: middle !important;
}}
/* Force nav row columns to vertically centre */
.vi-seg-active .stButton > button {{
    background: var(--acbg) !important;
    border-color: var(--ac) !important;
    color: var(--ac) !important;
    font-weight: 600 !important;
    height: 42px !important;
    min-height: 42px !important;
}}
.vi-seg-active .stButton > button p,
.vi-seg-active .stButton > button span {{
    color: var(--ac) !important;
    font-weight: 600 !important;
}}

/* All nav row buttons same height */
[data-testid="stHorizontalBlock"] [data-testid="column"] .stButton > button {{
    height: 42px !important;
    min-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

/* ════════════════
   FILE UPLOADER
   ════════════════ */
[data-testid="stFileUploader"] section {{
    background: var(--card2) !important;
    border: 1.5px dashed var(--bd2) !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {{
    color: var(--tx2) !important;
    font-size: 14px !important;
}}
[data-testid="stFileUploader"] button {{
    background: var(--card) !important;
    color: var(--tx2) !important;
    border: 1px solid var(--bd) !important;
    box-shadow: none !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
}}
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {{
    color: var(--tx2) !important;
    font-weight: 400 !important;
    font-size: 13px !important;
}}

/* ════════════════
   AUDIO INPUT
   ════════════════ */
[data-testid="stAudioInput"] {{
    background: var(--card2) !important;
    border: 1.5px solid var(--bd2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stAudioInput"] iframe {{
    filter: {t['af']} !important;
    border-radius: 8px !important;
}}

/* ════════════════
   SELECT SLIDER
   ════════════════ */
[data-testid="stRadio"] label {{
    color: var(--tx2) !important;
    font-size: 14px !important;
}}

/* ════════════════
   LINE CHART
   ════════════════ */
[data-testid="stArrowVegaLiteChart"] {{
    border-radius: 8px;
}}

/* ════════════════
   ANIMATION
   ════════════════ */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade {{ animation: fadeIn 0.25s ease forwards; }}
</style>
""", unsafe_allow_html=True)


# ── State ─────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "model": None, "label_encoder": None,
        "model_loaded": False, "load_error": "",
        "history": [], "total_preds": 0,
        "sum_conf": 0.0, "sum_time": 0.0,
        "page": "try", "mode": "upload",
        "result": None, "theme": "dark",
        "random_audio": None, "random_audio_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Model — TFLite (works on Streamlit Cloud, no TensorFlow needed) ───────────
@st.cache_resource(show_spinner=False)
def load_model_cached(tflite_path, encoder_path):
    import pickle

    # Load label encoder
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    # Try tflite-runtime first (lightweight, Streamlit Cloud compatible)
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        # Fallback to tensorflow.lite on local machines
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()
    return interpreter, le


def try_load_model():
    tflite_path = Path("models/voiceintent.tflite")
    encoder_path = Path("models/label_encoder.pkl")

    # Fallback: try .keras if .tflite not present (local dev)
    keras_path = Path("models/voiceintent_cnn.keras")

    if tflite_path.exists() and encoder_path.exists():
        try:
            interp, le = load_model_cached(str(tflite_path), str(encoder_path))
            st.session_state.update(model=interp, label_encoder=le,
                                    model_loaded=True, load_error="",
                                    model_type="tflite")
        except Exception as e:
            st.session_state["load_error"] = str(e)[:80]

    elif keras_path.exists() and encoder_path.exists():
        try:
            import tensorflow as tf, pickle
            m = tf.keras.models.load_model(str(keras_path))
            with open(str(encoder_path), "rb") as f:
                le = pickle.load(f)
            st.session_state.update(model=m, label_encoder=le,
                                    model_loaded=True, load_error="",
                                    model_type="keras")
        except Exception as e:
            st.session_state["load_error"] = str(e)[:80]
    else:
        st.session_state["load_error"] = "Model not found — add voiceintent.tflite to models/"


# ── Inference — works with both TFLite and Keras ───────────────────────────────
def predict(audio_bytes):
    from utils.audio    import bytes_to_array
    from utils.features import extract_mfcc
    model      = st.session_state["model"]
    le         = st.session_state["label_encoder"]
    model_type = st.session_state.get("model_type", "keras")
    t0         = time.perf_counter()

    waveform, sr = bytes_to_array(audio_bytes)
    mfcc         = extract_mfcc(waveform, sr)
    inp          = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Normalisation
    g_mean = np.load("models/global_mean.npy")[0] if Path("models/global_mean.npy").exists() else None
    g_std  = np.load("models/global_std.npy")[0]  if Path("models/global_std.npy").exists()  else None
    if g_mean is not None and g_std is not None:
        inp = (inp - g_mean) / g_std

    if model_type == "tflite":
        # TFLite inference
        input_details  = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], inp)
        model.invoke()
        probs = model.get_tensor(output_details[0]["index"])[0]
    else:
        # Keras inference
        probs = model.predict(inp, verbose=0)[0]

    idx    = int(np.argmax(probs))
    intent = le.inverse_transform([idx])[0]
    conf   = float(probs[idx])
    all_p  = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}

    return {
        "intent": intent, "confidence": conf, "all_probs": all_p,
        "waveform": waveform, "sr": sr, "mfcc": mfcc,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def update_stats(res):
    st.session_state["total_preds"] += 1
    st.session_state["sum_conf"]    += res["confidence"]
    st.session_state["sum_time"]    += res["duration_ms"]
    st.session_state["history"].insert(0, {
        "intent": res["intent"], "confidence": res["confidence"],
        "duration_ms": res["duration_ms"], "timestamp": time.strftime("%H:%M:%S"),
    })
    if len(st.session_state["history"]) > 30:
        st.session_state["history"].pop()


# ── Render helpers ─────────────────────────────────────────────────────────────
def render_result(res):
    pct = int(res["confidence"] * 100)
    st.markdown(f"""
<div class="vi-result fade">
    <span class="vi-intent-tag">Detected intent</span>
    <div class="vi-intent-name">{res['intent']}</div>
    <div class="vi-conf-row">
        <span class="vi-conf-lbl">Confidence score</span>
        <span class="vi-conf-num">{pct}%</span>
    </div>
    <div class="vi-bar"><div class="vi-bar-fill" style="width:{pct}%"></div></div>
    <span class="vi-top5-label">Top 5 predictions</span>
</div>
""", unsafe_allow_html=True)
    for lbl, prob in sorted(res["all_probs"].items(), key=lambda x: x[1], reverse=True)[:5]:
        p   = int(prob * 100)
        top = lbl == res["intent"]
        nc  = "vi-pred-name top" if top else "vi-pred-name"
        pc  = "vi-pred-pct top"  if top else "vi-pred-pct"
        fc  = "vi-mini-fill top" if top else "vi-mini-fill"
        st.markdown(f"""
<div class="vi-pred-row">
    <span class="{nc}">{lbl}</span>
    <span class="{pc}">{p}%</span>
</div>
<div class="vi-mini-bar"><div class="{fc}" style="width:{p}%"></div></div>
""", unsafe_allow_html=True)


def render_waveform(waveform):
    import pandas as pd
    step = max(1, len(waveform) // 400)
    df   = pd.DataFrame({"amplitude": waveform[::step][:400]})
    st.markdown('<span class="vi-eyebrow">Waveform</span>', unsafe_allow_html=True)
    st.line_chart(df, height=90, use_container_width=True)


def render_mfcc(mfcc):
    import matplotlib, matplotlib.pyplot as plt
    matplotlib.use("Agg")
    t = T()
    fig, ax = plt.subplots(figsize=(5, 1.6))
    fig.patch.set_facecolor(t["card"])
    ax.set_facecolor(t["card"])
    img = ax.imshow(mfcc, aspect="auto", origin="lower",
                    cmap=t["mfcc"], interpolation="nearest")
    ax.set_xlabel("Time", color=t["tx3"], fontsize=7)
    ax.set_ylabel("Coeff", color=t["tx3"], fontsize=7)
    ax.tick_params(colors=t["tx3"], labelsize=6)
    for s in ax.spines.values(): s.set_edgecolor(t["border"])
    plt.colorbar(img, ax=ax).ax.tick_params(colors=t["tx3"], labelsize=6)
    plt.tight_layout(pad=0.3)
    st.markdown('<span class="vi-eyebrow">MFCC Features</span>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_try():
    t   = T()
    err = st.session_state["load_error"]
    mode = st.session_state["mode"]

    # Status
    if err:
        st.markdown(f'<span class="vi-status err">⚠ {err}</span>',
                    unsafe_allow_html=True)
    elif st.session_state["model_loaded"]:
        st.markdown('<span class="vi-status ok">● Model ready</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="vi-status ld">◌ Loading model...</span>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # ── Input card with tab header ─────────────────────────────────────────
    # The actual tab buttons — these are the ONLY buttons for mode switching
    col_up, col_rec = st.columns(2)
    with col_up:
        if mode == "upload":
            st.markdown('<div class="vi-seg-active vi-seg">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="vi-seg">', unsafe_allow_html=True)
        if st.button("📂  Upload file", key="tab_upload", use_container_width=True):
            st.session_state["mode"] = "upload"
            st.session_state["result"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rec:
        if mode == "record":
            st.markdown('<div class="vi-seg-active vi-seg">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="vi-seg">', unsafe_allow_html=True)
        if st.button("🎙  Record voice", key="tab_record", use_container_width=True):
            st.session_state["mode"] = "record"
            st.session_state["result"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if mode == "upload":
        # ── Random sample button ───────────────────────────────────────────
        st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:10px">
    <span style="font-size:13px;color:var(--tx3)">
        WAV · FLAC · OGG · MP3 &nbsp;—&nbsp; 1 to 5 seconds recommended
    </span>
</div>
""", unsafe_allow_html=True)

        rand_col, _ = st.columns([2, 3])
        with rand_col:
            if st.button("🎲  Test a random sample!", key="random_sample",
                         use_container_width=True):
                import glob, random, os
                # Look for dataset in common locations
                dataset_paths = [
                    "data/fluent_speech_commands/wavs",
                    "../data/fluent_speech_commands/wavs",
                    "data/wavs",
                ]
                wav_files = []
                for base in dataset_paths:
                    if os.path.exists(base):
                        wav_files = glob.glob(f"{base}/**/*.wav", recursive=True)
                        if wav_files:
                            break

                if wav_files:
                    chosen = random.choice(wav_files)
                    with open(chosen, "rb") as f:
                        st.session_state["random_audio"] = f.read()
                    st.session_state["random_audio_name"] = os.path.basename(chosen)
                else:
                    st.session_state["random_audio"] = None
                    st.warning("Dataset not found locally. Upload a WAV file manually.")
                st.rerun()

        # Show random audio if loaded
        if st.session_state.get("random_audio"):
            import io
            audio_data = st.session_state["random_audio"]
            name = st.session_state.get("random_audio_name", "sample.wav")
            st.markdown(
                f'<p style="font-size:13px;color:var(--tx3);margin:8px 0 4px">'
                f'🎵 Random sample: <code>{name}</code></p>',
                unsafe_allow_html=True
            )
            st.audio(audio_data, format="audio/wav")
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="vi-cta">', unsafe_allow_html=True)
            if st.button("Classify this sample →", key="cls_random",
                         use_container_width=True):
                if not st.session_state["model_loaded"]:
                    st.error("Model not loaded.")
                else:
                    with st.spinner("Analysing audio..."):
                        res = predict(audio_data)
                    update_stats(res)
                    st.session_state["result"] = res
                    st.session_state["random_audio"] = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="vi-line-sm"></div>', unsafe_allow_html=True)

        # ── Manual upload ──────────────────────────────────────────────────
        st.markdown('<span class="vi-eyebrow">Or upload your own file</span>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "drop audio file",
            type=["wav", "flac", "ogg", "mp3"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.audio(uploaded)
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="vi-cta">', unsafe_allow_html=True)
            if st.button("Classify intent →", key="cls_up", use_container_width=True):
                if not st.session_state["model_loaded"]:
                    st.error("Model not loaded. Check model files in /models/")
                else:
                    with st.spinner("Analysing audio..."):
                        res = predict(uploaded.read())
                    update_stats(res)
                    st.session_state["result"] = res
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    else:  # record
        st.caption("Click the microphone button below · speak your command · click stop")

        # Wrapper div to make audio input visible on dark theme
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Speak your command",
                                      label_visibility="collapsed")
        else:
            recorded = None

        if recorded:
            st.audio(recorded)
            st.markdown(
                f'<p style="font-size:13px;color:{t["ok"]};'
                f'font-family:JetBrains Mono,monospace;margin:8px 0 12px">'
                f'✓ Recording captured — ready to classify</p>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="vi-cta">', unsafe_allow_html=True)
            if st.button("Classify intent →", key="cls_mic", use_container_width=True):
                if not st.session_state["model_loaded"]:
                    st.error("Model not loaded.")
                else:
                    with st.spinner("Analysing audio..."):
                        res = predict(recorded.getvalue())
                    update_stats(res)
                    st.session_state["result"] = res
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        elif not hasattr(st, "audio_input"):
            st.markdown(f"""
<div style="padding:16px 0;font-size:14px;color:var(--tx2);line-height:1.7">
    Mic recording needs Streamlit ≥ 1.31<br>
    Run <code>pip install --upgrade streamlit</code> then restart the app.
</div>
""", unsafe_allow_html=True)

        # Example commands
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<span class="vi-eyebrow">Example commands to try</span>',
                    unsafe_allow_html=True)
        for cmd in [
            "Turn on the lights",
            "Turn off the lights in the bedroom",
            "Increase the volume",
            "Decrease the heat in the kitchen",
            "What's the weather like",
            "Bring me the newspaper",
            "Change language to Chinese",
            "Activate the lamp",
        ]:
            st.markdown(
                f'<div class="vi-cmd">'
                f'<span class="vi-cmd-bullet"></span>'
                f'<span class="vi-cmd-text">{cmd}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Result ──────────────────────────────────────────────────────────────
    if st.session_state["result"]:
        render_result(st.session_state["result"])
    else:
        icon  = "🎙️" if mode == "record" else "📂"
        head  = "Record your voice" if mode == "record" else "Upload an audio file"
        st.markdown(f"""
<div class="vi-empty">
    <div class="vi-empty-icon">{icon}</div>
    <div class="vi-empty-head">{head}</div>
    <div class="vi-empty-sub">then click <strong>Classify intent</strong><br>to see the prediction appear here</div>
</div>
""", unsafe_allow_html=True)

    # ── Signal analysis ──────────────────────────────────────────────────────
    if st.session_state["result"]:
        res = st.session_state["result"]
        st.markdown('<span class="vi-eyebrow" style="margin-top:8px;display:block">Signal analysis</span>',
                    unsafe_allow_html=True)
        v1, v2 = st.columns(2, gap="medium")
        with v1:
            st.markdown('<div class="vi-viz">', unsafe_allow_html=True)
            render_waveform(res["waveform"])
            st.markdown('</div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="vi-viz">', unsafe_allow_html=True)
            render_mfcc(res["mfcc"])
            st.markdown('</div>', unsafe_allow_html=True)

    # ── History ────────────────────────────────────────────────────────────
    if st.session_state["history"]:
        st.markdown('<div class="vi-line-sm"></div>', unsafe_allow_html=True)
        st.markdown('<span class="vi-eyebrow">Recent predictions</span>',
                    unsafe_allow_html=True)
        st.markdown('<div class="vi-card-sm">', unsafe_allow_html=True)
        for item in st.session_state["history"][:6]:
            pct = int(item["confidence"] * 100)
            st.markdown(f"""
<div class="vi-hist-row">
    <span class="vi-hist-name">{item['intent']}</span>
    <span class="vi-hist-meta">{pct}% · {item['duration_ms']:.0f}ms · {item['timestamp']}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def page_explore():
    n  = st.session_state["total_preds"]
    ac = (st.session_state["sum_conf"] / n * 100) if n else 0.0
    at = (st.session_state["sum_time"] / n)       if n else 0.0

    st.markdown('<span class="vi-eyebrow">Session metrics</span>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="vi-metrics">
    <div class="vi-metric">
        <div class="vi-metric-val">{n}</div>
        <div class="vi-metric-lbl">Predictions</div>
    </div>
    <div class="vi-metric">
        <div class="vi-metric-val">93.1%</div>
        <div class="vi-metric-lbl">Model accuracy</div>
    </div>
    <div class="vi-metric">
        <div class="vi-metric-val">{ac:.0f}%</div>
        <div class="vi-metric-lbl">Avg confidence</div>
    </div>
    <div class="vi-metric">
        <div class="vi-metric-val">{at:.0f}ms</div>
        <div class="vi-metric-lbl">Avg latency</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="vi-line-sm"></div>', unsafe_allow_html=True)
    st.markdown('<span class="vi-eyebrow">31 supported intent classes</span>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size:14px;color:var(--tx2);margin-bottom:14px;line-height:1.6">Each intent is a combination of action + object + location. The model was trained on 30,043 utterances from 97 speakers.</p>',
                unsafe_allow_html=True)
    from utils.intents import INTENT_LIST
    chips = "".join(f'<span class="vi-chip">{i}</span>' for i in INTENT_LIST)
    st.markdown(f'<div class="vi-chips">{chips}</div>', unsafe_allow_html=True)

    if st.session_state["history"]:
        st.markdown('<div class="vi-line-sm"></div>', unsafe_allow_html=True)
        st.markdown('<span class="vi-eyebrow">Confidence distribution</span>',
                    unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame([{"Confidence (%)": int(h["confidence"]*100)}
                           for h in st.session_state["history"]])
        st.bar_chart(df, y="Confidence (%)", height=200, use_container_width=True)


def page_about():
    blocks = [
        ("Pipeline",
         "Raw audio\n"
         "    → Resampled to 16 kHz mono\n"
         "    → MFCC extraction  (40 coeff · 25ms window · 10ms hop · 128 frames)\n"
         "    → CNN inference\n"
         "    → Softmax over 31 intent classes"),
        ("Architecture",
         "Input  (40 × 128 × 1)\n\n"
         "Block 1   Conv2D(64)  × 2  →  BatchNorm  →  MaxPool  →  Dropout\n"
         "Block 2   Conv2D(128) × 2  →  BatchNorm  →  MaxPool  →  Dropout\n"
         "Block 3   Conv2D(256) × 2  →  BatchNorm  →  MaxPool  →  Dropout\n"
         "Block 4   Conv2D(512)      →  BatchNorm  →  GlobalAvgPool\n\n"
         "Dense(512) → Dense(256) → Softmax(31)"),
        ("Dataset",
         "Fluent Speech Commands\n"
         "30,043 utterances · 97 speakers · 248 unique phrases\n"
         "31 intent classes (action × object × location)\n"
         "Speaker-independent train / val / test split\n\n"
         "License: Fluent Speech Commands Public License\n"
         "Non-commercial and academic use only."),
        ("Performance",
         "Val accuracy     93.1%  (speaker-independent — unseen speakers)\n"
         "Inference time   < 150 ms on CPU\n\n"
         "Speaker-independent evaluation is significantly harder than\n"
         "speaker-dependent. Fine-tuned wav2vec / Whisper models reach 99%+.\n"
         "This project demonstrates the full end-to-end ML pipeline."),
        ("Tech stack",
         "Python · TensorFlow / Keras · Librosa\n"
         "Streamlit · Pandas · NumPy · Matplotlib · Scikit-learn"),
    ]
    for title, body in blocks:
        st.markdown(f"""
<div class="vi-about">
    <span class="vi-about-label">{title}</span>
    <pre class="vi-about-body">{body}</pre>
</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_state()
    inject_css()

    if not st.session_state["model_loaded"] and not st.session_state["load_error"]:
        try_load_model()

    t    = T()
    page = st.session_state["page"]

    # ── App header ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="vi-header">
    <span class="vi-app-name">VoiceIntent</span>
    <h1 class="vi-headline">Speak. Classify. Understand.</h1>
    <p style="font-size:16px;color:{t['tx2']};line-height:1.65;max-width:520px;margin:0 auto 20px;text-align:center;display:block">Upload an audio clip or record your voice. The CNN model identifies the intent behind your words in real time.</p>
</div>
""", unsafe_allow_html=True)

    # ── Page navigation — segmented control ─────────────────────────────────
    nav1, nav2, nav3, _, nav5 = st.columns([2, 2, 2, 3, 2])

    with nav1:
        cls = "vi-seg-active vi-seg" if page == "try" else "vi-seg"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("Try it out!", key="nav_try", use_container_width=True):
            st.session_state["page"] = "try"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with nav2:
        cls = "vi-seg-active vi-seg" if page == "explore" else "vi-seg"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("Explore", key="nav_exp", use_container_width=True):
            st.session_state["page"] = "explore"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with nav3:
        cls = "vi-seg-active vi-seg" if page == "about" else "vi-seg"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("About", key="nav_ab", use_container_width=True):
            st.session_state["page"] = "about"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with nav5:
        st.markdown('<div class="vi-seg">', unsafe_allow_html=True)
        if st.button(f"{t['swap_lbl']} mode", key="nav_theme", use_container_width=True):
            st.session_state["theme"] = t["swap"]; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="vi-line"></div>', unsafe_allow_html=True)

    # ── Page content ────────────────────────────────────────────────────────
    if page == "try":
        page_try()
    elif page == "explore":
        page_explore()
    elif page == "about":
        page_about()

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;padding:48px 0 24px;border-top:1px solid var(--bd);margin-top:48px">
    <p style="font-size:12px;color:var(--tx3);font-family:JetBrains Mono,monospace;margin:0">
        VoiceIntent · CNN + MFCC · Fluent Speech Commands Dataset
    </p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()