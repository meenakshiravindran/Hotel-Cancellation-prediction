
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
st.set_page_config(
    page_title="LUXE · Hotel Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)
BG      = "#070B14"
CARD    = "#0D1322"
CARD2   = "#131927"
BORDER  = "#1E2A3E"
TXT     = "#F0EAE0"
MUTED   = "#8A9BB5"
DIM     = "#3D4F6B"
GOLD    = "#C9903C"
GOLD2   = "#E8B86D"
RED     = "#E8735A"
TEAL    = "#4ECDC4"
BLUE    = "#5B8FD4"
PURPLE  = "#9B7FD4"
st.markdown("""
<style>
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap)');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes float-y {
    0%, 100% { transform: translateX(-50%) translateY(0); }
    50%       { transform: translateX(-50%) translateY(-8px); }
}
[data-testid="stAppViewContainer"] {
    background: #070B14 !important;
    min-height: 100vh;
    animation: fadeIn 0.5s ease;
}
[data-testid="collapsedControl"],
[data-testid="stSidebar"],
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
    padding: 0 !important; max-width: 100% !important;
}
h1,h2,h3,h4,h5,h6 { font-family:'Playfair Display',serif !important; color:#F0EAE0 !important; }
p,label,li,span    { font-family:'Inter',sans-serif !important; }
/* ── HERO ── */
.hero-wrapper {
    position: relative; width: 100%; min-height: 92vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(7,11,20,0.3) 0%, rgba(7,11,20,0.65) 50%, rgba(7,11,20,0.97) 100%),
                linear-gradient(135deg, #070B14 0%, #0D1322 100%);
}
.hero-bg-mosaic {
    position: absolute; inset: 0;
    display: grid; grid-template-columns: repeat(3,1fr); grid-template-rows: repeat(2,1fr);
    opacity: 0.22; filter: saturate(0.6);
}
.hero-bg-cell { overflow: hidden; }
.hero-bg-cell img { width:100%; height:100%; object-fit:cover; }
.hero-content {
    position: relative; z-index: 10; text-align: center;
    padding: 0 2rem; max-width: 900px;
    animation: fadeInUp 0.9s cubic-bezier(.22,1,.36,1) both;
}
.hero-eyebrow {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    letter-spacing:0.25em; color:#C9903C; text-transform:uppercase;
    margin-bottom:1.2rem; display:flex; align-items:center; justify-content:center; gap:12px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content:''; display:block; width:40px; height:1px;
    background:linear-gradient(90deg,transparent,#C9903C);
}
.hero-eyebrow::after { transform:scaleX(-1); }
.hero-title {
    font-family:'Playfair Display',serif !important;
    font-size:clamp(3rem,7vw,5.5rem); font-weight:800;
    line-height:1.05; color:#F0EAE0 !important; letter-spacing:-0.02em; margin-bottom:0.4rem;
}
.hero-title em {
    font-style:italic;
    background:linear-gradient(135deg,#C9903C 0%,#E8B86D 40%,#C9903C 60%,#E8B86D 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:shimmer 3s linear infinite;
}
.hero-subtitle {
    font-family:'Inter',sans-serif; font-size:1.05rem; color:#8A9BB5; font-weight:300;
    margin-bottom:3rem; line-height:1.7; max-width:560px; margin-left:auto; margin-right:auto;
}
.hero-stats {
    display:flex; gap:0; justify-content:center; margin-bottom:3.5rem;
    border:1px solid rgba(201,144,60,0.2); border-radius:16px;
    overflow:hidden; backdrop-filter:blur(12px); background:rgba(13,19,34,0.6);
    max-width:640px; margin-left:auto; margin-right:auto;
}
.hero-stat {
    flex:1; padding:20px 16px; text-align:center;
    border-right:1px solid rgba(201,144,60,0.15); transition:background 0.2s;
}
.hero-stat:last-child { border-right:none; }
.hero-stat:hover { background:rgba(201,144,60,0.06); }
.hero-stat-val { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; color:#E8B86D; line-height:1; display:block; margin-bottom:4px; }
.hero-stat-lbl { font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:#3D4F6B; letter-spacing:0.1em; text-transform:uppercase; }
.scroll-hint {
    position:absolute; bottom:2rem; left:50%; transform:translateX(-50%);
    display:flex; flex-direction:column; align-items:center; gap:8px;
    animation:float-y 2.2s ease-in-out infinite;
}
.scroll-hint-line { width:1px; height:40px; background:linear-gradient(180deg,#C9903C,transparent); }
.scroll-hint-lbl { font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:#3D4F6B; letter-spacing:0.15em; text-transform:uppercase; }
/* ── CONTENT ── */
.content-section { padding:5rem 4rem 2rem; max-width:1400px; margin:0 auto; }
.section-heading { display:flex; flex-direction:column; gap:8px; margin-bottom:2.5rem; padding-bottom:1.5rem; border-bottom:1px solid #1E2A3E; position:relative; }
.section-heading::after { content:''; position:absolute; bottom:-1px; left:0; width:60px; height:2px; background:linear-gradient(90deg,#C9903C,transparent); }
.section-tag { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; color:#C9903C; text-transform:uppercase; }
.section-title { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:700; color:#F0EAE0; line-height:1.15; letter-spacing:-0.02em; }
.section-desc { font-family:'Inter',sans-serif; font-size:0.88rem; color:#8A9BB5; line-height:1.7; max-width:560px; }
/* ── KPI STRIP ── */
.kpi-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:3rem; padding:0 4rem; }
.kpi-tile { background:#0D1322; border:1px solid #1E2A3E; border-radius:16px; padding:22px 20px; position:relative; overflow:hidden; transition:all 0.25s ease; }
.kpi-tile::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; background:var(--stripe,linear-gradient(90deg,#C9903C,#E8B86D)); opacity:0.7; }
.kpi-tile:hover { border-color:#2E3E58; transform:translateY(-3px); box-shadow:0 14px 36px rgba(0,0,0,0.35); }
.kpi-tile-icon { font-size:1.4rem; margin-bottom:12px; display:block; }
.kpi-tile-val  { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:700; color:var(--kval,#E8B86D); line-height:1; margin-bottom:6px; display:block; }
.kpi-tile-lbl  { font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:#3D4F6B; letter-spacing:0.12em; text-transform:uppercase; display:block; }
.kpi-tile-sub  { font-family:'Inter',sans-serif; font-size:0.72rem; color:#8A9BB5; margin-top:6px; display:block; }
/* ── VIZ CARD ── */
.viz-card-header-block {
    background:#0D1322; border:1px solid #1E2A3E; border-bottom:none;
    border-radius:20px 20px 0 0; padding:20px 24px 16px;
    display:flex; align-items:center; justify-content:space-between;
}
.viz-card-body {
    background:#0D1322; border:1px solid #1E2A3E; border-top:none;
    border-radius:0 0 20px 20px; padding:0 16px 20px; margin-bottom:24px;
}
.viz-card-title { font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:600; color:#F0EAE0; }
.viz-card-badge { font-family:'JetBrains Mono',monospace; font-size:0.58rem; padding:4px 10px; border-radius:20px; background:rgba(201,144,60,0.12); border:1px solid rgba(201,144,60,0.25); color:#C9903C; letter-spacing:0.08em; text-transform:uppercase; }
/* ═══════════════════════════════════════════════════════════════════════
   PREDICT PAGE — UNIFIED MEGA-CARD
   The entire predict workspace lives inside ONE element that has the
   class .predict-mega-card. We use :has() to style the parent block
   container as a single bordered box.
   ═══════════════════════════════════════════════════════════════════════ */
/* Outer wrapper limits width and adds page padding */
.predict-page-frame {
    padding: 6 4rem 4rem;
    max-width: 1200px;
    margin: 2 auto;
}
/* The mega card itself — a vertical block container that has our sentinel */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .predict-mega-sentinel) {
    background: linear-gradient(180deg, #0D1322 0%, #0A0F1C 100%) !important;
    border: 1px solid #1E2A3E !important;
    border-radius: 28px !important;
    padding: 36px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(201,144,60,0.06) !important;
    position: relative;
    overflow: hidden;
}
.predict-mega-sentinel { display: none !important; }
/* Inner sub-card: the two input columns */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .predict-input-sentinel) {
    background: rgba(7, 11, 20, 0.5) !important;
    border: 1px solid #1E2A3E !important;
    border-radius: 20px !important;
    padding: 36px !important;
    margin-bottom: 20px !important;
}
.predict-input-sentinel { display: none !important; }
/* Each individual input column inside the sub-card */
[data-testid="stColumn"]:has(> div > [data-testid="stElementContainer"] .pred-col-sentinel) {
    background: #0D1322 !important;
    border: 1px solid #1E2A3E !important;
    border-radius: 16px !important;
    padding: 22px 20px 26px !important;
    transition: border-color 0.25s ease, transform 0.25s ease !important;
}
[data-testid="stColumn"]:has(> div > [data-testid="stElementContainer"] .pred-col-sentinel):hover {
    border-color: rgba(201,144,60,0.35) !important;
    transform: translateY(-2px) !important;
}
.pred-col-sentinel { display: none !important; }
.pred-card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #F0EAE0 !important;
    padding-bottom: 14px !important;
    margin-bottom: 18px !important;
    border-bottom: 1px solid #1E2A3E !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.predict-mega-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    margin-bottom: 24px;
    border-bottom: 1px solid #1E2A3E;
}
.predict-mega-header-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #F0EAE0;
}
.predict-mega-header-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #C9903C;
    text-transform: uppercase;
    padding: 6px 14px;
    background: rgba(201,144,60,0.1);
    border: 1px solid rgba(201,144,60,0.3);
    border-radius: 30px;
}
/* ── RESULT CARDS ── */
.verdict-card { border-radius:24px; padding:36px 24px; min-height:260px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
.verdict-card.cancel { background:radial-gradient(ellipse at 50% 0%,rgba(232,115,90,0.15) 0%,transparent 70%),#0D1322; border:1px solid rgba(232,115,90,0.35); box-shadow:0 0 60px rgba(232,115,90,0.07); }
.verdict-card.keep   { background:radial-gradient(ellipse at 50% 0%,rgba(78,205,196,0.12) 0%,transparent 70%),#0D1322; border:1px solid rgba(78,205,196,0.30); box-shadow:0 0 60px rgba(78,205,196,0.06); }
.verdict-icon  { font-size:3rem; margin-bottom:16px; }
.verdict-label { font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:12px; }
.verdict-pct   { font-family:'Playfair Display',serif; font-size:4.5rem; font-weight:800; line-height:1; margin-bottom:6px; }
.verdict-sub   { font-family:'Inter',sans-serif; font-size:0.72rem; color:#3D4F6B; letter-spacing:0.06em; text-transform:uppercase; }
.risk-analysis { background:#0D1322; border:1px solid #1E2A3E; border-radius:24px; padding:24px 28px; }
.risk-row { display:flex; align-items:center; gap:14px; padding:11px 0; border-bottom:1px solid #1E2A3E; }
.risk-row:last-child { border-bottom:none; }
.risk-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.risk-row-text { font-family:'Inter',sans-serif; font-size:0.82rem; color:#8A9BB5; flex:1; }
.risk-badge { font-family:'JetBrains Mono',monospace; font-size:0.58rem; padding:3px 9px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase; }
.gauge-wrap { background:#0D1322; border:1px solid #1E2A3E; border-radius:16px; padding:20px 24px; margin-top:20px; }
.gauge-bar { height:12px; border-radius:6px; overflow:hidden; background:#1E2A3E; margin:10px 0; display:flex; }
.gauge-fill-keep   { height:100%; background:linear-gradient(90deg,#4ECDC4,#3DBDB5); }
.gauge-fill-cancel { height:100%; background:linear-gradient(90deg,#E8735A,#D4624A); }
.gauge-labels { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.6rem; margin-top:4px; }
/* Risk pre-flight panel */
.preflight-panel {
    background: rgba(7,11,20,0.4);
    border: 1px solid #1E2A3E;
    border-left: 3px solid var(--risk-color, #4ECDC4);
    border-radius: 16px;
    padding: 22px 24px;
    margin: 20px 0;
}
.preflight-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--risk-color, #4ECDC4);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
/* ── STREAMLIT OVERRIDES ── */
.stSelectbox > div > div { background:#070B14 !important; border:1px solid #1E2A3E !important; border-radius:10px !important; color:#F0EAE0 !important; font-family:'Inter',sans-serif !important; font-size:0.85rem !important; }
.stSelectbox > div > div:focus-within { border-color:#C9903C !important; box-shadow:0 0 0 2px rgba(201,144,60,0.15) !important; }
[data-baseweb="popover"] { background:#131927 !important; border:1px solid #1E2A3E !important; border-radius:12px !important; }
[data-baseweb="menu"] { background:#131927 !important; }
[data-baseweb="menu"] li { color:#8A9BB5 !important; font-family:'Inter',sans-serif !important; font-size:0.82rem !important; }
[data-baseweb="menu"] li:hover { background:#1E2A3E !important; color:#F0EAE0 !important; }
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:#C9903C !important; border:2px solid #070B14 !important; box-shadow:0 0 0 3px rgba(201,144,60,0.3) !important; }
/* DEFAULT button styling — the predict button */
.stButton > button {
    background:linear-gradient(135deg,#C9903C 0%,#A87030 100%) !important;
    color:#F0EAE0 !important; -webkit-text-fill-color:#F0EAE0 !important;
    font-family:'Inter',sans-serif !important; font-weight:700 !important;
    font-size:0.88rem !important; letter-spacing:0.08em !important;
    text-transform:uppercase !important; border:none !important;
    border-radius:12px !important; padding:14px 24px !important;
    transition:all 0.25s ease !important;
    box-shadow:0 6px 28px rgba(201,144,60,0.3) !important;
    width:100% !important; white-space:nowrap !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 36px rgba(201,144,60,0.45) !important;
    background:linear-gradient(135deg,#DBA04C 0%,#C9903C 100%) !important;
}
/* NAV pill bar — scoped using a sentinel class so it does NOT affect predict button */
.nav-bar-scope ~ div [data-testid="stHorizontalBlock"]:first-of-type,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .nav-bar-sentinel) > [data-testid="stHorizontalBlock"] {
    max-width:540px; margin:-46px auto 0 !important;
    position:relative; z-index:100;
    background:rgba(13,19,34,0.9); backdrop-filter:blur(20px);
    border:1px solid rgba(201,144,60,0.28); border-radius:60px;
    padding:5px !important; gap:4px !important;
    box-shadow:0 8px 40px rgba(0,0,0,0.5);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .nav-bar-sentinel) > [data-testid="stHorizontalBlock"] .stButton > button {
    background:transparent !important;
    color:#8A9BB5 !important; -webkit-text-fill-color:#8A9BB5 !important;
    box-shadow:none !important; border-radius:50px !important;
    padding:9px 18px !important; font-size:0.72rem !important;
    letter-spacing:0.06em !important; font-weight:600 !important;
    border:none !important; white-space:nowrap !important; transition:all 0.2s ease !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .nav-bar-sentinel) > [data-testid="stHorizontalBlock"] .stButton > button:hover {
    background:rgba(201,144,60,0.12) !important;
    color:#E8B86D !important; -webkit-text-fill-color:#E8B86D !important;
    transform:none !important; box-shadow:none !important;
}
.nav-bar-sentinel { display:none !important; }
label[data-testid="stWidgetLabel"] p { font-family:'Inter',sans-serif !important; font-size:0.75rem !important; color:#8A9BB5 !important; font-weight:500 !important; }
.stDataFrame { border:1px solid #1E2A3E !important; border-radius:14px !important; overflow:hidden !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#070B14; }
::-webkit-scrollbar-thumb { background:#1E2A3E; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#2E3E58; }
div[data-testid="stHorizontalBlock"] { gap:20px !important; }
.stCaption { font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important; color:#3D4F6B !important; }
.stSpinner > div { border-top-color:#C9903C !important; }
[data-testid="StyledFullScreenButton"] { display:none !important; }
</style>
""", unsafe_allow_html=True)
def dark_fig(w=12, h=5, nrows=1, ncols=1):
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor("#0D1322")
    axes_list = np.array(axs).flatten() if nrows * ncols > 1 else [axs]
    for ax in axes_list:
        ax.set_facecolor("#070B14")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED); ax.title.set_color(TXT)
        for s in ax.spines.values(): s.set_color(BORDER); s.set_linewidth(0.5)
        ax.grid(axis='y', color=BORDER, linewidth=0.4, alpha=0.6); ax.set_axisbelow(True)
    return fig, axs
def viz_card(title, badge, chart_fn, *args, caption=None, **kwargs):
    st.markdown(f"""
    <div class="viz-card-header-block">
        <span class="viz-card-title">{title}</span>
        <span class="viz-card-badge">{badge}</span>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="viz-card-body">', unsafe_allow_html=True)
        chart_fn(*args, **kwargs)
        if caption: st.caption(caption)
        st.markdown('</div>', unsafe_allow_html=True)
@st.cache_data
def build_pipeline():
    RAW = pd.read_excel("hotel_bookings.xlsx")
    df = RAW.copy()
    df.drop(columns=["reservation_status","reservation_status_date","company","index"], errors="ignore", inplace=True)
    df["children"] = df["children"].fillna(0)
    df["country"]  = df["country"].fillna(df["country"].mode()[0])
    df["agent"]    = df["agent"].fillna(-1)
    df = df[(df["adr"]>=0)&(df["adr"]<5000)]
    df = df[(df["adults"]>0)|(df["children"]>0)|(df["babies"]>0)]
    df = df[(df["stays_in_weekend_nights"]+df["stays_in_week_nights"])>0]
    lead_lo=df["lead_time"].quantile(0.01); lead_hi=df["lead_time"].quantile(0.99)
    adr_lo=df["adr"].quantile(0.01);       adr_hi=df["adr"].quantile(0.99)
    wait_lo=df["days_in_waiting_list"].quantile(0.01); wait_hi=df["days_in_waiting_list"].quantile(0.99)
    df["lead_time"]=df["lead_time"].clip(lead_lo,lead_hi)
    df["adr"]=df["adr"].clip(adr_lo,adr_hi)
    df["days_in_waiting_list"]=df["days_in_waiting_list"].clip(wait_lo,wait_hi)
    df["adults"]=df["adults"].clip(upper=4)
    top20=df["country"].value_counts().nlargest(20).index
    df["country"]=df["country"].apply(lambda x: x if x in top20 else "Other")
    df["meal"]=df["meal"].replace("Undefined","SC")
    df["total_nights"]=df["stays_in_weekend_nights"]+df["stays_in_week_nights"]
    df["total_guests"]=df["adults"]+df["children"]+df["babies"]
    df["is_family"]=((df["children"]>0)|(df["babies"]>0)).astype(int)
    df["room_changed"]=(df["reserved_room_type"]!=df["assigned_room_type"]).astype(int)
    df["revenue_estimate"]=df["adr"]*df["total_nights"]
    df["is_long_stay"]=(df["total_nights"]>7).astype(int)
    df["is_high_lead_time"]=(df["lead_time"]>90).astype(int)
    df["lead_time_bin"]=pd.cut(df["lead_time"],bins=[-1,7,30,90,180,9999],labels=[0,1,2,3,4]).astype(int)
    df["is_returning_guest"]=(df["previous_bookings_not_canceled"]>0).astype(int)
    cat_cols=df.select_dtypes(include="object").columns.tolist()
    encoders={}
    for col in cat_cols:
        enc=LabelEncoder(); df[col]=enc.fit_transform(df[col].astype(str)); encoders[col]=enc
    df=df.fillna(df.median(numeric_only=True))
    X=df.drop(columns=["is_canceled"]); y=df["is_canceled"]
    feature_cols=X.columns.tolist()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc=StandardScaler(); X_tr_sc=sc.fit_transform(X_train); X_te_sc=sc.transform(X_test)
    df_b=RAW.copy()
    df_b.drop(columns=["reservation_status","reservation_status_date","company","index"],errors="ignore",inplace=True)
    df_b["children"]=df_b["children"].fillna(0); df_b["country"]=df_b["country"].fillna("Unknown"); df_b["agent"]=df_b["agent"].fillna(-1)
    for col in df_b.columns:
        if df_b[col].dtype=="object" or str(df_b[col].dtype)=="string": df_b[col]=df_b[col].fillna("Missing")
        elif pd.api.types.is_numeric_dtype(df_b[col]): df_b[col]=df_b[col].fillna(df_b[col].median())
    for col in df_b.select_dtypes(include="object").columns:
        le_tmp=LabelEncoder(); df_b[col]=le_tmp.fit_transform(df_b[col].astype(str))
    Xb=df_b.drop(columns=["is_canceled"]); yb=df_b["is_canceled"]
    Xb_tr,Xb_te,yb_tr,yb_te=train_test_split(Xb,yb,test_size=0.2,random_state=42,stratify=yb)
    sc_b=StandardScaler(); Xb_tr_sc=sc_b.fit_transform(Xb_tr); Xb_te_sc=sc_b.transform(Xb_te)
    BASELINE_MODELS={
        "Logistic Regression":LogisticRegression(max_iter=300,random_state=42),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        "KNN":KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42),
        "XGBoost":XGBClassifier(n_estimators=100,random_state=42,eval_metric="logloss",verbosity=0),
    }
    PROC_MODELS={
        "Logistic Regression":LogisticRegression(max_iter=500,random_state=42),
        "Decision Tree":DecisionTreeClassifier(max_depth=12,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1),
        "KNN":KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=150,random_state=42),
        "XGBoost":XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=7,random_state=42,eval_metric="logloss",verbosity=0),
    }
    SCALED={"Logistic Regression","KNN"}
    base_acc,proc_acc,trained_proc_models={},{},{}
    for name in BASELINE_MODELS:
        mb=BASELINE_MODELS[name]; mb.fit(Xb_tr_sc if name in SCALED else Xb_tr,yb_tr)
        base_acc[name]=round(accuracy_score(yb_te,mb.predict(Xb_te_sc if name in SCALED else Xb_te))*100,2)
        mp=PROC_MODELS[name]; mp.fit(X_tr_sc if name in SCALED else X_train,y_train)
        preds=mp.predict(X_te_sc if name in SCALED else X_test)
        proc_acc[name]=round(accuracy_score(y_test,preds)*100,2); trained_proc_models[name]=mp
    best_m=trained_proc_models["XGBoost"]
    best_pred=best_m.predict(X_test); best_prob=best_m.predict_proba(X_test)[:,1]
    rf_imp=pd.Series(trained_proc_models["Random Forest"].feature_importances_,index=X_train.columns).sort_values(ascending=False)
    winsor_bounds={"lead_time":(lead_lo,lead_hi),"adr":(adr_lo,adr_hi),"days_in_waiting_list":(wait_lo,wait_hi)}
    return (RAW,df,base_acc,proc_acc,best_m,best_pred,best_prob,y_test,rf_imp,X_train,encoders,sc,feature_cols,cat_cols,winsor_bounds)
with st.spinner("Calibrating intelligence models…"):
    (RAW,df_proc,base_acc,proc_acc,best_model,best_pred,best_prob,
     y_test,rf_imp,X_train,encoders,scaler,feature_cols,cat_cols,winsor_bounds)=build_pipeline()
cancel_rate=RAW["is_canceled"].mean()*100
avg_adr=RAW["adr"].mean()
n_countries=RAW["country"].nunique()
n_records=len(RAW)
if "page" not in st.session_state:
    st.session_state.page="eda"
HOTEL_IMGS = [
    "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=900&auto=format&fit=crop&q=60",
    "https://images.unsplash.com/photo-1571003123894-1f0594d2b5d9?w=900&auto=format&fit=crop&q=60",
    "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=900&auto=format&fit=crop&q=60",
    "https://images.unsplash.com/photo-1445019980597-93fa8acb246c?w=900&auto=format&fit=crop&q=60",
    "https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=900&auto=format&fit=crop&q=60",
    "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=900&auto=format&fit=crop&q=60",
]
# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-bg-mosaic">
        {"".join(f'<div class="hero-bg-cell"><img src="{img}" loading="lazy"/></div>' for img in HOTEL_IMGS)}
    </div>
    <div class="hero-content">
        <div class="hero-eyebrow">Machine Learning · Hospitality Intelligence</div>
        <h1 class="hero-title">Predict Every<br><em>Cancellation</em></h1>
        <p class="hero-subtitle">An end-to-end ML system trained on {n_records:,} hotel bookings — turning raw reservation data into real-time cancellation risk scores.</p>
        <div class="hero-stats">
            <div class="hero-stat"><span class="hero-stat-val">{n_records:,}</span><span class="hero-stat-lbl">Bookings</span></div>
            <div class="hero-stat"><span class="hero-stat-val">{cancel_rate:.1f}%</span><span class="hero-stat-lbl">Cancel Rate</span></div>
            <div class="hero-stat"><span class="hero-stat-val">{proc_acc['XGBoost']}%</span><span class="hero-stat-lbl">XGBoost Acc.</span></div>
            <div class="hero-stat"><span class="hero-stat-val">{n_countries}</span><span class="hero-stat-lbl">Countries</span></div>
        </div>
    </div>
    <div class="scroll-hint"><div class="scroll-hint-line"></div><div class="scroll-hint-lbl">Explore</div></div>
</div>
""", unsafe_allow_html=True)
# ─── NAV ──────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
bn1, bn2, bn3 = st.columns(3)
with bn1:
    if st.button("📊 Explore Data", key="nav_eda", use_container_width=True):
        st.session_state.page = "eda"; st.rerun()
with bn2:
    if st.button("📈 Performance", key="nav_models", use_container_width=True):
        st.session_state.page = "models"; st.rerun()
with bn3:
    if st.button("🔮 Predict", key="nav_predict", use_container_width=True):
        st.session_state.page = "predict"; st.rerun()
st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page=="eda":
    st.markdown("""
    <div class="content-section">
      <div class="section-heading">
        <div class="section-tag">01 — Exploratory Analysis</div>
        <div class="section-title">Reading the Data\'s<br><em style="font-style:italic;color:#C9903C">Hidden Patterns</em></div>
        <div class="section-desc">Six visual investigations into what drives hotel booking cancellations — from outlier detection to correlation structure.</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#4ECDC4,#3DBDB5);--kval:#4ECDC4"><span class="kpi-tile-icon">🏨</span><span class="kpi-tile-val">{n_records:,}</span><span class="kpi-tile-lbl">Total Bookings</span><span class="kpi-tile-sub">Across city &amp; resort hotels</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#E8735A,#D4624A);--kval:#E8735A"><span class="kpi-tile-icon">📉</span><span class="kpi-tile-val">{cancel_rate:.1f}%</span><span class="kpi-tile-lbl">Cancellation Rate</span><span class="kpi-tile-sub">Nearly 2 in 5 bookings lost</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#C9903C,#E8B86D);--kval:#E8B86D"><span class="kpi-tile-icon">💰</span><span class="kpi-tile-val">${avg_adr:.0f}</span><span class="kpi-tile-lbl">Avg Daily Rate</span><span class="kpi-tile-sub">Mean ADR across all stays</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#5B8FD4,#4A7EC3);--kval:#5B8FD4"><span class="kpi-tile-icon">🌍</span><span class="kpi-tile-val">{n_countries}</span><span class="kpi-tile-lbl">Guest Countries</span><span class="kpi-tile-sub">Truly global clientele</span></div>
    </div>""", unsafe_allow_html=True)
    pad_l,pad_r=st.columns([1,20])
    with pad_r:
        def chart_target():
            fig,(a1,a2)=dark_fig(13,4,1,2)
            vc=RAW["is_canceled"].value_counts()
            bars=a1.bar(["Honoured","Cancelled"],vc.values,color=[TEAL,RED],edgecolor=BG,width=0.45,zorder=3)
            a1.set_title("Booking Outcome Count",fontweight="bold",pad=12)
            for bar,v in zip(bars,vc.values): a1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+300,f"{v:,}",ha="center",fontsize=11,color=TXT,fontweight="bold")
            a1.set_ylim(0,max(vc.values)*1.15)
            wedges,texts,autotexts=a2.pie(vc.values,labels=["Honoured","Cancelled"],autopct="%1.1f%%",colors=[TEAL,RED],startangle=90,wedgeprops=dict(edgecolor="#0D1322",linewidth=3),textprops={"color":TXT,"fontsize":10,"fontfamily":"Inter"})
            for at in autotexts: at.set_color(BG); at.set_fontweight("bold")
            a2.set_title("Proportion",fontweight="bold",pad=12)
            fig.tight_layout(pad=2); st.pyplot(fig); plt.close()
        viz_card("Target Variable — Cancellation Split","Distribution",chart_target)
        def chart_hotel_deposit():
            fig,(a1,a2)=dark_fig(13,4,1,2)
            hc=RAW.groupby("hotel")["is_canceled"].mean()*100
            bars1=a1.bar(hc.index,hc.values,color=[BLUE,GOLD],edgecolor=BG,width=0.45,zorder=3)
            a1.set_title("By Hotel Type",fontweight="bold",pad=12); a1.set_ylabel("Cancel Rate (%)"); a1.set_ylim(0,max(hc.values)*1.2)
            for bar in bars1: a1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{bar.get_height():.1f}%",ha="center",fontsize=12,color=TXT,fontweight="bold")
            dc=RAW.groupby("deposit_type")["is_canceled"].mean()*100
            bars2=a2.bar(dc.index,dc.values,color=[PURPLE,RED,TEAL],edgecolor=BG,width=0.45,zorder=3)
            a2.set_title("By Deposit Type",fontweight="bold",pad=12); a2.set_ylabel("Cancel Rate (%)"); a2.set_ylim(0,max(dc.values)*1.2)
            for bar in bars2: a2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.8,f"{bar.get_height():.1f}%",ha="center",fontsize=12,color=TXT,fontweight="bold")
            fig.tight_layout(pad=2); st.pyplot(fig); plt.close()
        viz_card("Cancellation Rate by Hotel Type & Deposit Policy","Segmentation",chart_hotel_deposit)
        MONTHS=["January","February","March","April","May","June","July","August","September","October","November","December"]
        def chart_monthly():
            fig,(a1,a2)=dark_fig(14,4,1,2)
            mc=RAW.groupby("arrival_date_month")["is_canceled"].mean()*100; mc=mc.reindex(MONTHS,fill_value=0)
            bars_m=a1.bar(range(12),mc.values,color=RED,edgecolor=BG,alpha=0.8,zorder=3,width=0.65)
            for bar,v in zip(bars_m,mc.values): bar.set_alpha(0.55+0.45*(v/mc.max()))
            a1.set_xticks(range(12)); a1.set_xticklabels([m[:3] for m in MONTHS],rotation=45,ha="right",color=MUTED)
            a1.set_title("Monthly Cancellation Rate (%)",fontweight="bold",pad=12)
            am=RAW.groupby("arrival_date_month")["adr"].mean().reindex(MONTHS)
            a2.plot(range(12),am.values,color=GOLD2,lw=2.5,marker="o",markersize=6,markerfacecolor="#0D1322",markeredgecolor=GOLD2,markeredgewidth=2,zorder=3)
            a2.fill_between(range(12),am.values,alpha=0.08,color=GOLD2)
            a2.set_xticks(range(12)); a2.set_xticklabels([m[:3] for m in MONTHS],rotation=45,ha="right",color=MUTED)
            a2.set_title("Monthly Avg ADR ($)",fontweight="bold",pad=12)
            fig.tight_layout(pad=2); st.pyplot(fig); plt.close()
        viz_card("Monthly Cancellation Rate & Average Daily Rate","Temporal",chart_monthly)
        BOX_COLS=["lead_time","adr","stays_in_week_nights","stays_in_weekend_nights","adults","days_in_waiting_list","previous_cancellations","booking_changes"]
        def chart_boxplots():
            fig,axes=dark_fig(18,7,2,4); axes=axes.flatten()
            colors_box=[TEAL,GOLD2,BLUE,PURPLE,RED,GOLD,TEAL,BLUE]
            for i,col in enumerate(BOX_COLS):
                data=RAW[col].dropna()
                axes[i].boxplot(data,patch_artist=True,widths=0.5,boxprops=dict(facecolor=colors_box[i],alpha=0.3,linewidth=0),medianprops=dict(color=RED,linewidth=2.5),whiskerprops=dict(color=MUTED,linewidth=1),capprops=dict(color=MUTED,linewidth=1),flierprops=dict(marker="o",markerfacecolor=RED,markersize=2,alpha=0.3,linewidth=0))
                axes[i].set_title(col.replace("_"," ").title(),fontweight="bold",fontsize=8.5)
                Q1,Q3=data.quantile(0.25),data.quantile(0.75)
                out=((data<Q1-1.5*(Q3-Q1))|(data>Q3+1.5*(Q3-Q1))).sum()
                axes[i].set_xlabel(f"{out:,} outliers",fontsize=8,color=RED)
            fig.suptitle("Distribution & Outlier Profile",fontsize=12,fontweight="bold",color=TXT,y=1.01)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        viz_card("Outlier Detection — Key Numeric Features","Boxplots",chart_boxplots)
        def chart_distributions():
            fig,(a1,a2)=dark_fig(13,5,1,2)
            g0=RAW[RAW["is_canceled"]==0]["lead_time"].dropna(); g1=RAW[RAW["is_canceled"]==1]["lead_time"].dropna()
            a1.hist(g0,bins=50,alpha=0.65,color=TEAL,edgecolor=BG,label="Honoured",density=True,zorder=3)
            a1.hist(g1,bins=50,alpha=0.6,color=RED,edgecolor=BG,label="Cancelled",density=True,zorder=3)
            a1.set_title("Lead Time by Outcome",fontweight="bold",pad=12); a1.set_xlabel("Lead Time (days)")
            a1.legend(facecolor=CARD2,labelcolor=TXT,edgecolor=BORDER,fontsize=9)
            g0a=RAW[RAW["is_canceled"]==0]["adr"].dropna(); g1a=RAW[RAW["is_canceled"]==1]["adr"].dropna()
            a2.hist(g0a[g0a<600],bins=50,alpha=0.65,color=TEAL,edgecolor=BG,label="Honoured",density=True,zorder=3)
            a2.hist(g1a[g1a<600],bins=50,alpha=0.6,color=RED,edgecolor=BG,label="Cancelled",density=True,zorder=3)
            a2.set_title("ADR by Outcome",fontweight="bold",pad=12); a2.set_xlabel("ADR ($)")
            a2.legend(facecolor=CARD2,labelcolor=TXT,edgecolor=BORDER,fontsize=9)
            fig.tight_layout(pad=2); st.pyplot(fig); plt.close()
        viz_card("Lead Time & ADR vs Cancellation Outcome","Distributions",chart_distributions)
        def chart_heatmap():
            NUM=RAW.drop(columns=["index"],errors="ignore").select_dtypes("number"); corr=NUM.corr()
            fig,ax=dark_fig(14,9); ax.set_facecolor(CARD)
            mask=np.triu(np.ones_like(corr,dtype=bool))
            cmap_custom=LinearSegmentedColormap.from_list("hotel",["#E8735A","#0D1322","#4ECDC4"])
            sns.heatmap(corr,mask=mask,cmap=cmap_custom,vmin=-1,vmax=1,linewidths=0.3,linecolor="#070B14",annot=True,fmt=".2f",annot_kws={"size":6.5,"color":"#F0EAE0","weight":"bold"},ax=ax,cbar_kws={"shrink":.55})
            ax.set_title("Correlation Matrix",fontweight="bold",pad=14,color=TXT); ax.tick_params(colors=MUTED,labelsize=7.5)
            cbar=ax.collections[0].colorbar; cbar.ax.tick_params(colors=MUTED,labelsize=8)
            ax.set_xticklabels(ax.get_xticklabels(),color=MUTED,fontsize=7.5)
            ax.set_yticklabels(ax.get_yticklabels(),color=MUTED,fontsize=7.5)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        viz_card("Feature Correlation Matrix","Heatmap",chart_heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page=="models":
    xgb_before=base_acc["XGBoost"]; xgb_after=proc_acc["XGBoost"]
    improvement=round(xgb_after-xgb_before,2); auc_val=round(roc_auc_score(y_test,best_prob),4)
    sign="+" if improvement>=0 else ""
    st.markdown(f"""
    <div class="content-section">
      <div class="section-heading">
        <div class="section-tag">02 — Model Evaluation</div>
        <div class="section-title">From Baseline to<br><em style="font-style:italic;color:#C9903C">Champion Performance</em></div>
        <div class="section-desc">Six algorithms benchmarked before and after feature engineering. XGBoost emerges as champion with {xgb_after}% accuracy.</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#4ECDC4,#3DBDB5);--kval:#4ECDC4"><span class="kpi-tile-icon">🏆</span><span class="kpi-tile-val">{xgb_after}%</span><span class="kpi-tile-lbl">XGBoost — Post Processing</span><span class="kpi-tile-sub">Tuned · cleaned · engineered</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#5B8FD4,#4A7EC3);--kval:#5B8FD4"><span class="kpi-tile-icon">📋</span><span class="kpi-tile-val">{xgb_before}%</span><span class="kpi-tile-lbl">XGBoost — Baseline</span><span class="kpi-tile-sub">Default params · raw features</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#C9903C,#E8B86D);--kval:#E8B86D"><span class="kpi-tile-icon">📈</span><span class="kpi-tile-val">{sign}{improvement}%</span><span class="kpi-tile-lbl">Uplift from Engineering</span><span class="kpi-tile-sub">Feature &amp; preprocessing gain</span></div>
        <div class="kpi-tile" style="--stripe:linear-gradient(90deg,#9B7FD4,#8A6EC3);--kval:#9B7FD4"><span class="kpi-tile-icon">🎯</span><span class="kpi-tile-val">{auc_val}</span><span class="kpi-tile-lbl">ROC-AUC Score</span><span class="kpi-tile-sub">{len(y_test):,} test samples</span></div>
    </div>""", unsafe_allow_html=True)
    pad_l2,pad_r2=st.columns([1,20])
    with pad_r2:
        st.markdown("""<div class="viz-card-header-block"><span class="viz-card-title">All Models — Before vs After Preprocessing</span><span class="viz-card-badge">Benchmark</span></div><div class="viz-card-body">""", unsafe_allow_html=True)
        rows=[]
        for name in base_acc:
            b=base_acc[name]; a=proc_acc[name]; d=round(a-b,2)
            arrow="↑" if d>0 else ("↓" if d<0 else "→"); tag=" ★ CHAMPION" if name=="XGBoost" else ""
            rows.append({"Model":name+tag,"Before (%)":b,"After (%)":a,"Δ":f"{arrow} {d:+.2f}%"})
        df_table=pd.DataFrame(rows).set_index("Model")
        def hl(row):
            if "CHAMPION" in row.name: return ["background-color:rgba(201,144,60,0.1);color:#E8B86D;font-weight:700"]*len(row)
            return [""]*len(row)
        st.dataframe(df_table.style.apply(hl,axis=1),use_container_width=True)
        st.caption("★ XGBoost champion. Baseline models use raw features; post-processing adds 9 engineered features.")
        st.markdown('</div>', unsafe_allow_html=True)
        def chart_accuracy():
            names=list(base_acc.keys()); bv=[base_acc[n] for n in names]; av=[proc_acc[n] for n in names]
            x,w=np.arange(len(names)),0.35
            before_colors=[BLUE if n=="XGBoost" else "#162340" for n in names]
            after_colors=[TEAL if n=="XGBoost" else "#0a2a1f" for n in names]
            before_alpha=[1.0 if n=="XGBoost" else 0.5 for n in names]
            after_alpha=[1.0 if n=="XGBoost" else 0.5 for n in names]
            fig,ax=dark_fig(13,5)
            b1=[ax.bar(xi-w/2,v,w,color=c,alpha=a,edgecolor=BG,zorder=3) for xi,v,c,a in zip(x,bv,before_colors,before_alpha)]
            b2=[ax.bar(xi+w/2,v,w,color=c,alpha=a,edgecolor=BG,zorder=3) for xi,v,c,a in zip(x,av,after_colors,after_alpha)]
            ax.set_ylim(60,110); ax.set_ylabel("Accuracy (%)")
            ax.set_title("Model Accuracy Comparison  (★ = Champion)",fontweight="bold",pad=12)
            ax.set_xticks(x); ax.set_xticklabels(["★ "+n if n=="XGBoost" else n for n in names],rotation=20,ha="right")
            ax.legend(handles=[mpatches.Patch(color=BLUE,label="Before Preprocessing"),mpatches.Patch(color=TEAL,label="After Preprocessing")],facecolor=CARD2,labelcolor=TXT,edgecolor=BORDER)
            for i,(bar_grp,v) in enumerate(list(zip(b1,bv))+list(zip(b2,av))):
                bar=bar_grp[0]; name=names[i%len(names)]; fs=10 if name=="XGBoost" else 7; col=TXT if name=="XGBoost" else MUTED
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{v}%",ha="center",fontsize=fs,fontweight="bold",color=col)
            xgb_idx=names.index("XGBoost"); xgb_b_bar=b1[xgb_idx][0]; xgb_a_bar=b2[xgb_idx][0]
            ytop=max(xgb_b_bar.get_height(),xgb_a_bar.get_height())+4
            ax.annotate("",xy=(xgb_a_bar.get_x()+xgb_a_bar.get_width()/2,max(xgb_b_bar.get_height(),xgb_a_bar.get_height())+1.5),xytext=(xgb_b_bar.get_x()+xgb_b_bar.get_width()/2,max(xgb_b_bar.get_height(),xgb_a_bar.get_height())+1.5),arrowprops=dict(arrowstyle="<->",color=GOLD2,lw=1.5))
            ax.text(xgb_idx,ytop+0.3,f"XGBoost: {sign}{improvement}%",ha="center",fontsize=9,fontweight="bold",color=GOLD2)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        viz_card("Accuracy Comparison — Before vs After (★ = XGBoost Champion)","Bar Chart",chart_accuracy)
        def chart_cm_roc():
            fig,(a1,a2)=dark_fig(13,5,1,2)
            cm=confusion_matrix(y_test,best_pred); cm_norm=cm.astype(float)/cm.max()
            cmap_cm=LinearSegmentedColormap.from_list("hotel_cm",["#0D1322","#0f2540","#1a4a7a","#2680d4","#4ECDC4"])
            a1.imshow(cm_norm,cmap=cmap_cm,aspect="auto",vmin=0,vmax=1)
            labels=["Honoured","Cancelled"]; a1.set_xticks([0,1]); a1.set_yticks([0,1])
            a1.set_xticklabels(labels,color=MUTED,fontsize=9); a1.set_yticklabels(labels,color=MUTED,fontsize=9)
            a1.set_xlabel("Predicted",color=MUTED,fontsize=10); a1.set_ylabel("Actual",color=MUTED,fontsize=10)
            a1.set_title("Confusion Matrix",fontweight="bold",pad=12,color=TXT); a1.grid(False)
            for i in range(2):
                for j in range(2):
                    tc="#070B14" if cm_norm[i,j]>0.5 else "#E6EDF3"
                    a1.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=14,fontweight="bold",color=tc)
            fpr,tpr,_=roc_curve(y_test,best_prob)
            a2.plot(fpr,tpr,color=TEAL,lw=2.5,label=f"XGBoost AUC = {auc_val}",zorder=3)
            a2.plot([0,1],[0,1],"--",color=DIM,lw=1); a2.fill_between(fpr,tpr,alpha=0.07,color=TEAL)
            a2.set_xlabel("False Positive Rate"); a2.set_ylabel("True Positive Rate")
            a2.set_title("ROC Curve",fontweight="bold",pad=12)
            a2.legend(facecolor=CARD2,labelcolor=TXT,edgecolor=BORDER)
            fig.tight_layout(pad=2); st.pyplot(fig); plt.close()
        viz_card("XGBoost — Confusion Matrix & ROC Curve","Deep Dive",chart_cm_roc)
        def chart_importance():
            fig,ax=dark_fig(11,7); fi=rf_imp.head(20); n=len(fi)
            palette=[plt.cm.RdYlGn(i/(n-1)*0.7+0.1) for i in range(n)][::-1]
            ax.barh(fi.index[::-1],fi.values[::-1],color=palette[::-1],edgecolor=BG,height=0.65,zorder=3)
            ax.set_title("Feature Importances",fontweight="bold",pad=12); ax.set_xlabel("Importance Score")
            fig.tight_layout(); st.pyplot(fig); plt.close()
        viz_card("Top 20 Feature Importances — Random Forest","Explainability",chart_importance)
# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "predict":
    # Page heading
    st.markdown("""
    <div class="content-section">
      <div class="section-heading">
        <div class="section-tag">03 — Live Inference</div>
        <div class="section-title">Score Any Booking<br>
          <em style="font-style:italic;color:#C9903C">In Real Time</em>
        </div>
        <div class="section-desc">
          Configure a hypothetical booking below. The XGBoost champion model
          will return a cancellation probability and risk profile instantly.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Decorative banner
    st.markdown(f"""
    <div style="margin:0 4rem 2rem;height:140px;border-radius:20px;
                position:relative;overflow:hidden;border:1px solid #1E2A3E;">
        <img src="{HOTEL_IMGS[1]}"
             style="width:100%;height:100%;object-fit:cover;
                    opacity:0.4;filter:saturate(0.6);" />
        <div style="position:absolute;inset:0;
                    background:linear-gradient(90deg,#070B14 0%,transparent 30%,
                    transparent 70%,#070B14 100%),
                    linear-gradient(180deg,rgba(7,11,20,0.4) 0%,rgba(7,11,20,0.85) 100%);"></div>
        <div style="position:absolute;inset:0;display:flex;
                    flex-direction:column;align-items:center;justify-content:center;gap:8px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.25em;color:#C9903C;text-transform:uppercase;">
                Booking Risk Calculator
            </div>
            <span style="font-family:'Playfair Display',serif;font-style:italic;
                         font-size:1.7rem;color:rgba(240,234,224,0.9);
                         letter-spacing:0.06em;text-shadow:0 2px 20px rgba(0,0,0,0.8);">
                Configure &nbsp;·&nbsp; Predict &nbsp;·&nbsp; Decide
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ═══ THE MEGA CARD — everything below sits inside ONE container ═══
    st.markdown('<div class="predict-page-frame">', unsafe_allow_html=True)
    with st.container():
        # Sentinel marks this vertical block as the mega-card
        st.markdown('<span class="predict-mega-sentinel"></span>', unsafe_allow_html=True)
        # Mega-card header
        st.markdown("""
        <div class="predict-mega-header">
            <div class="predict-mega-header-title">⚙ Booking Configuration Workspace</div>
            <div class="predict-mega-header-tag">XGBoost Live Inference</div>
        </div>
        """, unsafe_allow_html=True)
        # ── Inputs sub-card ────────────────────────────────────────────
        with st.container():
            st.markdown('<span class="predict-input-sentinel"></span>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<span class="pred-col-sentinel"></span>', unsafe_allow_html=True)
                st.markdown('<div class="pred-card-title">🏨 Booking Details</div>', unsafe_allow_html=True)
                hotel          = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
                deposit_type   = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
                market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct",
                                                                  "Corporate", "Groups", "Complementary", "Aviation"])
                customer_type  = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
                country_input  = st.selectbox("Guest Country", ["PRT","GBR","FRA","ESP","DEU","ITA","IRL",
                                                                 "BEL","BRA","NLD","USA","CHN","Other"])
                lead_time      = st.slider("Lead Time (days before arrival)", 0, 365, 60)
                adr            = st.slider("Average Daily Rate ($)", 0, 500, 100)
                week_nights    = st.slider("Weekday Nights", 0, 14, 2)
            with col2:
                st.markdown('<span class="pred-col-sentinel"></span>', unsafe_allow_html=True)
                st.markdown('<div class="pred-card-title">📋 Stay & Guest Profile</div>', unsafe_allow_html=True)
                reserved_room      = st.selectbox("Reserved Room Type", ["A","B","C","D","E","F","G","H","L","P"])
                assigned_room      = st.selectbox("Assigned Room Type", ["A","B","C","D","E","F","G","H","I","K","L","P"])
                weekend_nights     = st.slider("Weekend Nights", 0, 7, 1)
                adults             = st.slider("Number of Adults", 1, 4, 2)
                prev_cancellations = st.slider("Previous Cancellations", 0, 10, 0)
                prev_not_canceled  = st.slider("Previous Kept Bookings", 0, 10, 0)
                special_requests   = st.slider("Special Requests", 0, 5, 0)
                booking_changes    = st.slider("Booking Changes", 0, 10, 0)
        # ── Risk pre-flight panel ──────────────────────────────────────
        risk_items = []; overall_risk = "low"
        if lead_time > 150:
            risk_items.append(("danger", f"Lead time {lead_time} days — strong cancellation predictor"))
            overall_risk = "high"
        elif lead_time > 90:
            risk_items.append(("warn", f"Lead time {lead_time} days — moderate risk window"))
            if overall_risk == "low": overall_risk = "medium"
        if deposit_type == "Non Refund":
            risk_items.append(("danger", "Non-refundable deposit — historically very high cancel rate"))
            overall_risk = "high"
        if prev_cancellations > 0:
            risk_items.append(("danger", f"{prev_cancellations} prior cancellation(s) — strong behavioural signal"))
            overall_risk = "high"
        if special_requests > 0:
            risk_items.append(("safe", f"{special_requests} special request(s) — engaged, committed guest"))
        if prev_not_canceled > 0:
            risk_items.append(("safe", f"{prev_not_canceled} previous kept booking(s) — loyal returning guest"))
        if reserved_room != assigned_room:
            risk_items.append(("warn", "Room type reassigned — potential guest dissatisfaction"))
            if overall_risk == "low": overall_risk = "medium"
        if risk_items:
            dot_c   = {"danger": RED,  "warn": GOLD,  "safe": TEAL}
            badge_s = {
                "danger": f"background:rgba(232,115,90,0.12);border:1px solid rgba(232,115,90,0.3);color:{RED}",
                "warn":   f"background:rgba(201,144,60,0.12);border:1px solid rgba(201,144,60,0.3);color:{GOLD2}",
                "safe":   f"background:rgba(78,205,196,0.12);border:1px solid rgba(78,205,196,0.3);color:{TEAL}",
            }
            badge_l = {"danger": "High Risk", "warn": "Caution", "safe": "Positive"}
            rows_html = "".join([
                f'<div class="risk-row">'
                f'<div class="risk-dot" style="background:{dot_c[l]}"></div>'
                f'<span class="risk-row-text">{t}</span>'
                f'<span class="risk-badge" style="{badge_s[l]}">{badge_l[l]}</span>'
                f'</div>'
                for l, t in risk_items
            ])
            rb = {"high": RED, "medium": GOLD, "low": TEAL}[overall_risk]
            rt = {"high": "HIGH RISK PROFILE", "medium": "MODERATE RISK PROFILE", "low": "LOW RISK PROFILE"}[overall_risk]
            st.markdown(f"""
            <div class="preflight-panel" style="--risk-color:{rb};">
                <div class="preflight-header">⚑ Pre-Flight Check · {rt}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)
        # ── Predict button ──────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        btn_l, btn_c, btn_r = st.columns([1, 2, 1])
        with btn_c:
            predict_btn = st.button("✦ Run Cancellation Prediction", key="predict_run_btn", use_container_width=True)
        # ── Results inside same mega-card ──────────────────────────────
        if predict_btn:
            total_nights     = weekend_nights + week_nights
            room_changed     = int(reserved_room != assigned_room)
            revenue_estimate = adr * total_nights
            is_long_stay     = int(total_nights > 7)
            is_high_lt       = int(lead_time > 90)
            lead_time_bin    = (0 if lead_time <= 7  else 1 if lead_time <= 30 else
                                2 if lead_time <= 90 else 3 if lead_time <= 180 else 4)
            is_returning     = int(prev_not_canceled > 0)
            input_raw = {
                "hotel": hotel, "lead_time": float(lead_time),
                "arrival_date_year": 2024, "arrival_date_month": "July",
                "arrival_date_week_number": 27, "arrival_date_day_of_month": 15,
                "stays_in_weekend_nights": weekend_nights, "stays_in_week_nights": week_nights,
                "adults": adults, "children": 0.0, "babies": 0, "meal": "BB",
                "country": country_input, "market_segment": market_segment,
                "distribution_channel": "TA/TO", "is_repeated_guest": int(prev_not_canceled > 0),
                "previous_cancellations": prev_cancellations,
                "previous_bookings_not_canceled": prev_not_canceled,
                "reserved_room_type": reserved_room, "assigned_room_type": assigned_room,
                "booking_changes": booking_changes, "deposit_type": deposit_type,
                "agent": -1.0, "days_in_waiting_list": 0, "customer_type": customer_type,
                "adr": float(adr), "required_car_parking_spaces": 0,
                "total_of_special_requests": special_requests,
                "total_nights": total_nights, "total_guests": adults,
                "is_family": 0, "room_changed": room_changed,
                "revenue_estimate": revenue_estimate, "is_long_stay": is_long_stay,
                "is_high_lead_time": is_high_lt, "lead_time_bin": lead_time_bin,
                "is_returning_guest": is_returning,
            }
            row = pd.DataFrame([input_raw])
            lo_lt, hi_lt = winsor_bounds["lead_time"]
            lo_a,  hi_a  = winsor_bounds["adr"]
            lo_w,  hi_w  = winsor_bounds["days_in_waiting_list"]
            row["lead_time"]            = float(np.clip(lead_time, lo_lt, hi_lt))
            row["adr"]                  = float(np.clip(adr,       lo_a,  hi_a))
            row["days_in_waiting_list"] = float(np.clip(0,         lo_w,  hi_w))
            row["adults"]               = min(adults, 4)
            row["meal"]                 = row["meal"].replace("Undefined", "SC")
            if country_input not in ["PRT","GBR","FRA","ESP","DEU","ITA","IRL",
                                      "BEL","BRA","NLD","USA","CHN"]:
                row["country"] = "Other"
            row["is_high_lead_time"] = int(row["lead_time"].iloc[0] > 90)
            cl = row["lead_time"].iloc[0]
            row["lead_time_bin"] = (0 if cl<=7 else 1 if cl<=30 else
                                    2 if cl<=90 else 3 if cl<=180 else 4)
            for col in cat_cols:
                if col in row.columns:
                    enc = encoders[col]; val = row[col].astype(str).iloc[0]
                    row[col] = enc.transform([val])[0] if val in enc.classes_ else 0
            for col in feature_cols:
                if col not in row.columns: row[col] = 0
            row = row[feature_cols].fillna(0)
            pred        = best_model.predict(row)[0]
            prob        = best_model.predict_proba(row)[0]
            cancel_prob = round(float(prob[1]) * 100, 1)
            keep_prob   = round(float(prob[0]) * 100, 1)
            st.markdown("""
            <div style="border-top:1px solid #1E2A3E;margin:24px 0 20px;padding-top:20px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                             color:#C9903C;letter-spacing:0.2em;text-transform:uppercase;
                             margin-bottom:8px;">◆ Prediction Output</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                             font-weight:600;color:#F0EAE0;">Champion model verdict</div>
            </div>
            """, unsafe_allow_html=True)
            r_left, r_right = st.columns([1, 2], gap="large")
            with r_left:
                cls = "cancel" if pred == 1 else "keep"
                col = RED if pred == 1 else TEAL
                pct = cancel_prob if pred == 1 else keep_prob
                lbl = "Cancellation Predicted" if pred == 1 else "Booking Will Be Honoured"
                sub = "cancellation probability" if pred == 1 else "will-keep probability"
                ico = "✗" if pred == 1 else "✓"
                st.markdown(f"""
                <div class="verdict-card {cls}">
                    <div class="verdict-icon">{ico}</div>
                    <div class="verdict-label" style="color:{col}">{lbl}</div>
                    <div class="verdict-pct"   style="color:{col}">{pct}%</div>
                    <div class="verdict-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
            with r_right:
                def ri(label, color, value):
                    return (f'<div class="risk-row">'
                            f'<div class="risk-dot" style="background:{color}"></div>'
                            f'<span class="risk-row-text">{label}</span>'
                            f'<span style="font-family:Inter,sans-serif;font-size:0.82rem;'
                            f'color:#F0EAE0">{value}</span></div>')
                st.markdown(f"""
                <div class="risk-analysis">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                 color:#3D4F6B;letter-spacing:0.18em;text-transform:uppercase;
                                 margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #1E2A3E;">
                        Booking Profile Summary
                    </div>
                    {ri("Hotel",          GOLD,                                        hotel)}
                    {ri("Lead Time",      RED if lead_time > 90 else TEAL,             f"{lead_time} days")}
                    {ri("Deposit",        RED if deposit_type == "Non Refund" else TEAL, deposit_type)}
                    {ri("Stay Duration",  GOLD,                                        f"{total_nights} night(s)")}
                    {ri("Past Cancels",   RED if prev_cancellations > 0 else TEAL,     str(prev_cancellations))}
                    {ri("Spec. Requests", TEAL if special_requests > 0 else DIM,       str(special_requests))}
                    {ri("ADR",            GOLD,                                        f"${adr}")}
                    {ri("Room Changed",   RED if room_changed else TEAL,               "Yes" if room_changed else "No")}
                </div>""", unsafe_allow_html=True)
            # Probability gauge
            st.markdown(f"""
            <div class="gauge-wrap">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#3D4F6B;
                             letter-spacing:0.18em;text-transform:uppercase;margin-bottom:8px;">
                    Probability Gauge
                </div>
                <div class="gauge-bar">
                    <div class="gauge-fill-keep"   style="width:{keep_prob}%"></div>
                    <div class="gauge-fill-cancel" style="width:{cancel_prob}%"></div>
                </div>
                <div class="gauge-labels">
                    <span style="color:{TEAL}">Keep: {keep_prob}%</span>
                    <span style="color:{RED}">Cancel: {cancel_prob}%</span>
                </div>
            </div>""", unsafe_allow_html=True)
            # 3 metric cards
            vt = "Will Cancel" if pred == 1 else "Will Honour"
            vi = "✗" if pred == 1 else "✓"
            vc = RED if pred == 1 else TEAL
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);
                        gap:14px;margin-top:16px;">
                <div style="background:#0D1322;border:1px solid #1E2A3E;
                             border-radius:14px;padding:22px 20px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                 color:#3D4F6B;letter-spacing:0.12em;text-transform:uppercase;
                                 margin-bottom:10px;">Cancellation Probability</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                                 font-weight:700;color:{RED};line-height:1;">{cancel_prob}%</div>
                </div>
                <div style="background:#0D1322;border:1px solid #1E2A3E;
                             border-radius:14px;padding:22px 20px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                 color:#3D4F6B;letter-spacing:0.12em;text-transform:uppercase;
                                 margin-bottom:10px;">Will-Keep Probability</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                                 font-weight:700;color:{TEAL};line-height:1;">{keep_prob}%</div>
                </div>
                <div style="background:#0D1322;border:1px solid {vc}44;
                             border-left:3px solid {vc};border-radius:14px;padding:22px 20px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                 color:#3D4F6B;letter-spacing:0.12em;text-transform:uppercase;
                                 margin-bottom:10px;">Prediction Verdict</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.8rem;
                                 font-weight:700;color:{vc};line-height:1;white-space:nowrap;">
                        {vi} {vt}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # close .predict-page-frame
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:5rem;border-top:1px solid #1E2A3E;padding:3rem 4rem 2.5rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
    <div>
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#F0EAE0;margin-bottom:4px;">LUXE · Hotel Cancellation Intelligence</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#3D4F6B;letter-spacing:0.12em;">CHAMPION MODEL: XGBOOST · {proc_acc['XGBoost']}% ACCURACY · {len(y_test):,} TEST SAMPLES</div>
    </div>
    <div style="display:flex;gap:24px;align-items:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#3D4F6B;">BINARY CLASSIFICATION</div>
        <div style="width:1px;height:20px;background:#1E2A3E;"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#3D4F6B;">6 ALGORITHMS BENCHMARKED</div>
        <div style="width:1px;height:20px;background:#1E2A3E;"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#3D4F6B;">v3.0 · LUXE EDITION</div>
    </div>
</div>""", unsafe_allow_html=True)
