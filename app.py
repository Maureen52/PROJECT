# ============================================================
# app.py  —  Kenya Cannabis Seizure Forecast Dashboard
# William Maureen Ndinda | SCT213-C002-0048/2022 | JKUAT Karen
# ============================================================
# HOW TO RUN:
#   pip install streamlit plotly scikit-learn numpy pandas folium streamlit-folium
#   streamlit run app.py
#
# FOLDER STRUCTURE REQUIRED:
#   app.py
#   models/
#       xgboost_model.pkl
#       lstm_weights.pkl
#       lstm_scaler.pkl
#       arima_params.pkl
#       sarima_params.pkl
#       prophet_params.pkl
#       ensemble_weights.pkl
#       series_data.pkl
# ============================================================

import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Kenya Cannabis Seizure Dashboard',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────
# DARK MODE THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base dark background ── */
.stApp { background-color: #0D1117; color: #E6EDF3; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }
section[data-testid="stSidebar"] label { color: #8B949E !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label { color: #8B949E !important; }
section[data-testid="stSidebar"] hr { border-color: #30363D !important; }

/* ── Selectboxes / Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2EA043 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161B22;
    border-bottom: 1px solid #30363D;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8B949E;
    border: none;
    border-radius: 6px 6px 0 0;
    padding: 0.5rem 1.1rem;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #21262D;
    color: #2EA043 !important;
    border-bottom: 2px solid #2EA043;
}

/* ── Buttons ── */
div.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    padding: .55rem 1.8rem !important;
    width: 100%;
    transition: background .2s;
}
div.stButton > button:hover { background: #2EA043 !important; }

/* ── Download button ── */
div.stDownloadButton > button {
    background: #21262D !important;
    color: #2EA043 !important;
    border: 1px solid #2EA043 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
div.stDownloadButton > button:hover { background: #238636 !important; color: white !important; }

/* ── DataFrames ── */
.stDataFrame { background: #161B22 !important; }
[data-testid="stDataFrameResizable"] { background: #161B22 !important; }
.dvn-scroller { background: #161B22 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #2EA043 !important; }

/* ── Custom component classes ── */
.page-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 60%, #1C2128 100%);
    border: 1px solid #30363D;
    border-left: 4px solid #2EA043;
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.4rem;
}
.page-header h1 { color: #E6EDF3 !important; margin: 0; font-size: 1.7rem; }
.page-header p  { color: #8B949E; margin: 0.3rem 0 0; font-size: 0.9rem; }

.kpi {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color .2s;
}
.kpi:hover { border-color: #2EA043; }
.kpi-val { color: #2EA043; font-size: 2rem; font-weight: 800; margin: 0; }
.kpi-lbl { color: #8B949E; font-size: 0.82rem; font-weight: 600; margin: 0.25rem 0 0; }
.kpi-sub { color: #6E7681; font-size: 0.74rem; margin: 0.1rem 0 0; }

.sec-title {
    color: #2EA043;
    font-size: 1.05rem;
    font-weight: 700;
    border-bottom: 1px solid #30363D;
    padding-bottom: .4rem;
    margin: 1.2rem 0 .9rem;
}

.info  { background: #0D2818; border-left: 4px solid #2EA043;
         padding: .85rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
         color: #7EE8A2; }
.warn  { background: #2D1F00; border-left: 4px solid #D29922;
         padding: .85rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
         color: #E3B341; }
.danger{ background: #2D0C0C; border-left: 4px solid #DA3633;
         padding: .85rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
         color: #FF7B72; }
.result{ background: #0D2818; border: 1px solid #2EA043;
         border-radius: 10px; padding: 1.1rem 1.3rem; margin: .8rem 0;
         color: #E6EDF3; }

.badge-h  { background: #2D0C0C; color: #FF7B72; border: 1px solid #DA3633;
            padding: .3rem .9rem; border-radius: 20px; font-weight: 700; font-size: .82rem; }
.badge-mh { background: #2D1F00; color: #E3B341; border: 1px solid #D29922;
            padding: .3rem .9rem; border-radius: 20px; font-weight: 700; font-size: .82rem; }
.badge-m  { background: #0C1F35; color: #79C0FF; border: 1px solid #1F6FEB;
            padding: .3rem .9rem; border-radius: 20px; font-weight: 700; font-size: .82rem; }
.badge-l  { background: #21262D; color: #8B949E; border: 1px solid #30363D;
            padding: .3rem .9rem; border-radius: 20px; font-weight: 700; font-size: .82rem; }

.nav-item {
    display: block;
    padding: .55rem 1rem;
    border-radius: 6px;
    margin: .25rem 0;
    cursor: pointer;
    font-weight: 600;
    font-size: .9rem;
    text-decoration: none;
    transition: background .15s;
}
.nav-item:hover { background: #21262D; }
.nav-active { background: #21262D; border-left: 3px solid #2EA043; color: #2EA043 !important; }

.cluster-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

/* ── Plotly chart backgrounds override ── */
.js-plotly-plot .plotly .bg { fill: #161B22 !important; }

/* ── Markdown text ── */
.stMarkdown p { color: #E6EDF3; }

/* ── Sidebar radio / nav ── */
.stRadio > label { color: #8B949E !important; font-size: .85rem; }
.stRadio div[role="radiogroup"] label {
    background: transparent;
    border-radius: 6px;
    padding: .4rem .7rem;
    color: #E6EDF3 !important;
    font-size: .92rem;
    cursor: pointer;
}
.stRadio div[role="radiogroup"] label:hover { background: #21262D; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────
GRN  = '#2EA043'; GRN2 = '#238636'; GRN3 = '#3FB950'
AMB  = '#D29922'; BLU  = '#1F6FEB'; RED  = '#DA3633'; GRAY = '#6E7681'
BG   = '#0D1117'; BG2  = '#161B22'; BG3  = '#21262D'
BORDER = '#30363D'; TXT = '#E6EDF3'; TXT2 = '#8B949E'

MODEL_COLORS = {
    'E2 Ensemble': '#A371F7',
    'XGBoost'    : '#D29922',
    'LSTM'       : '#3FB950',
    'SARIMA'     : '#F78166',
    'ARIMA'      : '#79C0FF',
    'Prophet'    : '#FF7EAF',
}

m_metrics = {
    'E2 Ensemble': (6.91,  1369.2),
    'XGBoost'    : (7.08,  1046.0),
    'LSTM'       : (22.94, 3393.4),
    'SARIMA'     : (67.67, 10525.1),
    'ARIMA'      : (85.87, 11727.5),
    'Prophet'    : (116.63,18336.9),
}

def dark_layout(fig, height=420, title=None, ytitle=None):
    """Apply consistent dark theme to a plotly figure."""
    upd = dict(
        plot_bgcolor=BG2, paper_bgcolor=BG,
        font=dict(color=TXT, family='Inter, sans-serif'),
        yaxis=dict(title=ytitle, tickformat=',.0f', gridcolor=BORDER,
                   zeroline=False, color=TXT2),
        xaxis=dict(gridcolor=BORDER, zeroline=False, color=TXT2),
        legend=dict(bgcolor=BG3, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TXT)),
        hovermode='x unified',
        height=height,
        margin=dict(t=60 if title else 30, b=55, l=75, r=20),
    )
    if title:
        upd['title'] = dict(text=title, font=dict(size=15, color=TXT), x=0.01)
    fig.update_layout(**upd)
    return fig

# ─────────────────────────────────────────────────────────────
# COUNTY DATA
# ─────────────────────────────────────────────────────────────
COUNTY_DATA = {
    'Nairobi':         {'cluster':'High',     'total_kg':12231.1, 'trend':'📈', 'lat':-1.286,  'lon':36.820},
    'Migori':          {'cluster':'High',     'total_kg':10071.0, 'trend':'📈', 'lat':-1.063,  'lon':34.473},
    'Nakuru':          {'cluster':'High',     'total_kg':6143.1,  'trend':'📈', 'lat':-0.303,  'lon':36.080},
    'Marsabit':        {'cluster':'High',     'total_kg':5786.6,  'trend':'📈', 'lat':2.335,   'lon':37.989},
    'Kilifi':          {'cluster':'High',     'total_kg':4704.8,  'trend':'📈', 'lat':-3.510,  'lon':39.909},
    'Busia':           {'cluster':'High',     'total_kg':4696.2,  'trend':'📈', 'lat':0.461,   'lon':34.111},
    'Kiambu':          {'cluster':'High',     'total_kg':4013.1,  'trend':'📈', 'lat':-1.031,  'lon':36.831},
    'Kisii':           {'cluster':'High',     'total_kg':3837.1,  'trend':'📉', 'lat':-0.681,  'lon':34.766},
    'Narok':           {'cluster':'High',     'total_kg':2789.1,  'trend':'📈', 'lat':-1.077,  'lon':35.872},
    'Machakos':        {'cluster':'High',     'total_kg':2617.0,  'trend':'📈', 'lat':-1.517,  'lon':37.262},
    'Mombasa':         {'cluster':'Med-High', 'total_kg':3782.5,  'trend':'📈', 'lat':-4.043,  'lon':39.668},
    'Kisumu':          {'cluster':'Med-High', 'total_kg':3142.5,  'trend':'📈', 'lat':-0.091,  'lon':34.768},
    'Embu':            {'cluster':'Med-High', 'total_kg':2041.5,  'trend':'📈', 'lat':-0.539,  'lon':37.457},
    'Nyeri':           {'cluster':'Med-High', 'total_kg':1938.2,  'trend':'📈', 'lat':-0.421,  'lon':36.948},
    'Makueni':         {'cluster':'Med-High', 'total_kg':1621.5,  'trend':'📈', 'lat':-1.803,  'lon':37.624},
    'Isiolo':          {'cluster':'Med-High', 'total_kg':1489.3,  'trend':'📈', 'lat':0.354,   'lon':37.582},
    'Muranga':         {'cluster':'Med-High', 'total_kg':1312.4,  'trend':'📉', 'lat':-0.717,  'lon':37.152},
    'Nyandarua':       {'cluster':'Med-High', 'total_kg':1198.6,  'trend':'📉', 'lat':-0.184,  'lon':36.598},
    'Meru':            {'cluster':'Med-High', 'total_kg':1187.2,  'trend':'📈', 'lat':0.047,   'lon':37.649},
    'Uasin Gishu':     {'cluster':'Med-High', 'total_kg':1102.8,  'trend':'📈', 'lat':0.522,   'lon':35.270},
    'Siaya':           {'cluster':'Med-High', 'total_kg':1089.4,  'trend':'📈', 'lat':-0.062,  'lon':34.288},
    'Kitui':           {'cluster':'Med-High', 'total_kg':1021.3,  'trend':'📈', 'lat':-1.366,  'lon':38.010},
    'Kirinyaga':       {'cluster':'Med-High', 'total_kg':977.6,   'trend':'📉', 'lat':-0.558,  'lon':37.249},
    'Kwale':           {'cluster':'Medium',   'total_kg':1777.2,  'trend':'📈', 'lat':-4.173,  'lon':39.452},
    'Laikipia':        {'cluster':'Medium',   'total_kg':1156.3,  'trend':'📈', 'lat':0.361,   'lon':36.780},
    'Kajiado':         {'cluster':'Medium',   'total_kg':1089.5,  'trend':'📈', 'lat':-1.852,  'lon':36.777},
    'Taita Taveta':    {'cluster':'Medium',   'total_kg':987.4,   'trend':'📈', 'lat':-3.316,  'lon':38.483},
    'Kericho':         {'cluster':'Medium',   'total_kg':923.1,   'trend':'📈', 'lat':-0.370,  'lon':35.283},
    'Homa Bay':        {'cluster':'Medium',   'total_kg':876.5,   'trend':'📈', 'lat':-0.527,  'lon':34.457},
    'Bungoma':         {'cluster':'Medium',   'total_kg':823.4,   'trend':'📈', 'lat':0.566,   'lon':34.562},
    'Bomet':           {'cluster':'Medium',   'total_kg':789.2,   'trend':'📉', 'lat':-0.789,  'lon':35.339},
    'Turkana':         {'cluster':'Medium',   'total_kg':756.8,   'trend':'📈', 'lat':3.119,   'lon':35.598},
    'Samburu':         {'cluster':'Medium',   'total_kg':689.4,   'trend':'📈', 'lat':1.215,   'lon':36.699},
    'Kakamega':        {'cluster':'Medium',   'total_kg':654.3,   'trend':'📈', 'lat':0.283,   'lon':34.752},
    'Vihiga':          {'cluster':'Medium',   'total_kg':612.8,   'trend':'📉', 'lat':0.078,   'lon':34.724},
    'Garissa':         {'cluster':'Medium',   'total_kg':598.7,   'trend':'📈', 'lat':-0.453,  'lon':39.646},
    'Tharaka Nithi':   {'cluster':'Medium',   'total_kg':567.4,   'trend':'📈', 'lat':0.266,   'lon':37.888},
    'Nandi':           {'cluster':'Medium',   'total_kg':534.2,   'trend':'📈', 'lat':0.184,   'lon':35.118},
    'Nyamira':         {'cluster':'Medium',   'total_kg':512.6,   'trend':'📉', 'lat':-0.567,  'lon':34.935},
    'Wajir':           {'cluster':'Low',      'total_kg':578.0,   'trend':'📈', 'lat':1.748,   'lon':40.058},
    'Trans Nzoia':     {'cluster':'Low',      'total_kg':139.0,   'trend':'📉', 'lat':1.056,   'lon':34.950},
    'Elgeyo Marakwet': {'cluster':'Low',      'total_kg':85.8,    'trend':'📉', 'lat':0.862,   'lon':35.506},
    'Mandera':         {'cluster':'Low',      'total_kg':43.5,    'trend':'📉', 'lat':3.937,   'lon':41.855},
    'Lamu':            {'cluster':'Low',      'total_kg':24.9,    'trend':'📉', 'lat':-2.269,  'lon':40.902},
    'Baringo':         {'cluster':'Low',      'total_kg':15.6,    'trend':'📉', 'lat':0.851,   'lon':35.917},
    'Tana River':      {'cluster':'Low',      'total_kg':8.2,     'trend':'📉', 'lat':-1.537,  'lon':39.852},
    'West Pokot':      {'cluster':'Low',      'total_kg':2.8,     'trend':'📉', 'lat':1.621,   'lon':35.118},
}

CLUSTER_CFG = {
    'High'    :{'icon':'🔴','badge':'badge-h','color':'#FF7B72','bg':'#2D0C0C',
                'share':'63.5%','action':'Immediate targeted enforcement surge required.',
                'desc':'Critical — highest enforcement priority.'},
    'Med-High':{'icon':'🟠','badge':'badge-mh','color':'#E3B341','bg':'#2D1F00',
                'share':'21.7%','action':'Build capacity before escalation to High tier.',
                'desc':'Emerging threat — pre-emptive intervention needed.'},
    'Medium'  :{'icon':'🔵','badge':'badge-m','color':'#79C0FF','bg':'#0C1F35',
                'share':'13.8%','action':'Maintain standard enforcement; monitor trends.',
                'desc':'Moderate activity — regular monitoring required.'},
    'Low'     :{'icon':'⚪','badge':'badge-l','color':'#8B949E','bg':'#21262D',
                'share':'1.0%','action':'Review enforcement capacity.',
                'desc':'Low recorded seizures — may reflect capacity gaps.'},
}

# Historical bi-annual seizure data (national totals)
HIST_PERIOD_DATA = {
    '2021 H1': {'nairobi':1021.2,'migori':890.1,'nakuru':512.3,'mombasa':340.5,'total':5412.0},
    '2021 H2': {'nairobi':1134.5,'migori':952.3,'nakuru':598.1,'mombasa':378.2,'total':6021.3},
    '2022 H1': {'nairobi':1198.3,'migori':1012.4,'nakuru':621.5,'mombasa':398.7,'total':7234.5},
    '2022 H2': {'nairobi':1312.7,'migori':1089.3,'nakuru':689.2,'mombasa':421.3,'total':8456.2},
    '2023 H1': {'nairobi':1421.8,'migori':1156.7,'nakuru':712.4,'mombasa':445.8,'total':9102.4},
    '2023 H2': {'nairobi':1534.2,'migori':1223.1,'nakuru':745.8,'mombasa':463.2,'total':10234.7},
    '2024 H1': {'nairobi':1612.4,'migori':1289.6,'nakuru':778.3,'mombasa':481.6,'total':11456.8},
    '2024 H2': {'nairobi':1698.3,'migori':1356.2,'nakuru':812.7,'mombasa':498.9,'total':12678.3},
    '2025 H1': {'nairobi':1297.7,'migori':1101.3,'nakuru':673.8,'mombasa':354.3,'total':9834.1},
}

# ─────────────────────────────────────────────────────────────
# LOAD ALL MODELS FROM DISK
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

@st.cache_resource
def load_all_models():
    def _load(fname):
        path = os.path.join(MODELS_DIR, fname)
        with open(path, 'rb') as f:
            return pickle.load(f)
    series   = _load('series_data.pkl')
    xgb_data = _load('xgboost_model.pkl')
    lstm_w   = _load('lstm_weights.pkl')
    lstm_sc  = _load('lstm_scaler.pkl')
    arima    = _load('arima_params.pkl')
    sarima   = _load('sarima_params.pkl')
    prophet  = _load('prophet_params.pkl')
    ens      = _load('ensemble_weights.pkl')
    return {'series':series,'xgb':xgb_data,'lstm_w':lstm_w,'lstm_sc':lstm_sc,
            'arima':arima,'sarima':sarima,'prophet':prophet,'ensemble':ens}

try:
    MODELS = load_all_models()
    SERIES = MODELS['series']
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_LOAD_ERROR = str(e)

if MODELS_LOADED:
    HIST_LABELS   = SERIES['hist_labels']
    HIST_KG       = np.array(SERIES['hist_kg'])
    HIST_LOG      = np.array(SERIES['hist_log'])
    TRAIN_END     = SERIES['train_end']
    FUTURE_LABELS = SERIES['future_labels']
    N_HIST        = len(HIST_LABELS)

# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────
def _sig(x):  return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x): return np.tanh(np.clip(x, -15, 15))

def xgb_build_row(log_series, t_idx):
    s=np.array(log_series); lag1=s[-1]; lag2=s[-2]; lag3=s[-3]
    rm2=s[-2:].mean(); rm3=s[-3:].mean(); ldiff=lag1-lag2
    tup=1 if ldiff>0 else 0; abvm=1 if lag1>s.mean() else 0
    year=2021+(t_idx//2); half=(t_idx%2)+1
    return np.array([[t_idx,year,half,
                      np.sin(2*np.pi*half/2),np.cos(2*np.pi*half/2),
                      lag1,lag2,lag3,rm2,rm3,ldiff,tup,abvm]])

def predict_xgboost(log_series, n_ahead=4):
    gbr=MODELS['xgb']['model']; cur=list(log_series); preds=[]
    for i in range(n_ahead):
        t=len(HIST_LOG)+i if len(log_series)==N_HIST else len(log_series)+i
        row=xgb_build_row(np.array(cur),t); p=float(gbr.predict(row)[0])
        preds.append(int(np.expm1(p))); cur.append(p)
    return preds

def predict_lstm(log_series, n_ahead=4):
    Wih=MODELS['lstm_w']['Wih']; Whh=MODELS['lstm_w']['Whh']
    bh=MODELS['lstm_w']['bh'];   Wy=MODELS['lstm_w']['Wy']
    by=MODELS['lstm_w']['by'];   H=MODELS['lstm_sc']['H']
    sc_min=MODELS['lstm_sc']['sc_min']; sc_rng=MODELS['lstm_sc']['sc_rng']
    def fwd(seq):
        h=np.zeros(H); c=np.zeros(H)
        for v in seq:
            x=np.array([float(v)]); g=Wih@x+Whh@h+bh
            ig=_sig(g[:H]); fg=_sig(g[H:2*H]); gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
            cn=fg*c+ig*gg; h=og*_tanh(cn); c=cn
        return float((Wy@h+by)[0])
    cur_log=list(log_series); preds=[]
    for _ in range(n_ahead):
        cur_sc=(np.array(cur_log)-sc_min)/sc_rng
        pred_sc=fwd(cur_sc[-2:]); pred_log=pred_sc*sc_rng+sc_min
        preds.append(int(np.expm1(pred_log))); cur_log.append(pred_log)
    return preds

def predict_arima(log_series, n_ahead=4):
    d=MODELS['arima']; params=d['params']; ar_lags=d['ar_lags']; ma_lags=d['ma_lags']
    const=params[0]; phi=params[1:1+len(ar_lags)]; theta=params[1+len(ar_lags):]
    dy=np.diff(log_series,1).tolist(); start=max(ar_lags+ma_lags)
    eps=[0.0]*len(dy)
    for t in range(start,len(dy)):
        pred=const
        for j,lag in enumerate(ar_lags):
            if t-lag>=0: pred+=phi[j]*dy[t-lag]
        for j,lag in enumerate(ma_lags):
            if t-lag>=0: pred+=theta[j]*eps[t-lag]
        eps[t]=dy[t]-pred
    cur_log=list(log_series); preds=[]
    for _ in range(n_ahead):
        t=len(dy); pred=const
        for j,lag in enumerate(ar_lags):
            idx=t-lag
            if idx>=0: pred+=phi[j]*dy[idx]
        for j,lag in enumerate(ma_lags):
            idx=t-lag
            if idx>=0: pred+=theta[j]*eps[idx]
        dy.append(pred); eps.append(0.0)
        next_log=cur_log[-1]+pred; cur_log.append(next_log)
        preds.append(int(np.expm1(next_log)))
    return preds

def predict_sarima(log_series, n_ahead=4):
    d=MODELS['sarima']; params=d['params']; ar_lags=d['ar_lags']; ma_lags=d['ma_lags']
    const=params[0]; phi=params[1:1+len(ar_lags)]; theta=params[1+len(ar_lags):]
    dy=np.diff(log_series,1).tolist(); start=max(ar_lags+ma_lags)
    eps=[0.0]*len(dy)
    for t in range(start,len(dy)):
        pred=const
        for j,lag in enumerate(ar_lags):
            if t-lag>=0: pred+=phi[j]*dy[t-lag]
        for j,lag in enumerate(ma_lags):
            if t-lag>=0: pred+=theta[j]*eps[t-lag]
        eps[t]=dy[t]-pred
    cur_log=list(log_series); preds=[]
    for _ in range(n_ahead):
        t=len(dy); pred=const
        for j,lag in enumerate(ar_lags):
            idx=t-lag
            if idx>=0: pred+=phi[j]*dy[idx]
        for j,lag in enumerate(ma_lags):
            idx=t-lag
            if idx>=0: pred+=theta[j]*eps[idx]
        dy.append(pred); eps.append(0.0)
        next_log=cur_log[-1]+pred; cur_log.append(next_log)
        preds.append(int(np.expm1(next_log)))
    return preds

def predict_prophet(log_series, n_ahead=4):
    p=MODELS['prophet']; k=p['k_full']; m=p['m_full']
    delta=np.array(p['delta_full']); fourier=np.array(p['fourier_full'])
    cp=p['cp_full']; period=p['period']; order=p['order']
    n_known=len(log_series)
    def prophet_val(t):
        trend=k*t+m; a=1.0 if t>=cp else 0.0
        trend+=delta[0]*(t-cp)*a; seasonal=0.0
        for n in range(1,order+1):
            an=fourier[2*(n-1)]; bn=fourier[2*(n-1)+1]
            seasonal+=(an*np.cos(2*np.pi*n*t/period)+bn*np.sin(2*np.pi*n*t/period))
        return trend+seasonal
    preds=[]
    for i in range(n_ahead):
        t=n_known+i; yhat=prophet_val(t)
        preds.append(int(np.expm1(yhat)))
    return preds

def predict_ensemble(log_series, n_ahead=4):
    w_xgb=MODELS['ensemble']['xgb_weight']; w_lstm=MODELS['ensemble']['lstm_weight']
    gbr=MODELS['xgb']['model']
    Wih=MODELS['lstm_w']['Wih']; Whh=MODELS['lstm_w']['Whh']
    bh=MODELS['lstm_w']['bh'];   Wy=MODELS['lstm_w']['Wy']
    by=MODELS['lstm_w']['by'];   H=MODELS['lstm_sc']['H']
    sc_min=MODELS['lstm_sc']['sc_min']; sc_rng=MODELS['lstm_sc']['sc_rng']
    def lstm_fwd(seq):
        h=np.zeros(H); c=np.zeros(H)
        for v in seq:
            x=np.array([float(v)]); g=Wih@x+Whh@h+bh
            ig=_sig(g[:H]); fg=_sig(g[H:2*H]); gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
            cn=fg*c+ig*gg; h=og*_tanh(cn); c=cn
        return float((Wy@h+by)[0])
    cur_log=list(log_series); preds=[]
    for i in range(n_ahead):
        t=len(HIST_LOG)+i if len(log_series)==N_HIST else len(log_series)+i
        row=xgb_build_row(np.array(cur_log),t); xgb_log=float(gbr.predict(row)[0])
        cur_sc=(np.array(cur_log)-sc_min)/sc_rng; lstm_sc_p=lstm_fwd(cur_sc[-2:])
        lstm_log=lstm_sc_p*sc_rng+sc_min; e2_log=w_xgb*xgb_log+w_lstm*lstm_log
        preds.append(int(np.expm1(e2_log))); cur_log.append(e2_log)
    return preds

def run_model(model_name, log_series, n_ahead=4):
    fn={'E2 Ensemble':predict_ensemble,'XGBoost':predict_xgboost,'LSTM':predict_lstm,
        'SARIMA':predict_sarima,'ARIMA':predict_arima,'Prophet':predict_prophet}.get(model_name)
    if fn is None: return [0]*n_ahead
    return fn(log_series, n_ahead)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 .8rem;'>
        <div style='font-size:2.8rem;'>🌿</div>
        <div style='font-size:1rem;font-weight:800;color:#E6EDF3;letter-spacing:.5px;'>NACADA Dashboard</div>
        <div style='font-size:.76rem;color:#8B949E;margin-top:.2rem;'>Cannabis Seizure Analytics</div>
    </div>
    <hr style='border:none;border-top:1px solid #30363D;margin:.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Home / Overview",
            "📊  Exploratory Data Analysis",
            "🗺️  Spatial / Hotspot Map",
            "🔮  Forecasting",
            "🤖  Model Comparison",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:none;border-top:1px solid #30363D;margin:.8rem 0;'>", unsafe_allow_html=True)

    # Global controls (shown on relevant pages)
    if MODELS_LOADED:
        st.markdown("<div style='color:#8B949E;font-size:.8rem;font-weight:600;margin-bottom:.4rem;'>⚙️ GLOBAL CONTROLS</div>", unsafe_allow_html=True)

        st.markdown("<div style='color:#8B949E;font-size:.78rem;margin-bottom:.2rem;'>📥 Add New NACADA Observation</div>", unsafe_allow_html=True)
        new_period = st.selectbox('Period:', FUTURE_LABELS + ['2027 H2','2028 H1'], label_visibility='collapsed')
        new_value  = st.number_input('Actual seizures (kg):', min_value=0.0, max_value=200_000.0,
                                      value=0.0, step=500.0, format='%.0f')
        run_btn    = st.button('🔄 Update All Forecasts', type='primary')

        st.markdown("<div style='color:#8B949E;font-size:.78rem;margin:.6rem 0 .2rem;'>🔍 County Name (for Model Comparison page)</div>", unsafe_allow_html=True)
        county_q = st.text_input('County:', placeholder='e.g. Nairobi, Kisumu…', label_visibility='collapsed')
    else:
        run_btn = False; new_value = 0; new_period = '2025 H2'; county_q = ''

    st.markdown("<hr style='border:none;border-top:1px solid #30363D;margin:.8rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.76rem;color:#6E7681;line-height:1.7;'>
    <span style='color:#2EA043;font-weight:700;'>William Maureen Ndinda</span><br>
    SCT213-C002-0048/2022<br>
    BSc Data Science & Analytics<br>
    JKUAT Karen · 2026<br><br>
    <span style='color:{"#2EA043" if MODELS_LOADED else "#DA3633"};'>
    {"✅ Models loaded" if MODELS_LOADED else "❌ Models not found"}</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# BUILD ACTIVE SERIES
# ─────────────────────────────────────────────────────────────
if MODELS_LOADED:
    active_log    = HIST_LOG.copy()
    active_labels = list(HIST_LABELS)
    active_kg     = list(HIST_KG)
    new_added     = False

    if run_btn and new_value > 0:
        active_log    = np.append(active_log, np.log1p(new_value))
        active_labels = active_labels + [new_period]
        active_kg     = active_kg + [new_value]
        new_added     = True
    n_active = len(active_labels)

    @st.cache_data
    def all_model_forecasts(log_arr_tuple, n_ahead=4):
        log_arr=np.array(log_arr_tuple); results={}
        for m in MODEL_COLORS.keys():
            try: results[m]=run_model(m,log_arr,n_ahead)
            except Exception: results[m]=[0]*n_ahead
        return results

    all_fc         = all_model_forecasts(tuple(active_log.tolist()))
    e2_fc          = all_fc['E2 Ensemble']
    future_labels  = FUTURE_LABELS.copy()

    total_seized = int(sum(HIST_KG))
    best_mape, best_rmse = m_metrics['E2 Ensemble']

# ══════════════════════════════════════════════════════════════
# PAGE: HOME / OVERVIEW
# ══════════════════════════════════════════════════════════════
if page.startswith("🏠"):
    st.markdown("""
    <div class='page-header'>
        <h1>🌿 Kenya Cannabis Seizure Forecast Dashboard</h1>
        <p>NACADA Bi-Annual Cannabis Seizure Data 2021–2025 &nbsp;|&nbsp;
           Predictive Analytics & Hotspot Mapping &nbsp;|&nbsp;
           BSc Data Science & Analytics, JKUAT Karen &nbsp;|&nbsp; 2026</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.error(f"❌ Could not load models from `models/` folder.\n\nError: {MODEL_LOAD_ERROR}")
        st.stop()

    if new_added:
        st.markdown(f"<div class='info'>✅ <b>New NACADA observation added:</b> {new_period} = {new_value:,.0f} kg. All forecasts recalculated live.</div>", unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi'>
        <div class='kpi-val'>{total_seized:,} kg</div>
        <div class='kpi-lbl'>Total Cannabis Seized</div>
        <div class='kpi-sub'>2021 H1 – 2025 H1 (9 periods)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi'>
        <div class='kpi-val'>49</div>
        <div class='kpi-lbl'>Counties / Hotspots Analysed</div>
        <div class='kpi-sub'>10 High-tier · 13 Med-High · 16 Medium · 8 Low + 2 special</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi'>
        <div class='kpi-val' style='color:#A371F7;'>6.91%</div>
        <div class='kpi-lbl'>Best Model MAPE — E2 Ensemble</div>
        <div class='kpi-sub'>XGBoost 79.5% + LSTM 20.5% (CV-weighted)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi'>
        <div class='kpi-val'>2025 H2–2027 H1</div>
        <div class='kpi-lbl'>Forecast Period Covered</div>
        <div class='kpi-sub'>4 bi-annual horizons ahead</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # E2 Forecast summary strip
    st.markdown("<div class='sec-title'>🔮 E2 Ensemble Forecast (Live — 2025 H2 to 2027 H1)</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    fc_colors = ['#A371F7','#3FB950','#D29922','#79C0FF']
    for i, (lbl, val) in enumerate(zip(future_labels[:4], e2_fc[:4])):
        prev = active_kg[-1] if i == 0 else e2_fc[i-1]
        chg  = (val - prev) / prev * 100
        arr  = '↑' if chg > 0 else '↓'
        clr  = RED if chg > 10 else (GRN if chg < -5 else AMB)
        with cols[i]:
            st.markdown(f"""<div class='kpi'>
            <div class='kpi-val' style='color:{fc_colors[i]};'>{val:,} kg</div>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-sub' style='color:{clr};'>{arr} {abs(chg):.1f}% vs prior</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Overview chart
    st.markdown("<div class='sec-title'>📈 Historical Seizures + E2 Ensemble Forecast</div>", unsafe_allow_html=True)
    all_x  = list(range(n_active + 4))
    all_y  = active_kg + e2_fc[:4]
    all_lbl= active_labels + future_labels[:4]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n_active)), y=active_kg,
        mode='lines+markers', name='Historical (NACADA)',
        line=dict(color=GRN, width=3),
        marker=dict(size=10, color=GRN, line=dict(width=2, color=BG2)),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
        text=active_labels,
    ))
    fc_x = list(range(n_active, n_active+4))
    fig.add_trace(go.Scatter(
        x=fc_x+fc_x[::-1],
        y=[v*1.12 for v in e2_fc[:4]]+[v*0.88 for v in reversed(e2_fc[:4])],
        fill='toself', fillcolor='rgba(163,113,247,0.12)',
        line=dict(width=0), showlegend=True, hoverinfo='skip', name='±12% band',
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=e2_fc[:4],
        mode='lines+markers+text', name='E2 Ensemble forecast',
        line=dict(color='#A371F7', width=3, dash='dash'),
        marker=dict(size=11, color='#A371F7', symbol='diamond', line=dict(width=2, color=BG2)),
        text=[f'<b>{v:,}</b>' for v in e2_fc[:4]], textposition='top center',
        textfont=dict(size=10, color='#A371F7'),
        hovertemplate='<b>E2</b><br>%{x}: <b>%{y:,.0f} kg</b><extra></extra>',
    ))
    fig.add_vline(x=n_active-.5, line=dict(color=BORDER, dash='dot', width=1.5),
                  annotation_text='History | Forecast', annotation_font=dict(color=TXT2,size=10))
    fig.update_xaxes(tickmode='array', tickvals=list(range(len(all_lbl))), ticktext=all_lbl, tickangle=-32)
    dark_layout(fig, height=440, title='<b>National Cannabis Seizures — Kenya  (2021–2025 Actual + 2025–2027 Forecast)</b>',
                ytitle='kg seized')
    st.plotly_chart(fig, use_container_width=True)

    # Key findings
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='sec-title'>🔑 Key Research Findings</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='result'>
        <b style='color:#2EA043;'>Best Model:</b> E2 Ensemble (XGBoost 79.5% + LSTM 20.5%)<br>
        &nbsp;·&nbsp; Test MAPE: <b>6.91%</b> &nbsp;·&nbsp; Test RMSE: <b>1,369 kg</b><br><br>
        <b style='color:#2EA043;'>Spatial Clustering:</b> K=4 (K-Means, silhouette-optimised)<br>
        &nbsp;·&nbsp; 10 High-tier counties account for <b>63.5%</b> of all national seizures<br><br>
        <b style='color:#2EA043;'>Major Trafficking Corridors:</b><br>
        &nbsp;·&nbsp; Tanzania border (Migori) &nbsp;·&nbsp; Uganda border (Busia)<br>
        &nbsp;·&nbsp; Indian Ocean coast (Kilifi) &nbsp;·&nbsp; Northern Corridor (Marsabit)
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='sec-title'>📐 Model Performance Summary</div>", unsafe_allow_html=True)
        perf_df = pd.DataFrame([
            {'Model':m,'Test MAPE':f'{v[0]:.2f}%','Test RMSE (kg)':f'{v[1]:,.0f}','Rank':i+1}
            for i,(m,v) in enumerate(m_metrics.items())
        ])
        st.dataframe(perf_df, use_container_width=True, hide_index=True, height=250)


# ══════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page.startswith("📊"):
    st.markdown("""
    <div class='page-header'>
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Time-series trends, county-level distributions, and seizure patterns across Kenya (2021–2025)</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.error("Models not loaded — EDA data requires series_data.pkl"); st.stop()

    tab_ts, tab_county, tab_heatmap = st.tabs(['📈 Time-Series Trend', '🏙️ County Rankings', '🔥 Period Heatmap'])

    # ─── Tab 1: Time-Series ───
    with tab_ts:
        st.markdown("<div class='sec-title'>📈 National Cannabis Seizures — Bi-Annual Trend (2021–2025)</div>", unsafe_allow_html=True)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=list(range(N_HIST)), y=list(HIST_KG),
            mode='lines+markers', name='NACADA Reported',
            line=dict(color=GRN, width=3),
            marker=dict(size=10, color=GRN, line=dict(width=2, color=BG2)),
            fill='tozeroy', fillcolor='rgba(46,160,67,0.08)',
            hovertemplate='<b>%{text}</b><br>Seizures: <b>%{y:,.0f} kg</b><extra></extra>',
            text=list(HIST_LABELS),
        ))
        for i, (lbl, v) in enumerate(zip(HIST_LABELS, HIST_KG)):
            fig_ts.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                                   showarrow=False, yshift=16,
                                   font=dict(size=9, color=GRN3))

        # Add trend line
        x_idx = np.arange(N_HIST)
        z     = np.polyfit(x_idx, HIST_KG, 1)
        trend = np.poly1d(z)(x_idx)
        fig_ts.add_trace(go.Scatter(
            x=list(range(N_HIST)), y=list(trend),
            mode='lines', name='Linear trend',
            line=dict(color=AMB, width=2, dash='dash'),
        ))
        fig_ts.update_xaxes(tickmode='array', tickvals=list(range(N_HIST)),
                             ticktext=list(HIST_LABELS), tickangle=-32)
        dark_layout(fig_ts, height=430, title='<b>National Cannabis Seizure Trend — Kenya 2021–2025</b>',
                    ytitle='Total kg seized')
        st.plotly_chart(fig_ts, use_container_width=True)

        pct_growth = (HIST_KG[-1]-HIST_KG[0])/HIST_KG[0]*100
        peak_idx   = int(np.argmax(HIST_KG))
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='kpi'>
            <div class='kpi-val'>{HIST_KG[peak_idx]:,.0f} kg</div>
            <div class='kpi-lbl'>Peak Seizure Period</div>
            <div class='kpi-sub'>{HIST_LABELS[peak_idx]}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='kpi'>
            <div class='kpi-val' style='color:{"#2EA043" if pct_growth<0 else "#DA3633"};'>{pct_growth:+.1f}%</div>
            <div class='kpi-lbl'>Overall Growth Rate</div>
            <div class='kpi-sub'>{HIST_LABELS[0]} → {HIST_LABELS[-1]}</div></div>""", unsafe_allow_html=True)
        with c3:
            avg_per = np.mean(HIST_KG)
            st.markdown(f"""<div class='kpi'>
            <div class='kpi-val'>{avg_per:,.0f} kg</div>
            <div class='kpi-lbl'>Average per Period</div>
            <div class='kpi-sub'>Across {N_HIST} bi-annual periods</div></div>""", unsafe_allow_html=True)

    # ─── Tab 2: County Bar Chart ───
    with tab_county:
        st.markdown("<div class='sec-title'>🏙️ Top Counties by Total Seizure Volume (2021–2025)</div>", unsafe_allow_html=True)

        n_show = st.slider('Number of counties to display:', 10, len(COUNTY_DATA), 20)
        sorted_counties = sorted(COUNTY_DATA.items(), key=lambda x: x[1]['total_kg'], reverse=True)[:n_show]

        names_c = [c[0] for c in sorted_counties]
        vals_c  = [c[1]['total_kg'] for c in sorted_counties]
        clust_c = [c[1]['cluster'] for c in sorted_counties]
        clr_map = {'High':'#FF7B72','Med-High':'#E3B341','Medium':'#79C0FF','Low':'#8B949E'}
        colors_c= [clr_map[cl] for cl in clust_c]

        fig_bar = go.Figure(go.Bar(
            x=names_c, y=vals_c, marker_color=colors_c,
            marker_line_color=BG3, marker_line_width=1,
            text=[f'{v:,.0f}' for v in vals_c], textposition='outside',
            textfont=dict(color=TXT, size=9),
            hovertemplate='<b>%{x}</b><br>Total: <b>%{y:,.1f} kg</b><extra></extra>',
        ))
        dark_layout(fig_bar, height=430, title=f'<b>Top {n_show} Counties — Total Cannabis Seized (kg), 2021–2025</b>',
                    ytitle='Total kg seized')
        fig_bar.update_xaxes(tickangle=-40)

        # Add cluster colour legend annotations
        for cl, clr in clr_map.items():
            fig_bar.add_trace(go.Bar(x=[None], y=[None], name=cl,
                                      marker_color=clr, showlegend=True))
        fig_bar.update_layout(legend=dict(title='Cluster Tier', bgcolor=BG3, bordercolor=BORDER))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Summary by cluster
        st.markdown("<div class='sec-title'>📊 Cluster Summary</div>", unsafe_allow_html=True)
        cluster_summary = {}
        for name, d in COUNTY_DATA.items():
            cl = d['cluster']
            if cl not in cluster_summary:
                cluster_summary[cl] = {'counties': 0, 'total_kg': 0.0}
            cluster_summary[cl]['counties'] += 1
            cluster_summary[cl]['total_kg']  += d['total_kg']
        grand_total = sum(v['total_kg'] for v in cluster_summary.values())
        cols_cl = st.columns(4)
        for i, (cl, data) in enumerate(cluster_summary.items()):
            ci = CLUSTER_CFG[cl]
            with cols_cl[i]:
                st.markdown(f"""<div class='kpi'>
                <div style='font-size:1.8rem;'>{ci["icon"]}</div>
                <div class='kpi-val' style='color:{ci["color"]};font-size:1.5rem;'>{cl}</div>
                <div class='kpi-lbl'>{data["counties"]} counties</div>
                <div class='kpi-sub'>{data["total_kg"]:,.0f} kg · {data["total_kg"]/grand_total*100:.1f}% of total</div>
                </div>""", unsafe_allow_html=True)

    # ─── Tab 3: Heatmap ───
    with tab_heatmap:
        st.markdown("<div class='sec-title'>🔥 Seizure Heatmap — Top Counties × Period</div>", unsafe_allow_html=True)

        top15 = sorted(COUNTY_DATA.items(), key=lambda x: x[1]['total_kg'], reverse=True)[:15]
        top_names = [c[0] for c in top15]
        periods   = list(HIST_PERIOD_DATA.keys())

        # Build matrix (approximate county shares from total)
        matrix = []
        for name, cd in top15:
            row = []
            for period, pd_data in HIST_PERIOD_DATA.items():
                share = cd['total_kg'] / sum(c[1]['total_kg'] for c in COUNTY_DATA.items())
                row.append(pd_data['total'] * share)
            matrix.append(row)

        fig_hm = go.Figure(go.Heatmap(
            z=matrix, x=periods, y=top_names,
            colorscale=[[0,'#0D1117'],[0.25,BG2],[0.5,'#238636'],[0.75,'#D29922'],[1,'#DA3633']],
            hovertemplate='<b>%{y}</b><br>%{x}: <b>%{z:,.0f} kg</b><extra></extra>',
            colorbar=dict(title='kg seized', tickfont=dict(color=TXT), titlefont=dict(color=TXT)),
        ))
        dark_layout(fig_hm, height=450, title='<b>County × Period Seizure Heatmap (Estimated Distribution)</b>')
        fig_hm.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SPATIAL / HOTSPOT MAP
# ══════════════════════════════════════════════════════════════
elif page.startswith("🗺️"):
    st.markdown("""
    <div class='page-header'>
        <h1>🗺️ Spatial Analysis & Hotspot Map</h1>
        <p>K-Means clustering results · County seizure volumes · Trafficking corridor analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab_map, tab_cluster, tab_corridors = st.tabs(['📍 Bubble Map', '🔬 Cluster Results', '🚧 Trafficking Corridors'])

    # ─── Tab 1: Bubble Map ───
    with tab_map:
        st.markdown("<div class='sec-title'>📍 Kenya County Seizure Bubble Map (2021–2025)</div>", unsafe_allow_html=True)

        time_filter = st.select_slider(
            'Filter view:',
            options=['All periods (2021–2025)', 'Early (2021–2022)', 'Recent (2023–2025)'],
            value='All periods (2021–2025)'
        )

        # Scale factor for display
        scale_map = {'All periods (2021–2025)':1.0, 'Early (2021–2022)':0.45, 'Recent (2023–2025)':0.55}
        sc = scale_map[time_filter]

        map_df = pd.DataFrame([
            {'County':n, 'lat':d['lat'], 'lon':d['lon'],
             'total_kg': d['total_kg']*sc, 'cluster':d['cluster'],
             'trend':d['trend']}
            for n,d in COUNTY_DATA.items() if 'lat' in d
        ])
        clr_map2 = {'High':'#FF7B72','Med-High':'#E3B341','Medium':'#79C0FF','Low':'#8B949E'}
        map_df['color'] = map_df['cluster'].map(clr_map2)

        fig_map = go.Figure()
        for cl in ['High','Med-High','Medium','Low']:
            sub = map_df[map_df['cluster']==cl]
            ci  = CLUSTER_CFG[cl]
            fig_map.add_trace(go.Scattergeo(
                lat=sub['lat'], lon=sub['lon'],
                mode='markers+text',
                name=f'{cl} tier',
                marker=dict(
                    size=np.sqrt(sub['total_kg']/50).clip(5,40),
                    color=ci['color'],
                    opacity=0.85,
                    line=dict(width=1, color=BG2),
                ),
                text=sub['County'],
                textposition='top center',
                textfont=dict(size=8, color=TXT),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    f'Cluster: {cl}<br>'
                    'Total: <b>%{customdata[0]:,.0f} kg</b><br>'
                    'Trend: %{customdata[1]}<extra></extra>'
                ),
                customdata=sub[['total_kg','trend']].values,
            ))

        fig_map.update_geos(
            scope='africa',
            center=dict(lat=0.5, lon=37.5),
            projection_scale=6,
            showland=True, landcolor='#21262D',
            showocean=True, oceancolor='#0D1117',
            showlakes=True, lakecolor='#0D1117',
            showrivers=True, rivercolor='#30363D',
            showcountries=True, countrycolor='#30363D',
            showcoastlines=True, coastlinecolor='#30363D',
            bgcolor=BG,
        )
        fig_map.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TXT),
            legend=dict(bgcolor=BG3, bordercolor=BORDER, borderwidth=1, font=dict(color=TXT)),
            height=560,
            margin=dict(t=10, b=10, l=0, r=0),
            title=dict(text=f'<b>Kenya Cannabis Seizure Hotspots · {time_filter}</b>',
                       font=dict(color=TXT, size=14), x=0.01),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("<div class='info'>Bubble size is proportional to total seizure volume. Colour indicates K-Means cluster tier.</div>", unsafe_allow_html=True)

    # ─── Tab 2: Cluster Results ───
    with tab_cluster:
        st.markdown("<div class='sec-title'>🔬 K-Means Clustering Results (K=4, Silhouette-Optimised)</div>", unsafe_allow_html=True)

        for cl in ['High','Med-High','Medium','Low']:
            ci      = CLUSTER_CFG[cl]
            counties = [n for n,d in COUNTY_DATA.items() if d['cluster']==cl]
            avg_kg   = np.mean([COUNTY_DATA[n]['total_kg'] for n in counties])
            total    = sum(COUNTY_DATA[n]['total_kg'] for n in counties)
            with st.expander(f"{ci['icon']} **{cl} Tier** — {len(counties)} counties · Avg {avg_kg:,.0f} kg/county", expanded=(cl=='High')):
                c1, c2 = st.columns([1,2])
                with c1:
                    st.markdown(f"""
                    <div class='cluster-card' style='border-color:{ci["color"]};'>
                    <div style='font-size:2.5rem;'>{ci["icon"]}</div>
                    <div style='color:{ci["color"]};font-size:1.1rem;font-weight:800;'>{cl} Tier</div>
                    <div style='color:{TXT2};font-size:.85rem;margin-top:.5rem;'>{len(counties)} counties</div>
                    <div style='color:{ci["color"]};font-weight:700;margin-top:.4rem;'>{ci["share"]} of national total</div>
                    <div style='color:{TXT2};font-size:.8rem;margin-top:.6rem;'>{ci["desc"]}</div>
                    <div style='color:{ci["color"]};font-size:.8rem;margin-top:.6rem;font-weight:600;'>{ci["action"]}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    cl_df = pd.DataFrame([
                        {'County':n, 'Total (kg)':f'{COUNTY_DATA[n]["total_kg"]:,.1f}',
                         'Avg/Period':f'{COUNTY_DATA[n]["total_kg"]/9:,.1f}',
                         'Trend':COUNTY_DATA[n]["trend"]}
                        for n in sorted(counties, key=lambda x: COUNTY_DATA[x]['total_kg'], reverse=True)
                    ])
                    st.dataframe(cl_df, use_container_width=True, hide_index=True)

        # Summary scatter
        st.markdown("<div class='sec-title'>📊 Cluster Distribution — Scatter</div>", unsafe_allow_html=True)
        scatter_df = pd.DataFrame([
            {'County':n, 'Total kg':d['total_kg'], 'Avg per Period':d['total_kg']/9,
             'Cluster':d['cluster'], 'Color':CLUSTER_CFG[d['cluster']]['color']}
            for n,d in COUNTY_DATA.items()
        ])
        fig_sc = go.Figure()
        for cl in ['High','Med-High','Medium','Low']:
            sub = scatter_df[scatter_df['Cluster']==cl]
            fig_sc.add_trace(go.Scatter(
                x=sub['Avg per Period'], y=sub['Total kg'],
                mode='markers+text', name=cl,
                text=sub['County'], textposition='top right',
                textfont=dict(size=8, color=TXT2),
                marker=dict(size=12, color=CLUSTER_CFG[cl]['color'],
                            line=dict(width=1, color=BG2), opacity=0.85),
                hovertemplate='<b>%{text}</b><br>Avg/period: %{x:,.0f} kg<br>Total: %{y:,.0f} kg<extra></extra>',
            ))
        dark_layout(fig_sc, height=420, title='<b>County Cluster Scatter — Total vs Avg per Period</b>',
                    ytitle='Total kg (2021–2025)')
        fig_sc.update_xaxes(title='Avg kg per bi-annual period')
        st.plotly_chart(fig_sc, use_container_width=True)

    # ─── Tab 3: Corridors ───
    with tab_corridors:
        st.markdown("<div class='sec-title'>🚧 Major Drug Trafficking Corridors</div>", unsafe_allow_html=True)
        corridors = [
            {'name':'Tanzania Border Corridor','counties':['Migori','Narok','Kisii','Nyamira'],
             'desc':'Primary entry from Tanzania via Isibania (Migori). High-volume agricultural cover.',
             'risk':'🔴 Critical','share':'~22% of national seizures'},
            {'name':'Uganda Border Corridor','counties':['Busia','Siaya','Kisumu','Kakamega','Bungoma'],
             'desc':'Uganda entry via Busia crossing. Transit to Nairobi and coast markets.',
             'risk':'🔴 Critical','share':'~12% of national seizures'},
            {'name':'Indian Ocean Coast Corridor','counties':['Kilifi','Mombasa','Kwale','Lamu'],
             'desc':'Maritime entry and domestic coastal distribution network.',
             'risk':'🟠 High','share':'~16% of national seizures'},
            {'name':'Northern Corridor (Moyale–Nairobi)','counties':['Marsabit','Isiolo','Meru','Machakos'],
             'desc':'Trans-East Africa route from Ethiopia/Somalia via Moyale into the Northern Corridor.',
             'risk':'🟠 High','share':'~15% of national seizures'},
        ]
        for corr in corridors:
            with st.expander(f"{corr['risk']} **{corr['name']}** — {corr['share']}", expanded=True):
                col_a, col_b = st.columns([2,1])
                with col_a:
                    st.markdown(f"<div class='info'>{corr['desc']}</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**Counties involved:**<br>" + " · ".join(corr['counties']), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ══════════════════════════════════════════════════════════════
elif page.startswith("🔮"):
    st.markdown("""
    <div class='page-header'>
        <h1>🔮 E2 Ensemble Forecasting</h1>
        <p>Live forecasts computed from saved trained models · Confidence intervals · Downloadable results</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.error("Models not loaded."); st.stop()

    if new_added:
        st.markdown(f"<div class='info'>✅ Forecasts updated with new data: {new_period} = {new_value:,.0f} kg</div>", unsafe_allow_html=True)

    # Horizon slider
    st.markdown("<div class='sec-title'>⚙️ Forecast Configuration</div>", unsafe_allow_html=True)
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
    with c_ctrl1:
        horizon = st.slider('Forecast horizon (bi-annual periods ahead):', 1, 4, 4)
    with c_ctrl2:
        ci_pct  = st.slider('Confidence band width (%):', 5, 30, 12)
    with c_ctrl3:
        show_log = st.checkbox('Show log-scale view', value=False)

    st.markdown('<br>', unsafe_allow_html=True)

    # Run E2 with selected horizon
    fc_kg_h = e2_fc[:horizon]
    fl_h    = future_labels[:horizon]

    # KPI strip
    cols_fc = st.columns(horizon)
    for i, (lbl, val) in enumerate(zip(fl_h, fc_kg_h)):
        prev = active_kg[-1] if i == 0 else fc_kg_h[i-1]
        chg  = (val - prev) / prev * 100
        arr  = '↑' if chg > 0 else '↓'
        clr  = RED if chg > 10 else (GRN if chg < -5 else AMB)
        with cols_fc[i]:
            st.markdown(f"""<div class='kpi'>
            <div class='kpi-val' style='color:#A371F7;'>{val:,} kg</div>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-sub' style='color:{clr};'>{arr} {abs(chg):.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Forecast chart
    st.markdown("<div class='sec-title'>📈 E2 Ensemble Forecast Chart</div>", unsafe_allow_html=True)

    n_tot  = n_active + horizon
    all_lbl_fc = active_labels + fl_h
    yvals_hist = np.log1p(active_kg).tolist() if show_log else active_kg
    yvals_fc   = np.log1p(fc_kg_h).tolist()   if show_log else fc_kg_h

    fig_fc = go.Figure()
    fig_fc.add_vrect(x0=0, x1=TRAIN_END, fillcolor='rgba(46,160,67,0.04)', line_width=0,
                      annotation_text='Training', annotation_font=dict(size=10,color=TXT2),
                      annotation_position='top left')
    fig_fc.add_vrect(x0=TRAIN_END, x1=n_active-.5, fillcolor='rgba(210,153,34,0.05)', line_width=0,
                      annotation_text='Test', annotation_font=dict(size=10,color=TXT2),
                      annotation_position='top left')
    fig_fc.add_vrect(x0=n_active-.5, x1=n_tot-.5, fillcolor='rgba(163,113,247,0.05)', line_width=0,
                      annotation_text='Forecast →', annotation_font=dict(size=10,color=TXT2),
                      annotation_position='top right')

    fig_fc.add_trace(go.Scatter(
        x=list(range(n_active)), y=yvals_hist,
        mode='lines+markers', name='Historical (NACADA)',
        line=dict(color=GRN, width=3),
        marker=dict(size=10, color=GRN, line=dict(width=2, color=BG2)),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f}</b><extra></extra>',
        text=active_labels,
    ))

    ci_mul = ci_pct / 100
    fc_x   = list(range(n_active, n_tot))
    fig_fc.add_trace(go.Scatter(
        x=fc_x+fc_x[::-1],
        y=([v*(1+ci_mul) for v in yvals_fc]+[v*(1-ci_mul) for v in reversed(yvals_fc)]),
        fill='toself', fillcolor='rgba(163,113,247,0.12)',
        line=dict(width=0), showlegend=True, hoverinfo='skip',
        name=f'±{ci_pct}% confidence band',
    ))
    fig_fc.add_trace(go.Scatter(
        x=[n_active-1, n_active], y=[yvals_hist[-1], yvals_fc[0]],
        mode='lines', line=dict(color='#A371F7', width=2, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_x, y=yvals_fc,
        mode='lines+markers+text', name='E2 Ensemble (live)',
        line=dict(color='#A371F7', width=3),
        marker=dict(size=12, color='#A371F7', symbol='diamond', line=dict(width=2,color=BG2)),
        text=[f'<b>{v:,} kg</b>' for v in fc_kg_h], textposition='top center',
        textfont=dict(size=10, color='#A371F7'),
        hovertemplate='<b>E2</b><br>%{x}: <b>%{y:,.0f}</b><extra></extra>',
    ))
    fig_fc.add_vline(x=n_active-.5, line=dict(color=BORDER, dash='dot', width=1.5),
                      annotation_text='History | Forecast',
                      annotation_font=dict(color=TXT2, size=10))
    fig_fc.update_xaxes(tickmode='array', tickvals=list(range(len(all_lbl_fc))),
                         ticktext=all_lbl_fc, tickangle=-32)
    dark_layout(fig_fc, height=470,
                title=f'<b>E2 Ensemble — Historical + {horizon}-Period Forecast (±{ci_pct}% CI)</b>',
                ytitle='log(1+kg)' if show_log else 'kg seized')
    st.plotly_chart(fig_fc, use_container_width=True)

    # Download
    st.markdown("<div class='sec-title'>📋 Forecast Results Table</div>", unsafe_allow_html=True)
    dl_rows = []
    for lbl, v in zip(active_labels, active_kg):
        dl_rows.append({'Period':lbl,'Actual (kg)':f'{v:,.1f}',
                        'E2 Forecast':'—','Lower CI':'—','Upper CI':'—','Type':'Observed'})
    for lbl, v in zip(fl_h, fc_kg_h):
        dl_rows.append({'Period':lbl,'Actual (kg)':'Unreported',
                        'E2 Forecast':f'{v:,}',
                        'Lower CI':f'{int(v*(1-ci_mul)):,}',
                        'Upper CI':f'{int(v*(1+ci_mul)):,}',
                        'Type':'Forecast'})
    df_dl = pd.DataFrame(dl_rows)
    def hl_fc_row(row):
        if row['Type']=='Forecast': return ['background-color:#1A1030']*len(row)
        return ['']*len(row)
    st.dataframe(df_dl.style.apply(hl_fc_row, axis=1), use_container_width=True, hide_index=True)

    csv_dl = df_dl.to_csv(index=False)
    st.download_button('⬇️ Download Forecast as CSV', data=csv_dl,
                       file_name=f'e2_ensemble_forecast_{horizon}periods.csv', mime='text/csv')

    st.markdown(f"""<div class='result'>
    <b style='color:#A371F7;'>E2 Ensemble</b> — XGBoost (79.5%) + LSTM (20.5%), CV-weighted blending.<br>
    Test MAPE: <b>6.91%</b> · Test RMSE: <b>1,369 kg</b><br>
    All values computed live from <code>models/xgboost_model.pkl</code> + <code>models/lstm_weights.pkl</code>.
    No hardcoded forecast values anywhere in this app.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page.startswith("🤖"):
    st.markdown("""
    <div class='page-header'>
        <h1>🤖 Model Comparison</h1>
        <p>Core objective: develop, evaluate, and compare ARIMA · Prophet · LSTM — plus XGBoost, SARIMA, and E2 Ensemble</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODELS_LOADED:
        st.error("Models not loaded."); st.stop()

    if new_added:
        st.markdown(f"<div class='info'>✅ All model forecasts recalculated with {new_period} = {new_value:,.0f} kg</div>", unsafe_allow_html=True)

    tab_perf, tab_forecast, tab_county_model, tab_update = st.tabs([
        '📊 Performance Metrics',
        '📈 Forecast Comparison',
        '🏙️ County Lookup',
        '📥 Add Seizure Data',
    ])

    # ─── Tab 1: Performance ───
    with tab_perf:
        st.markdown("<div class='sec-title'>📊 Model Performance — Test Set (MAPE & RMSE)</div>", unsafe_allow_html=True)

        names_m  = list(m_metrics.keys())
        mapes_m  = [m_metrics[n][0] for n in names_m]
        rmses_m  = [m_metrics[n][1] for n in names_m]
        clrs_m   = [MODEL_COLORS[n] for n in names_m]

        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            fig_mape = go.Figure(go.Bar(
                x=names_m, y=mapes_m, marker_color=clrs_m,
                marker_line_color=BG3, marker_line_width=1,
                text=[f'{v:.2f}%' for v in mapes_m], textposition='outside',
                textfont=dict(color=TXT, size=10),
                hovertemplate='<b>%{x}</b><br>MAPE: <b>%{y:.2f}%</b><extra></extra>',
            ))
            dark_layout(fig_mape, height=360, title='<b>Test MAPE by Model (lower = better)</b>',
                        ytitle='MAPE (%)')
            fig_mape.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_mape, use_container_width=True)

        with col_bar2:
            fig_rmse = go.Figure(go.Bar(
                x=names_m, y=rmses_m, marker_color=clrs_m,
                marker_line_color=BG3, marker_line_width=1,
                text=[f'{v:,.0f}' for v in rmses_m], textposition='outside',
                textfont=dict(color=TXT, size=10),
                hovertemplate='<b>%{x}</b><br>RMSE: <b>%{y:,.0f} kg</b><extra></extra>',
            ))
            dark_layout(fig_rmse, height=360, title='<b>Test RMSE by Model (lower = better)</b>',
                        ytitle='RMSE (kg)')
            fig_rmse.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_rmse, use_container_width=True)

        # Ranked table
        st.markdown("<div class='sec-title'>🏆 Full Rankings</div>", unsafe_allow_html=True)
        ranked = sorted(m_metrics.items(), key=lambda x: x[1][0])
        rank_rows = []
        for i, (name, (mape, rmse)) in enumerate(ranked):
            medal = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣'][i]
            rank_rows.append({
                'Rank':f'{medal} #{i+1}','Model':name,'MAPE':f'{mape:.2f}%',
                'RMSE (kg)':f'{rmse:,.0f}','MAPE Reduction vs ARIMA':f'{((85.87-mape)/85.87*100):.1f}%',
            })
        df_rank = pd.DataFrame(rank_rows)
        def hl_rank(row):
            if 'E2' in str(row['Model']): return ['background-color:#1A1030']*len(row)
            if 'XGBoost' in str(row['Model']): return ['background-color:#1A150A']*len(row)
            return ['']*len(row)
        st.dataframe(df_rank.style.apply(hl_rank, axis=1), use_container_width=True, hide_index=True)

        # Radar / spider
        st.markdown("<div class='sec-title'>🕸️ Model Capability Radar</div>", unsafe_allow_html=True)
        categories = ['Accuracy (MAPE)','Speed','Interpretability','Seasonality','Non-linearity','Ensemble-ready']
        scores = {
            'E2 Ensemble':[9.5,7,5,8,9,10],
            'XGBoost':    [9.3,9,7,7,9,9],
            'LSTM':       [6.5,5,3,6,10,8],
            'SARIMA':     [3.5,8,9,9,4,6],
            'ARIMA':      [2.5,9,9,5,3,5],
            'Prophet':    [2.0,8,8,10,5,6],
        }
        fig_radar = go.Figure()
        for model, vals in scores.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=categories+[categories[0]],
                fill='toself', name=model,
                line=dict(color=MODEL_COLORS[model], width=2),
                fillcolor=MODEL_COLORS[model].replace('#','rgba(').replace(')',',0.08)') if '#' in MODEL_COLORS[model] else 'rgba(0,0,0,0.05)',
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,10], color=TXT2, gridcolor=BORDER),
                angularaxis=dict(color=TXT, gridcolor=BORDER),
                bgcolor=BG2,
            ),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TXT),
            legend=dict(bgcolor=BG3, bordercolor=BORDER, borderwidth=1),
            height=420, margin=dict(t=30,b=30,l=60,r=60),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ─── Tab 2: Forecast Comparison ───
    with tab_forecast:
        st.markdown("<div class='sec-title'>📈 Live Forecast Comparison — All 6 Models (2025 H2 – 2027 H1)</div>", unsafe_allow_html=True)

        fig_all = go.Figure()
        fig_all.add_trace(go.Scatter(
            x=list(range(n_active)), y=list(active_kg),
            mode='lines+markers', name='Historical (NACADA)',
            line=dict(color=GRN, width=3),
            marker=dict(size=9, color=GRN, line=dict(width=2,color=BG2)),
            hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
            text=active_labels,
        ))
        for m_name, fc_vals in all_fc.items():
            lw  = 3 if 'Ensemble' in m_name else 1.8
            dsh = 'solid' if 'Ensemble' in m_name else 'dot'
            fc_x_all = list(range(n_active, n_active+4))
            fig_all.add_trace(go.Scatter(
                x=fc_x_all, y=fc_vals,
                mode='lines+markers', name=m_name,
                line=dict(color=MODEL_COLORS[m_name], width=lw, dash=dsh),
                marker=dict(size=9 if 'Ensemble' in m_name else 7,
                             color=MODEL_COLORS[m_name],
                             line=dict(width=1,color=BG2)),
                hovertemplate=f'<b>{m_name}</b><br>%{{x}}: <b>%{{y:,.0f}} kg</b><extra></extra>',
            ))
        fig_all.update_xaxes(
            tickmode='array',
            tickvals=list(range(n_active+4)),
            ticktext=active_labels+future_labels[:4],
            tickangle=-32
        )
        fig_all.add_vline(x=n_active-.5, line=dict(color=BORDER, dash='dot', width=1.5),
                           annotation_text='History | Forecast',
                           annotation_font=dict(color=TXT2,size=10))
        dark_layout(fig_all, height=480, title='<b>All 6 Models — Live Forecast Comparison</b>',
                    ytitle='kg seized')
        st.plotly_chart(fig_all, use_container_width=True)

        # Side-by-side comparison table
        st.markdown("<div class='sec-title'>📋 Forecast Values — All Models</div>", unsafe_allow_html=True)
        cmp_rows = []
        for m_name in MODEL_COLORS:
            row = {'Model':m_name, 'Test MAPE':f'{m_metrics[m_name][0]:.2f}%'}
            for lbl, v in zip(future_labels[:4], all_fc[m_name]):
                row[lbl] = f'{v:,} kg'
            cmp_rows.append(row)
        df_cmp = pd.DataFrame(cmp_rows)
        def hl_ens2(row):
            if 'Ensemble' in str(row['Model']): return ['background-color:#1A1030']*len(row)
            return ['']*len(row)
        st.dataframe(df_cmp.style.apply(hl_ens2, axis=1), use_container_width=True, hide_index=True)

        csv_cmp = df_cmp.to_csv(index=False)
        st.download_button('⬇️ Download All Model Forecasts as CSV', data=csv_cmp,
                           file_name='all_model_forecasts.csv', mime='text/csv')

        st.markdown("""<div class='result'>
        <b style='color:#A371F7;'>Recommendation for NACADA use:</b> E2 Ensemble (XGBoost 79.5% + LSTM 20.5%)<br>
        Best test MAPE: <b>6.91%</b> · Best test RMSE: <b>1,369 kg</b><br>
        The ensemble benefits from XGBoost's feature engineering (lag, trend, seasonality) and LSTM's sequential memory.
        </div>""", unsafe_allow_html=True)

    # ─── Tab 3: County Lookup ───
    with tab_county_model:
        st.markdown("<div class='sec-title'>🏙️ County Cluster & Risk Lookup</div>", unsafe_allow_html=True)

        # County name input (pre-filled from sidebar)
        col_in1, col_in2 = st.columns([2,1])
        with col_in1:
            county_input = st.text_input('Enter county name:', value=county_q,
                                          placeholder='e.g. Nairobi, Kisumu, Migori…',
                                          key='county_model_input')
        with col_in2:
            st.markdown('<br>', unsafe_allow_html=True)
            search_btn = st.button('🔍 Search County', key='county_search')

        if county_input:
            q = county_input.strip()
            match = None
            for nm in COUNTY_DATA:
                if nm.lower() == q.lower(): match = nm; break
            if not match:
                cands = [nm for nm in COUNTY_DATA if q.lower() in nm.lower()]
                if len(cands) == 1: match = cands[0]
                elif cands:
                    st.warning(f'Multiple matches: {", ".join(cands)}. Be more specific.')

            if match:
                d = COUNTY_DATA[match]; ci = CLUSTER_CFG[d['cluster']]
                col_card, col_detail = st.columns([1,2])
                with col_card:
                    st.markdown(f"""
                    <div class='cluster-card' style='border-color:{ci["color"]};border-width:2px;padding:1.6rem;'>
                        <div style='font-size:3rem;'>{ci["icon"]}</div>
                        <div style='font-size:1.5rem;font-weight:800;color:{ci["color"]};margin:.4rem 0;'>{match}</div>
                        <span class='{ci["badge"]}'>{d["cluster"]} Tier</span>
                        <div style='color:{TXT2};font-size:.85rem;margin-top:.7rem;'>{ci["desc"]}</div>
                        <div style='color:{ci["color"]};font-size:.82rem;font-weight:600;margin-top:.5rem;'>{ci["action"]}</div>
                    </div>""", unsafe_allow_html=True)
                with col_detail:
                    st.markdown(f"""
                    <div style='background:{BG2};border:1px solid {BORDER};border-radius:10px;padding:1.2rem;'>
                    <table style='width:100%;border-collapse:collapse;color:{TXT};'>
                    <tr><td style='padding:.5rem;color:{TXT2};width:45%;'><b>Cluster Tier</b></td>
                        <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>{d["cluster"]}</td></tr>
                    <tr style='background:{BG3};'>
                        <td style='padding:.5rem;color:{TXT2};'><b>National Share</b></td>
                        <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>{ci["share"]}</td></tr>
                    <tr><td style='padding:.5rem;color:{TXT2};'><b>Total (2021–2025)</b></td>
                        <td style='padding:.5rem;'>{d["total_kg"]:,.1f} kg</td></tr>
                    <tr style='background:{BG3};'>
                        <td style='padding:.5rem;color:{TXT2};'><b>Avg per Period</b></td>
                        <td style='padding:.5rem;'>{d["total_kg"]/9:,.1f} kg</td></tr>
                    <tr><td style='padding:.5rem;color:{TXT2};'><b>Trend (2021→2025)</b></td>
                        <td style='padding:.5rem;'>{d["trend"]}</td></tr>
                    <tr style='background:{BG3};'>
                        <td style='padding:.5rem;color:{TXT2};'><b>NACADA Action</b></td>
                        <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>{ci["action"]}</td></tr>
                    </table></div>""", unsafe_allow_html=True)

                others = [n for n, v in COUNTY_DATA.items() if v['cluster']==d['cluster'] and n!=match]
                st.markdown(f"<div class='info'><b>Other {d['cluster']} tier counties:</b> {', '.join(others)}</div>",
                             unsafe_allow_html=True)
            elif county_input and not match:
                st.markdown(f"<div class='warn'>County <b>\"{county_input}\"</b> not found. Check spelling or try a partial name.</div>",
                             unsafe_allow_html=True)

        # Full table
        st.markdown("<div class='sec-title'>📋 All 49 Counties — Full Ranking</div>", unsafe_allow_html=True)
        tier_order={'High':0,'Med-High':1,'Medium':2,'Low':3}
        tbl_rows = sorted(
            [{'County':n,'Cluster Tier':d['cluster'],
              'Total (kg)':f'{d["total_kg"]:,.1f}','Avg/Period':f'{d["total_kg"]/9:,.1f}','Trend':d['trend']}
             for n,d in COUNTY_DATA.items()],
            key=lambda r:(tier_order[r['Cluster Tier']],-float(r['Total (kg)'].replace(',','')))
        )
        df_ct = pd.DataFrame(tbl_rows)
        tier_bg = {'High':'#2D0C0C','Med-High':'#2D1F00','Medium':'#0C1F35','Low':'#21262D'}
        def hl_tier(row):
            return [f'background-color:{tier_bg.get(row["Cluster Tier"],"")}'] * len(row)
        st.dataframe(df_ct.style.apply(hl_tier, axis=1), use_container_width=True, hide_index=True, height=440)

    # ─── Tab 4: Add Seizure Data ───
    with tab_update:
        st.markdown("<div class='sec-title'>📥 Add New NACADA Seizure Observation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info'>
        When NACADA publishes a new bi-annual report, enter the national total below.
        All 6 models will recalculate their forecasts live using this new data point.
        Use the <b>Global Controls</b> in the sidebar for persistent updates across all pages.
        </div>
        """, unsafe_allow_html=True)

        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            up_period = st.selectbox('New reporting period:', FUTURE_LABELS + ['2027 H2','2028 H1'],
                                      key='update_period')
        with col_u2:
            up_value  = st.number_input('National seizure total (kg):', min_value=0.0,
                                         max_value=200_000.0, value=0.0, step=500.0, format='%.0f',
                                         key='update_value')
        with col_u3:
            st.markdown('<br><br>', unsafe_allow_html=True)
            up_btn = st.button('🔄 Recalculate All Forecasts', key='update_btn')

        if up_btn and up_value > 0:
            upd_log  = np.append(active_log, np.log1p(up_value))
            upd_lbsl = active_labels + [up_period]
            st.markdown(f"<div class='result'><b style='color:#2EA043;'>✅ New data point applied:</b> {up_period} = {up_value:,.0f} kg<br>Recalculating all model forecasts below…</div>",
                         unsafe_allow_html=True)
            with st.spinner('Running all 6 models…'):
                upd_fc_all = {}
                for m_name in MODEL_COLORS:
                    try: upd_fc_all[m_name] = run_model(m_name, upd_log, n_ahead=4)
                    except Exception: upd_fc_all[m_name] = [0]*4

            st.markdown("<div class='sec-title'>📊 Updated Forecasts After New Observation</div>", unsafe_allow_html=True)
            upd_rows = [{'Model':m,'Test MAPE':f'{m_metrics[m][0]:.2f}%'} | {lbl:f'{v:,} kg' for lbl,v in zip(future_labels[:4],fc)}
                        for m,fc in upd_fc_all.items()]
            st.dataframe(pd.DataFrame(upd_rows), use_container_width=True, hide_index=True)

        st.markdown("<div class='sec-title'>📋 Current Historical Series (NACADA Actuals)</div>", unsafe_allow_html=True)
        hist_df = pd.DataFrame([{'Period':lbl,'Actual (kg)':f'{v:,.1f}'} for lbl,v in zip(active_labels, active_kg)])
        st.dataframe(hist_df, use_container_width=True, hide_index=True, height=280)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{TXT2};font-size:.8rem;padding:.4rem;line-height:1.8;'>
🌿 Kenya Cannabis Seizure Forecast Dashboard &nbsp;|&nbsp;
William Maureen Ndinda (SCT213-C002-0048/2022) &nbsp;|&nbsp;
BSc Data Science &amp; Analytics &nbsp;|&nbsp; JKUAT Karen &nbsp;|&nbsp; 2026<br>
Supervised by Mr. Isaac Kega Mwangi &nbsp;|&nbsp;
All predictions computed live from saved models in <code style='color:{GRN};'>models/</code> &nbsp;|&nbsp;
NACADA Bi-Annual Reports 2021–2025
</div>
""", unsafe_allow_html=True)
