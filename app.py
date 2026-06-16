# app.py  —  Kenya Cannabis Seizure Forecast Tool
# William Maureen Ndinda | SCT213-C002-0048/2022 | JKUAT Karen
# ============================================================
# This Streamlit app provides an interactive interface for forecasting cannabis 
# seizures in Kenya using multiple models, including a custom ensemble. 
# It is designed for NACADA to visualize historical trends, input new data, 
# and see updated forecasts along with county-level risk zones.
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
#
# WHAT THIS APP DOES:
#   1. Shows the full historical seizure series (2021–2025)
#   2. Lets user input the latest NACADA observed value
#   3. Forecasts the next 4 periods using XGBoost, LSTM, or E2 Ensemble
#   4. Displays forecast values and confidence range
#   5. Shows county High-tier risk zone
#   6. User can switch between all five models + ensemble
# All predictions are made LIVE from the saved trained models.
# ============================================================

import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import pickle
try:
    import joblib
except Exception:
    joblib = None
import warnings
warnings.filterwarnings('ignore')

@@ -42,7 +38,7 @@
)

# ─────────────────────────────────────────────────────────────
# CSS — GREEN PRIMARY THEME
# GREEN THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@@ -52,36 +48,33 @@
    background: linear-gradient(180deg, #1A5C2A 0%, #2D8B47 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label { color: #C8F0D0 !important; }
section[data-testid="stSidebar"] label { color: #C8F0D0 !important; }

.header {
    background: linear-gradient(135deg, #1A5C2A 0%, #2D8B47 60%, #52A96A 100%);
    padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 18px rgba(26,92,42,0.25);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(26,92,42,0.25);
}
.header h1 { color: white !important; margin: 0; font-size: 1.85rem; font-weight: 800; }
.header p  { color: #C8F0D0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.header h1 { color: white !important; margin:0; font-size:1.85rem; }
.header p  { color: #C8F0D0; margin:0.3rem 0 0; font-size:0.95rem; }

.kpi {
    background: white; border: 2px solid #2D8B47;
    border-radius: 12px; padding: 1.1rem 0.8rem;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.kpi-val  { color: #1A5C2A; font-size: 2rem; font-weight: 800; margin: 0; }
.kpi-lbl  { color: #2D8B47; font-size: 0.82rem; font-weight: 600; margin: 0.2rem 0 0; }
.kpi-sub  { color: #888; font-size: 0.76rem; margin: 0; }
.kpi-val { color: #1A5C2A; font-size:2rem; font-weight:800; margin:0; }
.kpi-lbl { color: #2D8B47; font-size:0.82rem; font-weight:600; margin:0.2rem 0 0; }
.kpi-sub { color: #888; font-size:0.76rem; margin:0; }

.info  { background:#E8F5E9; border-left:4px solid #2D8B47; color:#000;
.info  { background:#E8F5E9; border-left:4px solid #2D8B47;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.warn  { background:#FFF8E1; border-left:4px solid #EF9F27;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.danger{ background:#FDECEA; border-left:4px solid #C00000;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.result{ background:#E8F5E9; border:2px solid #2D8B47; border-radius:12px;
         padding:1.2rem; margin:.8rem 0; }
.result{ background:#E8F5E9; border:2px solid #2D8B47;
         border-radius:12px; padding:1.2rem; margin:.8rem 0; }

.sec-title {
    color:#1A5C2A; font-size:1.1rem; font-weight:700;
@@ -91,54 +84,39 @@
div.stButton > button {
    background:#2D8B47 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:.55rem 1.8rem !important; width:100%;
    font-weight:700 !important; padding:.55rem 1.8rem !important; width:100%;
}
div.stButton > button:hover { background:#1A5C2A !important; }

.badge-h  { background:#FDECEA; color:#C00000;  border:2px solid #C00000;
.badge-h  { background:#FDECEA; color:#C00000; border:2px solid #C00000;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-mh { background:#FFF3CD; color:#7F4F00;  border:2px solid #EF9F27;
.badge-mh { background:#FFF3CD; color:#7F4F00; border:2px solid #EF9F27;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-m  { background:#D6E4F7; color:#1F3864;  border:2px solid #2E5FA3;
.badge-m  { background:#D6E4F7; color:#1F3864; border:2px solid #2E5FA3;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-l  { background:#F2F2F2; color:#444;     border:2px solid #888;
.badge-l  { background:#F2F2F2; color:#444; border:2px solid #888;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# COLOURS
# ─────────────────────────────────────────────────────────────
GRN  = '#1A5C2A'; GRN2 = '#2D8B47'; GRN3 = '#52A96A'; GPALE = '#E8F5E9'
AMB  = '#EF9F27'; BLU  = '#2E5FA3'; RED  = '#C00000'; GRAY  = '#888780'

HIST_LABELS = ['2021 H1','2021 H2','2022 H1','2022 H2','2023 H1',
               '2023 H2','2024 H1','2024 H2','2025 H1']
HIST_KG     = np.array([6932.09, 4781.42, 3621.41, 7249.73, 11866.28,
                         13714.12, 13415.20, 14737.70, 12752.52])
HIST_LOG    = np.log1p(HIST_KG)

NEXT_PERIODS = ['2025 H2','2026 H1','2026 H2','2027 H1']
N_HIST       = len(HIST_LABELS)
GRN  = '#1A5C2A'; GRN2 = '#2D8B47'; GRN3 = '#52A96A'
AMB  = '#EF9F27'; BLU  = '#2E5FA3'; RED  = '#C00000'; GRAY = '#888780'

MODEL_COLORS = {
    'E2 Ensemble (Recommended)': '#B5268B',
    'XGBoost'                  : AMB,
    'LSTM'                     : BLU,
    'SARIMA'                   : '#D85A30',
    'ARIMA'                    : '#1D6FA4',
    'Prophet'                  : '#7F77DD',
}
MODEL_METRICS = {
    'ARIMA'                   :{'rmse':11727.5,'mape':85.87, 'cv':9777.6, 'rank':6},
    'SARIMA'                  :{'rmse':10525.1,'mape':67.67, 'cv':3459.8, 'rank':5},
    'Prophet'                 :{'rmse':18336.9,'mape':116.63,'cv':5092.6, 'rank':4},  # noqa: E501 wait actually 4th best on test
    'LSTM'                    :{'rmse': 3393.4,'mape':22.94, 'cv':5928.7, 'rank':3},
    'XGBoost'                 :{'rmse': 1046.0,'mape': 7.08, 'cv':1532.1, 'rank':2},
    'E2 Ensemble (Recommended)':{'rmse': 1369.2,'mape': 6.91, 'cv':1369.2, 'rank':1},
    'E2 Ensemble': '#B5268B',
    'XGBoost'    : '#EF9F27',
    'LSTM'       : '#1D9E75',
    'SARIMA'     : '#D85A30',
    'ARIMA'      : '#1D6FA4',
    'Prophet'    : '#7F77DD',
}

# ─────────────────────────────────────────────────────────────
# COUNTY DATA
# ─────────────────────────────────────────────────────────────
COUNTY_DATA = {
    'Nairobi':{'cluster':'High','total_kg':12231.1,'trend':'📈'},
    'Migori':{'cluster':'High','total_kg':10071.0,'trend':'📈'},
@@ -189,368 +167,308 @@
    'Railways (CPIU)':{'cluster':'Low','total_kg':5.3,'trend':'📉'},
    'West Pokot':{'cluster':'Low','total_kg':2.8,'trend':'📉'},
}

CLUSTER_CFG = {
    'High'    :{'icon':'🔴','badge':'badge-h', 'color':RED, 'bg':'#FDECEA',
    'High'    :{'icon':'🔴','badge':'badge-h','color':RED,'bg':'#FDECEA',
                'share':'63.5%','action':'Immediate targeted enforcement surge required.',
                'desc':'Critical — highest enforcement priority.'},
    'Med-High':{'icon':'🟠','badge':'badge-mh','color':'#7F4F00','bg':'#FFF3CD',
                'share':'21.7%','action':'Build capacity before escalation to High tier.',
                'desc':'Emerging threat — pre-emptive intervention needed.'},
    'Medium'  :{'icon':'🔵','badge':'badge-m', 'color':BLU, 'bg':'#D6E4F7',
    'Medium'  :{'icon':'🔵','badge':'badge-m','color':BLU,'bg':'#D6E4F7',
                'share':'13.8%','action':'Maintain standard enforcement; monitor trends.',
                'desc':'Moderate activity — regular monitoring required.'},
    'Low'     :{'icon':'⚪','badge':'badge-l', 'color':'#444','bg':'#F2F2F2',
    'Low'     :{'icon':'⚪','badge':'badge-l','color':'#444','bg':'#F2F2F2',
                'share':'1.0%','action':'Review enforcement capacity.',
                'desc':'Low recorded seizures — may reflect capacity gaps.'},
}

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING FUNCTIONS
# LOAD ALL MODELS FROM DISK
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / 'artifacts'
MODEL_BUNDLE_CANDIDATES = [
    MODEL_DIR / 'model_bundle.pkl',
    MODEL_DIR / 'trained_models.pkl',
    MODEL_DIR / 'model_bundle.pickle',
]
MODEL_FILES_DIR = Path(__file__).resolve().parent / 'models'
EXTRA_MODEL_FILE_CANDIDATES = {
    'ARIMA': [MODEL_FILES_DIR / 'arima_model.kpl', MODEL_FILES_DIR / 'arima_model.pkl'],
    'SARIMA': [MODEL_FILES_DIR / 'sarima_model.kpl', MODEL_FILES_DIR / 'sarima_model.pkl'],
    'Prophet': [MODEL_FILES_DIR / 'prophet_model.kpl', MODEL_FILES_DIR / 'prophet_model.pkl'],
}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_all_models():
    """Load all saved models from disk. Cached for the session."""
    def _load(fname):
        path = os.path.join(MODELS_DIR, fname)
        with open(path, 'rb') as f:
            return pickle.load(f)

    series     = _load('series_data.pkl')
    xgb_data   = _load('xgboost_model.pkl')
    lstm_w     = _load('lstm_weights.pkl')
    lstm_sc    = _load('lstm_scaler.pkl')
    arima      = _load('arima_params.pkl')
    sarima     = _load('sarima_params.pkl')
    prophet    = _load('prophet_params.pkl')
    ens        = _load('ensemble_weights.pkl')

    return {
        'series'  : series,
        'xgb'     : xgb_data,
        'lstm_w'  : lstm_w,
        'lstm_sc' : lstm_sc,
        'arima'   : arima,
        'sarima'  : sarima,
        'prophet' : prophet,
        'ensemble': ens,
    }

try:
    MODELS = load_all_models()
    SERIES = MODELS['series']
except Exception as e:
    st.error(f"❌ Could not load models from `models/` folder.\n\nError: {e}\n\n"
             "Make sure the `models/` folder is in the same directory as `app.py`.")
    st.stop()

HIST_LABELS = SERIES['hist_labels']
HIST_KG     = np.array(SERIES['hist_kg'])
HIST_LOG    = np.array(SERIES['hist_log'])
TRAIN_END   = SERIES['train_end']
FUTURE_LABELS = SERIES['future_labels']
N_HIST      = len(HIST_LABELS)


# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS — ALL LIVE FROM SAVED MODELS
# ─────────────────────────────────────────────────────────────

def build_row(log_series, t_idx):
def _sig(x):  return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x): return np.tanh(np.clip(x, -15, 15))


def xgb_build_row(log_series, t_idx):
    """Build one XGBoost feature row from log-scale series."""
    s = np.array(log_series)
    lag1=s[-1]; lag2=s[-2]; lag3=s[-3]
    rm2=s[-2:].mean(); rm3=s[-3:].mean()
    ldiff=lag1-lag2; tup=1 if ldiff>0 else 0
    abvm=1 if lag1>s.mean() else 0
    year=2021+(t_idx//2); half=(t_idx%2)+1
    return np.array([[t_idx,year,half,
                      np.sin(2*np.pi*half/2),np.cos(2*np.pi*half/2),
                      lag1,lag2,lag3,rm2,rm3,ldiff,tup,abvm]])


def _load_serialized_models():
    for bundle_path in MODEL_BUNDLE_CANDIDATES:
        if not bundle_path.exists():
            continue

        bundle = None
        if joblib is not None:
            try:
                bundle = joblib.load(bundle_path)
            except Exception:
                bundle = None

        if bundle is None:
            with bundle_path.open('rb') as fh:
                bundle = pickle.load(fh)

        if isinstance(bundle, (tuple, list)) and len(bundle) >= 4:
            gbr, lstm_params, sc_min, sc_rng = bundle[:4]
            return gbr, lstm_params, sc_min, sc_rng, bundle_path.name

        if isinstance(bundle, dict):
            gbr = bundle.get('gbr_model') or bundle.get('xgb_model') or bundle.get('model')
            lstm_params = bundle.get('lstm_params') or bundle.get('lstm_model')
            sc_min = bundle.get('sc_min')
            sc_rng = bundle.get('sc_rng')

            if gbr is not None and lstm_params is not None and sc_min is not None and sc_rng is not None:
                return gbr, lstm_params, sc_min, sc_rng, bundle_path.name

    return None


def _load_colab_models():
    xgb_path = MODEL_FILES_DIR / 'xgboost_model.pkl'
    lstm_weights_path = MODEL_FILES_DIR / 'lstm_weights.pkl'
    lstm_scaler_path = MODEL_FILES_DIR / 'lstm_scaler.pkl'
    ensemble_weights_path = MODEL_FILES_DIR / 'ensemble_weights.pkl'
    metadata_path = MODEL_FILES_DIR / 'model_metadata.pkl'

    if not (xgb_path.exists() and lstm_weights_path.exists()):
        return None

    def load_any(path):
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                pass
        with path.open('rb') as fh:
            return pickle.load(fh)

    gbr = load_any(xgb_path)
    lstm_params = load_any(lstm_weights_path)
    sc_min, sc_rng = HIST_LOG[:5].min(), (HIST_LOG[:5].max() - HIST_LOG[:5].min() + 1e-9)

    scaler_bundle = None
    if lstm_scaler_path.exists():
        scaler_bundle = load_any(lstm_scaler_path)
        if isinstance(scaler_bundle, dict):
            sc_min = scaler_bundle.get('sc_min', sc_min)
            sc_rng = scaler_bundle.get('sc_rng', sc_rng)
        elif isinstance(scaler_bundle, (tuple, list)) and len(scaler_bundle) >= 2:
            sc_min, sc_rng = scaler_bundle[:2]

    ensemble_weights = {'xgb': 0.795, 'lstm': 0.205}
    if ensemble_weights_path.exists():
        ensemble_bundle = load_any(ensemble_weights_path)
        if isinstance(ensemble_bundle, dict):
            ensemble_weights['xgb'] = float(ensemble_bundle.get('xgb', ensemble_weights['xgb']))
            ensemble_weights['lstm'] = float(ensemble_bundle.get('lstm', ensemble_weights['lstm']))
        elif isinstance(ensemble_bundle, (tuple, list)) and len(ensemble_bundle) >= 2:
            ensemble_weights['xgb'] = float(ensemble_bundle[0])
            ensemble_weights['lstm'] = float(ensemble_bundle[1])

    metadata = None
    if metadata_path.exists():
        metadata = load_any(metadata_path)

    extra_models = {}
    for model_name, candidate_paths in EXTRA_MODEL_FILE_CANDIDATES.items():
        for candidate_path in candidate_paths:
            if not candidate_path.exists():
                continue
            try:
                extra_models[model_name] = load_any(candidate_path)
                break
            except Exception:
                continue

    return gbr, lstm_params, sc_min, sc_rng, ensemble_weights, metadata, 'models/', extra_models


@st.cache_resource
def get_models():
    """Load saved Colab-trained models when available, otherwise train fallback models."""
    loaded = _load_colab_models()
    if loaded is not None:
        return loaded

    loaded = _load_serialized_models()
    if loaded is not None:
        gbr, lstm_params, sc_min, sc_rng, bundle_name = loaded
        return gbr, lstm_params, sc_min, sc_rng, {'xgb': 0.795, 'lstm': 0.205}, None, bundle_name, {}

    log = HIST_LOG.copy()

    # ── XGBoost ────────────────────────────────────────────────
    X = np.array([build_row(log[:i], i)[0] for i in range(3, len(log))])
    y = log[3:]
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=1,
                                    learning_rate=0.1, random_state=42)
    gbr.fit(X, y)

    # ── LSTM from scratch ───────────────────────────────────────
    H = 8; LR = 0.01; LB = 2
    sc_min = log[:5].min(); sc_max = log[:5].max()
    sc_rng = sc_max - sc_min + 1e-9
    y_all_sc = (log - sc_min) / sc_rng
    y_tr_sc  = y_all_sc[:5]
    y_val_sc = y_all_sc[5:7]

    np.random.seed(42); s2 = np.sqrt(1/(H+1))
    Wih=np.random.randn(4*H,1)*s2; Whh=np.random.randn(4*H,H)*s2
    bh=np.zeros(4*H); bh[H:2*H]=1.0
    Wy=np.random.randn(1,H)*s2; by=np.zeros(1)
    ma={k:np.zeros_like(v) for k,v in[('Wi',Wih),('Wh',Whh),('bh',bh),('Wy',Wy),('by',by)]}
    va={k:np.zeros_like(v) for k,v in[('Wi',Wih),('Wh',Whh),('bh',bh),('Wy',Wy),('by',by)]}
    ta=[0]

    def sg(x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    def th(x): return np.tanh(np.clip(x,-15,15))

    def fwd(seq,cache=False):
        h=np.zeros(H); c=np.zeros(H); ch=[]
        for v in seq:
            x=np.array([float(v)])
            g=Wih@x+Whh@h+bh
            ig=sg(g[:H]); fg=sg(g[H:2*H])
            gg=th(g[2*H:3*H]); og=sg(g[3*H:])
            cn=fg*c+ig*gg; hn=og*th(cn)
            if cache: ch.append((ig,fg,gg,og,cn,x,h.copy(),c.copy()))
            h,c=hn,cn
        yh=float((Wy@h+by)[0])
        return (yh,h,ch) if cache else yh

    def step(seq,tgt):
        nonlocal Wih,Whh,bh,Wy,by
        yh,hl,ch=fwd(seq,True); ta[0]+=1
        dL=2*(yh-tgt)
        dWy=dL*hl[None,:]; dby=np.array([dL])
        dh=dL*Wy[0]; dc=np.zeros(H)
        dWih=np.zeros_like(Wih); dWhh=np.zeros_like(Whh); dbh_=np.zeros_like(bh)
        for ig,fg,gg,og,cn,x,hp,cp in reversed(ch):
            tc=th(cn); dct=dh*og*(1-tc**2)+dc
            do=dh*tc*og*(1-og); di=dct*gg*ig*(1-ig)
            df=dct*cp*fg*(1-fg); dg_=dct*ig*(1-gg**2)
            dg=np.concatenate([di,df,dg_,do])
            dWih+=np.outer(dg,x); dWhh+=np.outer(dg,hp); dbh_+=dg
            dh=Whh.T@dg; dc=dct*fg
        for nm,p,g in[('Wi',Wih,dWih),('Wh',Whh,dWhh),('bh',bh,dbh_),
                       ('Wy',Wy,dWy),('by',by,dby)]:
            g=np.clip(g,-5,5)
            ma[nm]=.9*ma[nm]+(1-.9)*g; va[nm]=.999*va[nm]+(1-.999)*g**2
            mh=ma[nm]/(1-.9**ta[0]); vh=va[nm]/(1-.999**ta[0])
            p[:] -= LR*mh/(np.sqrt(vh)+1e-8)

    Xtr=[y_tr_sc[i:i+LB] for i in range(len(y_tr_sc)-LB)]
    ytr=[y_tr_sc[i+LB]   for i in range(len(y_tr_sc)-LB)]
    vseed=np.concatenate([y_tr_sc[-LB:],y_val_sc])
    Xva=[vseed[i:i+LB]   for i in range(len(vseed)-LB)]
    yva=[vseed[i+LB]      for i in range(len(vseed)-LB)]

    bv=1e9; wait=0; bs=None
    for ep in range(2000):
        for i in np.random.permutation(len(Xtr)): step(Xtr[i],ytr[i])
        vl=np.mean([(fwd(x)-y)**2 for x,y in zip(Xva,yva)])
        if vl<bv-1e-7: bv=vl; wait=0; bs=(Wih.copy(),Whh.copy(),bh.copy(),Wy.copy(),by.copy())
        else:
            wait+=1
            if wait>=60: break

    Wih,Whh,bh,Wy,by=bs
    return gbr, (Wih,Whh,bh,Wy,by), sc_min, sc_rng, {'xgb': 0.795, 'lstm': 0.205}, None, 'fallback-training', {}


def run_forecast(log_series, model_name, gbr, lstm_params, sc_min, sc_rng, ensemble_weights=None, extra_models=None, n=4):
    """Return list of n forecast values in kg for the chosen model."""
    Wih,Whh,bh,Wy,by = lstm_params
    H = Wih.shape[0]//4
    ensemble_weights = ensemble_weights or {'xgb': 0.795, 'lstm': 0.205}
    extra_models = extra_models or {}

    def sg(x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    def th(x): return np.tanh(np.clip(x,-15,15))

    def lstm_step(seq):
    return np.array([[t_idx, year, half,
                      np.sin(2*np.pi*half/2), np.cos(2*np.pi*half/2),
                      lag1, lag2, lag3, rm2, rm3, ldiff, tup, abvm]])


def predict_xgboost(log_series, n_ahead=4):
    """Recursive XGBoost forecast using the saved GBR model."""
    gbr = MODELS['xgb']['model']
    cur = list(log_series)
    preds = []
    for i in range(n_ahead):
        t   = len(HIST_LOG) + i if len(log_series) == N_HIST else len(log_series) + i
        row = xgb_build_row(np.array(cur), t)
        p   = float(gbr.predict(row)[0])
        preds.append(int(np.expm1(p)))
        cur.append(p)
    return preds


def predict_lstm(log_series, n_ahead=4):
    """Recursive LSTM forecast using saved weight matrices."""
    Wih = MODELS['lstm_w']['Wih']; Whh = MODELS['lstm_w']['Whh']
    bh  = MODELS['lstm_w']['bh'];  Wy  = MODELS['lstm_w']['Wy']
    by  = MODELS['lstm_w']['by']
    H   = MODELS['lstm_sc']['H']
    sc_min = MODELS['lstm_sc']['sc_min']
    sc_rng = MODELS['lstm_sc']['sc_rng']

    def fwd(seq):
        h=np.zeros(H); c=np.zeros(H)
        for v in seq:
            x=np.array([float(v)])
            g=Wih@x+Whh@h+bh
            ig=sg(g[:H]); fg=sg(g[H:2*H])
            gg=th(g[2*H:3*H]); og=sg(g[3*H:])
            cn=fg*c+ig*gg; h=og*th(cn); c=cn
            ig=_sig(g[:H]); fg=_sig(g[H:2*H])
            gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
            cn=fg*c+ig*gg; h=og*_tanh(cn); c=cn
        return float((Wy@h+by)[0])

    cur_log = list(log_series)
    results = []

    for i in range(n):
        t = N_HIST + i
        cur = np.array(cur_log)

        if model_name == 'XGBoost':
            row  = build_row(cur, t)
            p_log = float(gbr.predict(row)[0])

        elif model_name == 'LSTM':
            sc   = (cur - sc_min) / sc_rng
            p_sc = lstm_step(sc[-2:])
            p_log= p_sc * sc_rng + sc_min

        elif model_name == 'E2 Ensemble (Recommended)':
            xgb_log  = float(gbr.predict(build_row(cur, t))[0])
            sc       = (cur - sc_min) / sc_rng
            lstm_log = lstm_step(sc[-2:]) * sc_rng + sc_min
            p_log    = ensemble_weights.get('xgb', 0.795) * xgb_log + ensemble_weights.get('lstm', 0.205) * lstm_log

        elif model_name == 'SARIMA':
            model_obj = extra_models.get('SARIMA')
            if model_obj is not None:
                if hasattr(model_obj, 'forecast'):
                    pred = model_obj.forecast(steps=1)
                    p_log = float(np.asarray(pred).reshape(-1)[0])
                elif hasattr(model_obj, 'predict'):
                    pred = model_obj.predict(1)
                    p_log = float(np.asarray(pred).reshape(-1)[0])
                else:
                    p_log = cur[-1]
            else:
                p_log = cur[-1]

        elif model_name == 'ARIMA':
            model_obj = extra_models.get('ARIMA')
            if model_obj is not None:
                if hasattr(model_obj, 'forecast'):
                    pred = model_obj.forecast(steps=1)
                    p_log = float(np.asarray(pred).reshape(-1)[0])
                elif hasattr(model_obj, 'predict'):
                    pred = model_obj.predict(1)
                    p_log = float(np.asarray(pred).reshape(-1)[0])
                else:
                    p_log = cur[-1]
            else:
                p_log = cur[-1]

        elif model_name == 'Prophet':
            model_obj = extra_models.get('Prophet')
            if model_obj is not None and hasattr(model_obj, 'predict'):
                try:
                    if hasattr(model_obj, 'make_future_dataframe'):
                        future_df = model_obj.make_future_dataframe(periods=1, freq='6M')
                        pred = model_obj.predict(future_df)
                        p_log = float(pred['yhat'].iloc[-1])
                    else:
                        pred = model_obj.predict(pd.DataFrame({'ds': [pd.Timestamp.today()]}))
                        if isinstance(pred, pd.DataFrame) and 'yhat' in pred:
                            p_log = float(pred['yhat'].iloc[0])
                        else:
                            p_log = float(np.asarray(pred).reshape(-1)[0])
                except Exception:
                    p_log = cur[-1]
            else:
                p_log = cur[-1]

        else:
            p_log = cur[-1]

        p_kg = float(np.expm1(p_log))
        results.append(max(0, round(p_kg)))
        cur_log.append(p_log)

    return results
    preds   = []
    for _ in range(n_ahead):
        cur_sc   = (np.array(cur_log) - sc_min) / sc_rng
        pred_sc  = fwd(cur_sc[-2:])
        pred_log = pred_sc * sc_rng + sc_min
        preds.append(int(np.expm1(pred_log)))
        cur_log.append(pred_log)
    return preds


def predict_arima(log_series, n_ahead=4):
    """Recursive ARIMA(1,1,2) forecast using saved parameters."""
    d       = MODELS['arima']
    params  = d['params']
    ar_lags = d['ar_lags']
    ma_lags = d['ma_lags']
    const   = params[0]
    phi     = params[1:1+len(ar_lags)]
    theta   = params[1+len(ar_lags):]

    dy  = np.diff(log_series, 1).tolist()
    start = max(ar_lags + ma_lags)

    # Compute residuals on known data
    eps = [0.0] * len(dy)
    for t in range(start, len(dy)):
        pred = const
        for j, lag in enumerate(ar_lags):
            if t-lag >= 0: pred += phi[j]*dy[t-lag]
        for j, lag in enumerate(ma_lags):
            if t-lag >= 0: pred += theta[j]*eps[t-lag]
        eps[t] = dy[t] - pred

    cur_log = list(log_series)
    preds   = []
    for _ in range(n_ahead):
        t    = len(dy)
        pred = const
        for j, lag in enumerate(ar_lags):
            idx = t - lag
            if idx >= 0: pred += phi[j]*dy[idx]
        for j, lag in enumerate(ma_lags):
            idx = t - lag
            if idx >= 0: pred += theta[j]*eps[idx]
        dy.append(pred)
        eps.append(0.0)
        next_log = cur_log[-1] + pred
        cur_log.append(next_log)
        preds.append(int(np.expm1(next_log)))
    return preds


def predict_sarima(log_series, n_ahead=4):
    """Recursive SARIMA(1,1,2)(1,0,1)_2 forecast using saved parameters."""
    d       = MODELS['sarima']
    params  = d['params']
    ar_lags = d['ar_lags']
    ma_lags = d['ma_lags']
    const   = params[0]
    phi     = params[1:1+len(ar_lags)]
    theta   = params[1+len(ar_lags):]

    dy  = np.diff(log_series, 1).tolist()
    start = max(ar_lags + ma_lags)

    eps = [0.0] * len(dy)
    for t in range(start, len(dy)):
        pred = const
        for j, lag in enumerate(ar_lags):
            if t-lag >= 0: pred += phi[j]*dy[t-lag]
        for j, lag in enumerate(ma_lags):
            if t-lag >= 0: pred += theta[j]*eps[t-lag]
        eps[t] = dy[t] - pred

def build_model_comparison_rows(base_log, gbr, lstm_params, sc_min, sc_rng, ensemble_weights, extra_models):
    model_names = [
        'E2 Ensemble (Recommended)',
        'XGBoost',
        'LSTM',
        'SARIMA',
        'ARIMA',
        'Prophet',
    ]
    cur_log = list(log_series)
    preds   = []
    for _ in range(n_ahead):
        t    = len(dy)
        pred = const
        for j, lag in enumerate(ar_lags):
            idx = t - lag
            if idx >= 0: pred += phi[j]*dy[idx]
        for j, lag in enumerate(ma_lags):
            idx = t - lag
            if idx >= 0: pred += theta[j]*eps[idx]
        dy.append(pred)
        eps.append(0.0)
        next_log = cur_log[-1] + pred
        cur_log.append(next_log)
        preds.append(int(np.expm1(next_log)))
    return preds


def predict_prophet(log_series, n_ahead=4):
    """Prophet forecast using saved k, m, delta, fourier parameters."""
    p = MODELS['prophet']
    # Use full-series params (k_full, m_full etc.) for forecasting
    k       = p['k_full']
    m       = p['m_full']
    delta   = np.array(p['delta_full'])
    fourier = np.array(p['fourier_full'])
    cp      = p['cp_full']
    period  = p['period']
    order   = p['order']
    n_known = len(log_series)

    def prophet_val(t):
        trend    = k * t + m
        a        = 1.0 if t >= cp else 0.0
        trend   += delta[0] * (t - cp) * a
        seasonal = 0.0
        for n in range(1, order+1):
            an = fourier[2*(n-1)]; bn = fourier[2*(n-1)+1]
            seasonal += (an*np.cos(2*np.pi*n*t/period) +
                         bn*np.sin(2*np.pi*n*t/period))
        return trend + seasonal

    preds = []
    for i in range(n_ahead):
        t    = n_known + i
        yhat = prophet_val(t)
        preds.append(int(np.expm1(yhat)))
    return preds


def predict_ensemble(log_series, n_ahead=4):
    """
    E2 Ensemble: XGBoost (79.5%) + LSTM (20.5%) on log scale.
    Weights from saved ensemble_weights.pkl.
    """
    w_xgb  = MODELS['ensemble']['xgb_weight']
    w_lstm = MODELS['ensemble']['lstm_weight']

    # Need log-scale forecasts from each component
    gbr     = MODELS['xgb']['model']
    Wih     = MODELS['lstm_w']['Wih']; Whh = MODELS['lstm_w']['Whh']
    bh      = MODELS['lstm_w']['bh'];  Wy  = MODELS['lstm_w']['Wy']
    by      = MODELS['lstm_w']['by']
    H       = MODELS['lstm_sc']['H']
    sc_min  = MODELS['lstm_sc']['sc_min']
    sc_rng  = MODELS['lstm_sc']['sc_rng']

    def lstm_fwd(seq):
        h=np.zeros(H); c=np.zeros(H)
        for v in seq:
            x=np.array([float(v)])
            g=Wih@x+Whh@h+bh
            ig=_sig(g[:H]); fg=_sig(g[H:2*H])
            gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
            cn=fg*c+ig*gg; h=og*_tanh(cn); c=cn
        return float((Wy@h+by)[0])

    rows = []
    for model_name in model_names:
        future_vals = run_forecast(
            base_log,
            model_name,
            gbr,
            lstm_params,
            sc_min,
            sc_rng,
            ensemble_weights,
            extra_models,
            n=4,
        )
        row = {
            'Model': model_name,
            'Type': '⭐ Ensemble' if 'Ensemble' in model_name else ('Individual' if model_name in {'XGBoost', 'LSTM'} else 'Statistical'),
            'Test MAPE': f"{MODEL_METRICS.get(model_name, {}).get('mape', '—')}%",
        }
        for lbl, val in zip(future_labels_used, future_vals):
            row[lbl] = f'{val:,}'
        rows.append(row)

    return rows
    cur_log = list(log_series)
    preds   = []
    for i in range(n_ahead):
        t      = len(HIST_LOG) + i if len(log_series)==N_HIST else len(log_series)+i
        # XGBoost log prediction
        row       = xgb_build_row(np.array(cur_log), t)
        xgb_log   = float(gbr.predict(row)[0])
        # LSTM log prediction
        cur_sc    = (np.array(cur_log) - sc_min) / sc_rng
        lstm_sc_p = lstm_fwd(cur_sc[-2:])
        lstm_log  = lstm_sc_p * sc_rng + sc_min
        # Ensemble on log scale then invert
        e2_log    = w_xgb * xgb_log + w_lstm * lstm_log
        preds.append(int(np.expm1(e2_log)))
        cur_log.append(e2_log)   # extend with E2 for next recursion
    return preds


def run_model(model_name, log_series, n_ahead=4):
    """Dispatch to the right model prediction function."""
    fn = {
        'E2 Ensemble': predict_ensemble,
        'XGBoost'    : predict_xgboost,
        'LSTM'       : predict_lstm,
        'SARIMA'     : predict_sarima,
        'ARIMA'      : predict_arima,
        'Prophet'    : predict_prophet,
    }.get(model_name)
    if fn is None:
        return [0]*n_ahead
    return fn(log_series, n_ahead)


# ─────────────────────────────────────────────────────────────
@@ -559,151 +477,153 @@ def build_model_comparison_rows(base_log, gbr, lstm_params, sc_min, sc_rng, ense
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem;'>
        <div style='font-size:3rem'>🌿</div>
        <div style='font-size:1.05rem;font-weight:800;color:white;'>
            NACADA Forecast Tool</div>
        <div style='font-size:.78rem;color:#C8F0D0;'>
            Cannabis Seizure Prediction</div>
        <div style='font-size:3rem;'>🌿</div>
        <div style='font-size:1.05rem;font-weight:800;color:white;'>NACADA Forecast</div>
        <div style='font-size:.78rem;color:#C8F0D0;'>Cannabis Seizure Prediction</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.3);'>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Choose Model")
    st.markdown("### 🤖 Model")
    selected_model = st.selectbox(
        'Select forecasting model:',
        options=list(MODEL_COLORS.keys()),
        index=0,
        help='E2 Ensemble is recommended — best MAPE 6.91%'
        help='E2 Ensemble recommended — best MAPE 6.91%'
    )

    st.markdown("### 📥 Enter New NACADA Value")
    st.markdown("<small style='color:#C8F0D0;'>When NACADA publishes a new bi-annual report, enter the latest seizure figure here to update the forecast.</small>",
    st.markdown("### 📥 Add New NACADA Value")
    st.markdown("<small style='color:#C8F0D0;'>When NACADA publishes a new bi-annual report, enter the value here to update the forecast with real new data.</small>",
                unsafe_allow_html=True)

    new_period_lbl = st.selectbox(
        'New observed period:',
        ['2025 H2', '2026 H1', '2026 H2', '2027 H1']
    )
    new_value_kg = st.number_input(
        f'Actual seizures for {new_period_lbl} (kg):',
        min_value=0.0, max_value=200000.0,
        value=0.0, step=500.0,
        format='%.0f'
    new_period = st.selectbox('New period:', FUTURE_LABELS + ['2027 H2','2028 H1'])
    new_value  = st.number_input(
        f'Actual seizures (kg):', min_value=0.0, max_value=200000.0,
        value=0.0, step=500.0, format='%.0f'
    )
    run_update = st.button('🔄 Update Forecast', type='primary')
    run_btn = st.button('🔄 Update Forecast', type='primary')

    st.markdown("<hr style='border-color:rgba(255,255,255,.3);'>", unsafe_allow_html=True)
    st.markdown("### 🔍 County Cluster Lookup")
    county_query = st.text_input(
        'Type county name:',
        placeholder='e.g. Nairobi, Kisumu...'
    )
    st.markdown("### 🔍 County Lookup")
    county_q = st.text_input('Type county name:', placeholder='e.g. Nairobi, Kisumu…')

    st.markdown("<hr style='border-color:rgba(255,255,255,.3);'>", unsafe_allow_html=True)
    st.markdown("""
    st.markdown(f"""
    <div style='font-size:.8rem;color:#C8F0D0;'>
    <b>Student:</b> William Maureen Ndinda<br>
    <b>Reg:</b> SCT213-C002-0048/2022<br>
    <b>Programme:</b> BSc Data Science<br>
    <b>JKUAT Karen | 2026</b><br><br>
    <b>Data:</b> NACADA 2021–2025<br>
    <b>Models:</b> XGBoost · LSTM · E2 Ensemble
    <b>Models loaded from:</b><br>
    <code style='color:#A8DBB8;font-size:.75rem;'>models/</code> folder ✓
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner('🌿 Loading trained models…'):
    gbr_model, lstm_params, sc_min, sc_rng, ensemble_weights, model_metadata, model_source, extra_models = get_models()

if model_source == 'fallback-training':
    st.warning('Saved Colab model artifacts were not found, so the app is using the built-in fallback training logic.')
else:
    st.success(f'Loaded trained model artifacts from {model_source}.')

# ─────────────────────────────────────────────────────────────
# DETERMINE ACTIVE SERIES (with or without new value)
# BUILD ACTIVE SERIES (with optional new value)
# ─────────────────────────────────────────────────────────────
active_log    = HIST_LOG.copy()
active_labels = list(HIST_LABELS)
active_kg     = list(HIST_KG)
new_added     = False

if run_update and new_value_kg > 0:
    active_log    = np.append(active_log, np.log1p(new_value_kg))
    active_labels = active_labels + [new_period_lbl]
    active_kg     = active_kg + [new_value_kg]
if run_btn and new_value > 0:
    active_log    = np.append(active_log, np.log1p(new_value))
    active_labels = active_labels + [new_period]
    active_kg     = active_kg + [new_value]
    new_added     = True

# Compute forecast
forecast_kg = run_forecast(
    active_log, selected_model,
    gbr_model, lstm_params, sc_min, sc_rng, ensemble_weights, extra_models, n=4
)
n_active = len(active_labels)
future_labels_used = NEXT_PERIODS[:4]

# ─────────────────────────────────────────────────────────────
# RUN SELECTED MODEL — LIVE PREDICTIONS
# ─────────────────────────────────────────────────────────────
with st.spinner(f'🌿 Running {selected_model} forecast…'):
    forecast_kg = run_model(selected_model, active_log, n_ahead=4)

future_labels_show = FUTURE_LABELS.copy()

# Also compute all-model forecasts for comparison tab
@st.cache_data
def all_model_forecasts(log_arr_tuple):
    log_arr = np.array(log_arr_tuple)
    results = {}
    for m in MODEL_COLORS.keys():
        try:
            results[m] = run_model(m, log_arr, n_ahead=4)
        except Exception:
            results[m] = [0]*4
    return results

all_fc = all_model_forecasts(tuple(active_log.tolist()))


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
mc = MODEL_COLORS[selected_model]
st.markdown(f"""
<div class="header">
  <h1>🌿 Kenya Cannabis Seizure Forecast Tool</h1>
  <p>
    Model: <b>{selected_model}</b> &nbsp;|&nbsp;
    Active model: <b>{selected_model}</b> &nbsp;|&nbsp;
    NACADA Data 2021–2025 &nbsp;|&nbsp;
    4-Period Forecast (2025 H2 – 2027 H1) &nbsp;|&nbsp;
    All predictions from saved trained models &nbsp;|&nbsp;
    BSc Data Science, JKUAT Karen
  </p>
</div>
""", unsafe_allow_html=True)

if new_added:
    st.markdown(f"""<div class='info'>
    ✅ <b>New NACADA observation added:</b>
    {new_period} = {new_value:,.0f} kg.
    All forecasts recalculated live from the trained models.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
met = MODEL_METRICS.get(selected_model, {})

with c1:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{MODEL_COLORS[selected_model]};'>
        {forecast_kg[0]:,} kg</div>
    <div class='kpi-lbl'>Forecast — Next Period</div>
    <div class='kpi-sub'>2025 H2 · {selected_model}</div>
    <div class='kpi-val' style='color:{mc};'>{forecast_kg[0]:,} kg</div>
    <div class='kpi-lbl'>Forecast — {future_labels_show[0]}</div>
    <div class='kpi-sub'>{selected_model}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    last = active_kg[-1]
    chg  = (forecast_kg[0]-last)/last*100
    arr  = '↑' if chg>0 else '↓'
    col  = RED if chg>10 else (GRN if chg<-5 else AMB)
    chg  = (forecast_kg[0] - last) / last * 100
    arr  = '↑' if chg > 0 else '↓'
    col  = RED if chg > 10 else (GRN2 if chg < -5 else AMB)
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{col};'>{arr} {abs(chg):.1f}%</div>
    <div class='kpi-lbl'>Change vs Last Period</div>
    <div class='kpi-sub'>Last: {last:,.0f} kg ({active_labels[-1]})</div>
    <div class='kpi-lbl'>vs Last Period ({active_labels[-1]})</div>
    <div class='kpi-sub'>Last: {last:,.0f} kg</div>
    </div>""", unsafe_allow_html=True)

with c3:
    m_metrics = {
        'E2 Ensemble': (6.91, 1369.2),
        'XGBoost'    : (7.08, 1046.0),
        'LSTM'       : (22.94, 3393.4),
        'SARIMA'     : (67.67, 10525.1),
        'ARIMA'      : (85.87, 11727.5),
        'Prophet'    : (116.63, 18336.9),
    }
    mape_val, rmse_val = m_metrics.get(selected_model, (0, 0))
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>{met.get('mape','—')}%</div>
    <div class='kpi-lbl'>Model Test MAPE</div>
    <div class='kpi-sub'>Rank #{met.get('rank','—')} of 6 models</div>
    <div class='kpi-val'>{mape_val:.2f}%</div>
    <div class='kpi-lbl'>Test MAPE — {selected_model}</div>
    <div class='kpi-sub'>Lower is better</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>{met.get('rmse',0):,.0f} kg</div>
    <div class='kpi-lbl'>Model Test RMSE</div>
    <div class='kpi-sub'>CV RMSE: {met.get('cv',0):,.0f} kg</div>
    </div>""", unsafe_allow_html=True)

if new_added:
    st.markdown(f"""<div class='info'>
    ✅ <b>Forecast updated with new observation:</b>
    {new_period_lbl} = {new_value_kg:,.0f} kg.
    All four future period forecasts have been recalculated.
    <div class='kpi-val'>{rmse_val:,.0f} kg</div>
    <div class='kpi-lbl'>Test RMSE — {selected_model}</div>
    <div class='kpi-sub'>CV RMSE also available</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
@@ -714,203 +634,182 @@ def build_model_comparison_rows(base_log, gbr, lstm_params, sc_min, sc_rng, ense
tab1, tab2, tab3, tab4 = st.tabs([
    '📈 Forecast Chart',
    '📋 Forecast Table',
    '🏘️ County Risk Map',
    '📊 Model Comparison',
    '🏘️ County Cluster Lookup',
    '📊 All Models Comparison',
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">📈 Full Historical Series + 4-Period Forecast</div>',
    st.markdown('<div class="sec-title">📈 Historical Actuals + Live Model Forecast</div>',
                unsafe_allow_html=True)

    mc = MODEL_COLORS[selected_model]
    all_x_labels = active_labels + future_labels_used
    n_tot        = len(all_x_labels)

    fig = go.Figure()
    all_x_labels = active_labels + future_labels_show
    n_tot = len(all_x_labels)
    fig   = go.Figure()

    # Region shading
    fig.add_vrect(x0=0, x1=7,
    # Shading
    fig.add_vrect(x0=0, x1=TRAIN_END,
                  fillcolor='rgba(26,92,42,.04)', line_width=0,
                  annotation_text='Training', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=7, x1=N_HIST-.5,
    fig.add_vrect(x0=TRAIN_END, x1=N_HIST-.5,
                  fillcolor='rgba(239,159,39,.07)', line_width=0,
                  annotation_text='Test', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=n_active-.5, x1=n_tot-.5,
                  fillcolor=f'rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast', annotation_font_size=10,
                  fillcolor='rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast →', annotation_font_size=10,
                  annotation_position='top right')

    # Historical actuals — thick green line
    # Historical actuals
    fig.add_trace(go.Scatter(
        x=list(range(n_active)), y=active_kg,
        mode='lines+markers',
        name='Actual seizures',
        mode='lines+markers', name='Actual (NACADA)',
        line=dict(color=GRN, width=3.5),
        marker=dict(size=10, color=GRN, line=dict(width=2, color='white')),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
        text=active_labels,
    ))

    # Value labels on actual points
    # Value annotations on actuals
    for i, (lbl, v) in enumerate(zip(active_labels, active_kg)):
        fig.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                           showarrow=False, yshift=16,
                           showarrow=False, yshift=18,
                           font=dict(size=9.5, color=GRN))

    # Connector from last actual to first forecast
    fig.add_trace(go.Scatter(
        x=[n_active-1, n_active], y=[active_kg[-1], forecast_kg[0]],
        mode='lines', line=dict(color=mc, width=2.2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Forecast line + confidence band
    # Confidence band
    fc_x = list(range(n_active, n_tot))
    fig.add_trace(go.Scatter(
        x=fc_x+fc_x[::-1],
        x=fc_x + fc_x[::-1],
        y=[v*1.12 for v in forecast_kg]+[v*0.88 for v in reversed(forecast_kg)],
        fill='toself',
        fillcolor=f'rgba({int(mc[1:3],16)},{int(mc[3:5],16)},{int(mc[5:],16)},0.13)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
        name='95% confidence band',
        fillcolor=f'rgba({int(mc[1:3],16)},{int(mc[3:5],16)},{int(mc[5:],16)},0.12)',
        line=dict(width=0), showlegend=True, hoverinfo='skip',
        name='±12% uncertainty band',
    ))

    # Connector
    fig.add_trace(go.Scatter(
        x=[n_active-1, n_active], y=[active_kg[-1], forecast_kg[0]],
        mode='lines', line=dict(color=mc, width=2, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_x, y=forecast_kg,
        mode='lines+markers+text',
        name=f'{selected_model} forecast',
        name=f'{selected_model} forecast (live)',
        line=dict(color=mc, width=3),
        marker=dict(size=11, color=mc, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=[f'<b>{v:,}</b>' for v in forecast_kg],
        textposition='top center',
        textfont=dict(size=10, color=mc),
        hovertemplate=(f'<b>{selected_model}</b><br>'
                       '%{text} — %{x}<extra></extra>'),
        text=[f'<b>{v:,} kg</b>' for v in forecast_kg],
        textposition='top center', textfont=dict(size=10, color=mc),
        hovertemplate=f'<b>{selected_model}</b><br>%{{x}}: <b>%{{y:,.0f}} kg</b><extra></extra>',
    ))

    # Vertical separator
    # Separator
    fig.add_vline(x=n_active-.5,
                  line=dict(color=GRAY, dash='dot', width=1.5),
                  annotation_text='← History | Forecast →',
                  annotation_position='top', annotation_font=dict(color=GRAY, size=10))
                  annotation_text='History | Forecast',
                  annotation_position='top',
                  annotation_font=dict(color=GRAY, size=10))

    fig.update_layout(
        title=dict(
            text=(f'<b>National Cannabis Seizures — Kenya</b><br>'
                  f'<span style="font-size:13px;color:{GRAY};">'
                  f'Historical 2021–{active_labels[-1]}  ·  '
                  f'{selected_model} Forecast 2025 H2–2027 H1</span>'),
                  f'Historical: {active_labels[0]}–{active_labels[-1]}  ·  '
                  f'{selected_model} Live Forecast: '
                  f'{future_labels_show[0]}–{future_labels_show[-1]}</span>'),
            x=0.01, font=dict(size=16, color=GRN)
        ),
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        xaxis=dict(tickmode='array', tickvals=list(range(n_tot)),
                   ticktext=all_x_labels, tickangle=-32,
                   title='Period', gridcolor='#E8F5E9',
                   showline=True, linecolor=GRN2),
                   title='Period', gridcolor='#E8F5E9'),
        yaxis=dict(title='Cannabis seized (kg)', tickformat=',.0f',
                   gridcolor='#E8F5E9', showline=True, linecolor=GRN2),
                   gridcolor='#E8F5E9'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,.85)',
                    bordercolor=GRN2, borderwidth=1),
        hovermode='x unified', height=510,
        hovermode='x unified', height=500,
        margin=dict(t=90, b=65, l=75, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Confidence note
    st.markdown(f"""<div class='info'>
    <b>Shaded band</b> = ±12% uncertainty range around the {selected_model} forecast.
    The actual confidence interval is derived from the model's training residuals.
    <b>E2 Ensemble</b> (XGBoost 79.5% + LSTM 20.5%) achieved the best test MAPE
    of <b>6.91%</b> — the lowest of all models evaluated.
    <b>Live prediction:</b> All values above are computed in real time by the
    <b>{selected_model}</b> model loaded from <code>models/</code>.
    No values are hardcoded. Uncertainty band = ±12% around each forecast point.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📋 Complete Forecast Values (kg)</div>',
    st.markdown('<div class="sec-title">📋 Historical Actuals & Live Forecasts (kg)</div>',
                unsafe_allow_html=True)

    # Historical rows
    rows = []
    for lbl, val in zip(active_labels, active_kg):
        rows.append({'Period':lbl, 'Actual (kg)':f'{val:,.1f}',
                     'Forecast (kg)':'—',
                     'Type':'✅ Observed',
                     'Model':'-'})

    # Future rows
    for lbl, val in zip(future_labels_used, forecast_kg):
        rows.append({'Period':lbl,
                     'Actual (kg)':'Not yet reported',
                     'Forecast (kg)':f'{val:,}',
                     'Type':'🔮 Forecast',
                     'Model':selected_model})
    for lbl, v in zip(active_labels, active_kg):
        rows.append({'Period':lbl, 'Actual (kg)':f'{v:,.1f}',
                     f'{selected_model} Forecast':'—', 'Type':'✅ Observed'})
    for lbl, v in zip(future_labels_show, forecast_kg):
        rows.append({'Period':lbl, 'Actual (kg)':'Not yet reported',
                     f'{selected_model} Forecast':f'{v:,}', 'Type':'🔮 Live Forecast'})

    df_show = pd.DataFrame(rows)

    def hl(row):
        if row['Type'] == '🔮 Forecast':
            return ['background-color:#E8F5E9; font-weight:bold']*len(row)
    def hl_fc(row):
        if '🔮' in str(row['Type']):
            return ['background-color:#E8F5E9;font-weight:bold']*len(row)
        return ['']*len(row)

    st.dataframe(df_show.style.apply(hl, axis=1),
    st.dataframe(df_show.style.apply(hl_fc, axis=1),
                 use_container_width=True, hide_index=True)

    # Download
    csv = df_show.to_csv(index=False)
    st.download_button(
        label='⬇️ Download as CSV',
        data=csv,
        file_name=f'kenya_seizure_forecast_{selected_model.replace(" ","_")}.csv',
        mime='text/csv'
    )
    st.download_button('⬇️ Download as CSV', data=csv,
                       file_name=f'seizure_forecast_{selected_model.replace(" ","_")}.csv',
                       mime='text/csv')

    # All-model future forecasts
    st.markdown('<div class="sec-title">📊 All Models — Future Forecast Comparison (kg)</div>',
    # All models side by side
    st.markdown('<div class="sec-title">📊 All Models — Live Forecast Comparison (kg)</div>',
                unsafe_allow_html=True)

    cmp_rows = build_model_comparison_rows(
        active_log,
        gbr_model,
        lstm_params,
        sc_min,
        sc_rng,
        ensemble_weights,
        extra_models,
    )

    df_cmp = pd.DataFrame(cmp_rows, columns=['Model', 'Type', *future_labels_used, 'Test MAPE'])

    def hl2(row):
    cmp_rows = []
    for m in MODEL_COLORS.keys():
        row = {'Model':m}
        for lbl, v in zip(future_labels_show, all_fc[m]):
            row[lbl] = f'{v:,}'
        mape_v = m_metrics.get(m,(0,0))[0]
        row['Test MAPE'] = f'{mape_v:.2f}%'
        cmp_rows.append(row)

    df_cmp = pd.DataFrame(cmp_rows)
    def hl_ens(row):
        if 'Ensemble' in str(row['Model']):
            return ['background-color:#E8F5E9; font-weight:bold']*len(row)
            return ['background-color:#E8F5E9;font-weight:bold']*len(row)
        return ['']*len(row)

    st.dataframe(df_cmp.style.apply(hl2, axis=1),
    st.dataframe(df_cmp.style.apply(hl_ens, axis=1),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — COUNTY RISK MAP
# TAB 3 — COUNTY CLUSTER LOOKUP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏘️ County Cluster Risk Zones</div>',
    st.markdown('<div class="sec-title">🏘️ County Cluster Lookup</div>',
                unsafe_allow_html=True)

    # County lookup
    if county_query:
        q = county_query.strip()
    if county_q:
        q     = county_q.strip()
        match = None
        for nm in COUNTY_DATA:
            if nm.lower() == q.lower(): match=nm; break
@@ -921,16 +820,14 @@ def hl2(row):
                st.warning(f'Multiple matches: {", ".join(cands)}. Be more specific.')

        if match:
            d  = COUNTY_DATA[match]
            ci = CLUSTER_CFG[d['cluster']]
            c1, c2 = st.columns([1, 2])
            d=COUNTY_DATA[match]; ci=CLUSTER_CFG[d['cluster']]
            c1,c2=st.columns([1,2])
            with c1:
                st.markdown(f"""
                <div style='background:{ci["bg"]};border:2px solid {ci["color"]};
                     border-radius:14px;padding:1.5rem;text-align:center;'>
                  <div style='font-size:3rem;'>{ci["icon"]}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ci["color"]};'>
                    {match}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ci["color"]};'>{match}</div>
                  <div style='margin:.5rem 0;'>
                    <span class='{ci["badge"]}'>{d["cluster"]} Tier</span></div>
                  <div style='color:{ci["color"]};font-weight:600;font-size:.9rem;'>
@@ -942,12 +839,10 @@ def hl2(row):
                     border-radius:12px;padding:1.2rem;'>
                <table style='width:100%;border-collapse:collapse;'>
                  <tr><td style='padding:.5rem;color:#666;width:42%;'><b>Cluster Tier</b></td>
                      <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>
                        {d["cluster"]}</td></tr>
                      <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>{d["cluster"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>National Share</b></td>
                    <td style='padding:.5rem;font-weight:700;color:{ci["color"]};'>
                      {ci["share"]} of all seizures</td></tr>
                    <td style='padding:.5rem;font-weight:700;color:{ci["color"]};'>{ci["share"]}</td></tr>
                  <tr><td style='padding:.5rem;color:#666;'><b>Total (2021–2025)</b></td>
                      <td style='padding:.5rem;'>{d["total_kg"]:,.1f} kg</td></tr>
                  <tr style='background:#f9f9f9;'>
@@ -957,132 +852,110 @@ def hl2(row):
                      <td style='padding:.5rem;'>{d["trend"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>NACADA Action</b></td>
                    <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>
                      {ci["action"]}</td></tr>
                    <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>{ci["action"]}</td></tr>
                </table></div>""", unsafe_allow_html=True)

            others = [n for n,v in COUNTY_DATA.items()
                      if v['cluster']==d['cluster'] and n!=match]
            others=[n for n,v in COUNTY_DATA.items() if v['cluster']==d['cluster'] and n!=match]
            st.markdown(f"""<div class='info'>
            <b>Other counties in the {d["cluster"]} tier:</b><br>
            {", ".join(others)}
            <b>Other {d["cluster"]} tier counties:</b> {", ".join(others)}
            </div>""", unsafe_allow_html=True)

        elif county_query and not match:
        elif county_q and not match:
            st.markdown(f"""<div class='warn'>
            County <b>"{county_query}"</b> not found. Check spelling or use the table below.
            County <b>"{county_q}"</b> not found. Check spelling.
            </div>""", unsafe_allow_html=True)

    # High-tier alert box
    high_counties = [n for n,d in COUNTY_DATA.items() if d['cluster']=='High']
    # High-tier alert
    high = [n for n,d in COUNTY_DATA.items() if d['cluster']=='High']
    st.markdown(f"""<div class='danger'>
    🔴 <b>10 HIGH-TIER COUNTIES (63.5% of national seizures):</b><br>
    {" · ".join(high_counties)}<br><br>
    🔴 <b>10 HIGH-TIER COUNTIES (63.5% of all seizures):</b><br>
    {" · ".join(high)}<br><br>
    <b>Trafficking corridors:</b> Tanzania border (Migori) ·
    Uganda border (Busia) · Indian Ocean coast (Kilifi) ·
    Northern Corridor/Moyale highway (Marsabit)
    Northern Corridor/Moyale (Marsabit)
    </div>""", unsafe_allow_html=True)

    # Full county table
    st.markdown('<div class="sec-title">📋 All 49 Counties — Cluster Assignments</div>',
                unsafe_allow_html=True)

    tier_order = {'High':0,'Med-High':1,'Medium':2,'Low':3}
    tbl_rows   = sorted(
        [{'County':n, 'Cluster Tier':d['cluster'],
          'Total kg (2021–2025)':f"{d['total_kg']:,.1f}",
          'Avg per Period (kg)':f"{d['total_kg']/9:,.1f}",
    st.markdown('<div class="sec-title">📋 All 49 Counties</div>', unsafe_allow_html=True)
    tier_order={'High':0,'Med-High':1,'Medium':2,'Low':3}
    tbl_rows=sorted(
        [{'County':n,'Cluster Tier':d['cluster'],
          'Total kg':f"{d['total_kg']:,.1f}",
          'Avg/Period':f"{d['total_kg']/9:,.1f}",
          'Trend':d['trend']}
         for n,d in COUNTY_DATA.items()],
        key=lambda r:(tier_order[r['Cluster Tier']], -float(r['Total kg (2021–2025)'].replace(',','')))
        key=lambda r:(tier_order[r['Cluster Tier']],
                      -float(r['Total kg'].replace(',','')))
    )
    df_ct = pd.DataFrame(tbl_rows)

    tier_bg = {'High':'#FDECEA','Med-High':'#FFF3CD',
               'Medium':'#D6E4F7','Low':'#F2F2F2'}

    def hl_tier(row):
        bg = tier_bg.get(row['Cluster Tier'],'')
        return [f'background-color:{bg}']*len(row)

    st.dataframe(df_ct.style.apply(hl_tier, axis=1),
                 use_container_width=True, hide_index=True, height=440)
    df_ct=pd.DataFrame(tbl_rows)
    tier_bg={'High':'#FDECEA','Med-High':'#FFF3CD','Medium':'#D6E4F7','Low':'#F2F2F2'}
    def hl_t(row):
        return [f'background-color:{tier_bg.get(row["Cluster Tier"],"")}']*len(row)
    st.dataframe(df_ct.style.apply(hl_t,axis=1),
                 use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# TAB 4 — ALL MODELS COMPARISON
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">📊 Model Performance Comparison</div>',
    st.markdown('<div class="sec-title">📊 Live Forecast Comparison — All 6 Models</div>',
                unsafe_allow_html=True)

    cmp = [
        {'Model':'E2 Ensemble (Recommended)','Type':'⭐ Ensemble',
         'Test RMSE (kg)':'1,369.2','Test MAPE (%)':'6.91',
         'CV RMSE (kg)':'1,369.2','Rank':'#1 — Best MAPE','Recommended':'✅ PRIMARY'},
        {'Model':'XGBoost','Type':'Individual',
         'Test RMSE (kg)':'1,046.0','Test MAPE (%)':'7.08',
         'CV RMSE (kg)':'1,532.1','Rank':'#2 — Best RMSE','Recommended':'✅ BACKUP'},
        {'Model':'LSTM','Type':'Individual',
         'Test RMSE (kg)':'3,393.4','Test MAPE (%)':'22.94',
         'CV RMSE (kg)':'5,928.7','Rank':'#3','Recommended':'✅ Consistency check'},
        {'Model':'SARIMA','Type':'Statistical',
         'Test RMSE (kg)':'10,525.1','Test MAPE (%)':'67.67',
         'CV RMSE (kg)':'3,459.8','Rank':'#4','Recommended':''},
        {'Model':'ARIMA','Type':'Statistical',
         'Test RMSE (kg)':'11,727.5','Test MAPE (%)':'85.87',
         'CV RMSE (kg)':'9,777.6','Rank':'#5','Recommended':''},
        {'Model':'Prophet','Type':'Statistical',
         'Test RMSE (kg)':'18,336.9','Test MAPE (%)':'116.63',
         'CV RMSE (kg)':'5,092.6','Rank':'#6','Recommended':''},
    ]
    df_cmp2 = pd.DataFrame(cmp)

    def hl3(row):
        if 'Ensemble' in str(row['Model']):
            return ['background-color:#E8F5E9;font-weight:bold']*len(row)
        if row['Model']=='XGBoost':
            return ['background-color:#FFF8E8']*len(row)
        return ['']*len(row)
    # Bar chart — Test MAPE
    names  = list(MODEL_COLORS.keys())
    mapes  = [m_metrics.get(n,(0,0))[0] for n in names]
    bclrs  = [MODEL_COLORS[n] for n in names]

    st.dataframe(df_cmp2.style.apply(hl3, axis=1),
                 use_container_width=True, hide_index=True)

    # MAPE bar chart
    st.markdown('<div class="sec-title">Test MAPE by Model (lower = better)</div>',
                unsafe_allow_html=True)

    names  = [r['Model'] for r in cmp]
    mapes  = [float(r['Test MAPE (%)']) for r in cmp]
    bcolors= [MODEL_COLORS.get(n, GRN2) for n in names]

    fig2 = go.Figure(go.Bar(
    fig2=go.Figure(go.Bar(
        x=[n.replace(' (Recommended)','') for n in names],
        y=mapes,
        marker_color=bcolors,
        marker_color=bclrs,
        marker_line_color='white', marker_line_width=1.2,
        text=[f'{v:.1f}%' for v in mapes],
        textposition='outside',
        hovertemplate='%{x}<br>Test MAPE: <b>%{y:.2f}%</b><extra></extra>',
    ))
    fig2.update_layout(
        title='Test MAPE by Model (lower = better)',
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        yaxis=dict(title='Test MAPE (%)', gridcolor='#E8F5E9', range=[0, max(mapes)*1.22]),
        xaxis=dict(title='Model', tickangle=-20),
        height=380, margin=dict(t=30,b=60,l=60,r=20),
        yaxis=dict(title='Test MAPE (%)', gridcolor='#E8F5E9',
                   range=[0,max(mapes)*1.22]),
        xaxis=dict(tickangle=-20),
        height=360, margin=dict(t=50,b=60,l=60,r=20),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Live forecast line chart — all models
    fig3=go.Figure()
    for m in MODEL_COLORS.keys():
        col=MODEL_COLORS[m]
        lw =3 if 'Ensemble' in m else 1.8
        fig3.add_trace(go.Scatter(
            x=future_labels_show, y=all_fc[m],
            mode='lines+markers', name=m,
            line=dict(color=col, width=lw,
                      dash='solid' if 'Ensemble' in m else 'dot'),
            marker=dict(size=8 if 'Ensemble' in m else 6),
            hovertemplate=f'<b>{m}</b><br>%{{x}}: <b>%{{y:,.0f}} kg</b><extra></extra>',
        ))
    fig3.update_layout(
        title='Live Forecasts — All Models (2025 H2 – 2027 H1)',
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        yaxis=dict(title='kg seized', tickformat=',.0f', gridcolor='#E8F5E9'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=380, margin=dict(t=70,b=40,l=70,r=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""<div class='result'>
    <b>🌿 Why E2 Ensemble is recommended:</b><br>
    The E2 Ensemble combines XGBoost (weight 79.5%) and LSTM (weight 20.5%) using
    inverse cross-validation weighting. It achieves the best overall Test MAPE of
    <b>6.91%</b> — beating XGBoost alone (7.08%) — and forecasts a stable plateau
    of approximately <b>12,600–12,800 kg per half-year</b> through 2027 H1.<br><br>
    <b>Model selection principle:</b> Cross-validation RMSE is the primary criterion
    because the holdout test set contains only 2 data points — too few for reliable
    ranking by test RMSE alone.
    <b>🌿 All forecasts above are computed live from saved trained models.</b><br>
    No values are hardcoded in this app. When you enter a new NACADA
    observation in the sidebar, all six models recalculate their forecasts
    in real time using the new data point as the latest known value.<br><br>
    <b>Recommended for NACADA use:</b> E2 Ensemble (XGBoost 79.5% + LSTM 20.5%,
    CV-weighted) — best test MAPE of <b>6.91%</b>.
    </div>""", unsafe_allow_html=True)


@@ -1094,8 +967,8 @@ def hl3(row):
<div style='text-align:center;color:{GRAY};font-size:.82rem;padding:.4rem;'>
🌿 Kenya Cannabis Seizure Forecast Tool &nbsp;|&nbsp;
William Maureen Ndinda (SCT213-C002-0048/2022) &nbsp;|&nbsp;
BSc Data Science &amp; Analytics &nbsp;|&nbsp; JKUAT Karen &nbsp;|&nbsp; 2026<br>
Data: NACADA Bi-Annual Reports 2021–2025 &nbsp;|&nbsp;
Models: E2 Ensemble · XGBoost · LSTM · SARIMA · ARIMA · Prophet
BSc Data Science & Analytics &nbsp;|&nbsp; JKUAT Karen &nbsp;|&nbsp; 2026<br>
All predictions computed live from saved models in <code>models/</code> folder &nbsp;|&nbsp;
NACADA Bi-Annual Reports 2021–2025
</div>
""", unsafe_allow_html=True)
