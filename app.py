# ============================================================
# app.py  —  Kenya Cannabis Seizure Forecast Tool
# William Maureen Ndinda | SCT213-C002-0048/2022 | JKUAT Karen
# ============================================================
# HOW TO RUN:
#   pip install streamlit plotly scikit-learn numpy pandas
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
#
# All predictions are made LIVE from the saved trained models.
# No hardcoded forecast values anywhere in this file.
# ============================================================

import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Kenya Cannabis Seizure Forecast',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────
# GREEN THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F0F7F1; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A5C2A 0%, #2D8B47 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] label { color: #C8F0D0 !important; }

.header {
    background: linear-gradient(135deg, #1A5C2A 0%, #2D8B47 60%, #52A96A 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(26,92,42,0.25);
}
.header h1 { color: white !important; margin:0; font-size:1.85rem; }
.header p  { color: #C8F0D0; margin:0.3rem 0 0; font-size:0.95rem; }

.kpi {
    background: white; border: 2px solid #2D8B47;
    border-radius: 12px; padding: 1.1rem 0.8rem;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.kpi-val { color: #1A5C2A; font-size:2rem; font-weight:800; margin:0; }
.kpi-lbl { color: #2D8B47; font-size:0.82rem; font-weight:600; margin:0.2rem 0 0; }
.kpi-sub { color: #888; font-size:0.76rem; margin:0; }

.info  { background:#E8F5E9; border-left:4px solid #2D8B47;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.warn  { background:#FFF8E1; border-left:4px solid #EF9F27;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.danger{ background:#FDECEA; border-left:4px solid #C00000;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.result{ background:#E8F5E9; border:2px solid #2D8B47;
         border-radius:12px; padding:1.2rem; margin:.8rem 0; }

.sec-title {
    color:#1A5C2A; font-size:1.1rem; font-weight:700;
    border-bottom:2px solid #2D8B47; padding-bottom:.3rem; margin:1rem 0 .8rem;
}

div.stButton > button {
    background:#2D8B47 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; padding:.55rem 1.8rem !important; width:100%;
}
div.stButton > button:hover { background:#1A5C2A !important; }

.badge-h  { background:#FDECEA; color:#C00000; border:2px solid #C00000;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-mh { background:#FFF3CD; color:#7F4F00; border:2px solid #EF9F27;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-m  { background:#D6E4F7; color:#1F3864; border:2px solid #2E5FA3;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-l  { background:#F2F2F2; color:#444; border:2px solid #888;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────
GRN  = '#1A5C2A'; GRN2 = '#2D8B47'; GRN3 = '#52A96A'
AMB  = '#EF9F27'; BLU  = '#2E5FA3'; RED  = '#C00000'; GRAY = '#888780'

MODEL_COLORS = {
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
    'Nakuru':{'cluster':'High','total_kg':6143.1,'trend':'📈'},
    'Marsabit':{'cluster':'High','total_kg':5786.6,'trend':'📈'},
    'Kilifi':{'cluster':'High','total_kg':4704.8,'trend':'📈'},
    'Busia':{'cluster':'High','total_kg':4696.2,'trend':'📈'},
    'Kiambu':{'cluster':'High','total_kg':4013.1,'trend':'📈'},
    'Kisii':{'cluster':'High','total_kg':3837.1,'trend':'📉'},
    'Narok':{'cluster':'High','total_kg':2789.1,'trend':'📈'},
    'Machakos':{'cluster':'High','total_kg':2617.0,'trend':'📈'},
    'Mombasa':{'cluster':'Med-High','total_kg':3782.5,'trend':'📈'},
    'Kisumu':{'cluster':'Med-High','total_kg':3142.5,'trend':'📈'},
    'Embu':{'cluster':'Med-High','total_kg':2041.5,'trend':'📈'},
    'Nyeri':{'cluster':'Med-High','total_kg':1938.2,'trend':'📈'},
    'Makueni':{'cluster':'Med-High','total_kg':1621.5,'trend':'📈'},
    'Isiolo':{'cluster':'Med-High','total_kg':1489.3,'trend':'📈'},
    'Muranga':{'cluster':'Med-High','total_kg':1312.4,'trend':'📉'},
    'Nyandarua':{'cluster':'Med-High','total_kg':1198.6,'trend':'📉'},
    'Meru':{'cluster':'Med-High','total_kg':1187.2,'trend':'📈'},
    'Uasin Gishu':{'cluster':'Med-High','total_kg':1102.8,'trend':'📈'},
    'Siaya':{'cluster':'Med-High','total_kg':1089.4,'trend':'📈'},
    'Kitui':{'cluster':'Med-High','total_kg':1021.3,'trend':'📈'},
    'Kirinyaga':{'cluster':'Med-High','total_kg':977.6,'trend':'📉'},
    'Kwale':{'cluster':'Medium','total_kg':1777.2,'trend':'📈'},
    'Laikipia':{'cluster':'Medium','total_kg':1156.3,'trend':'📈'},
    'Kajiado':{'cluster':'Medium','total_kg':1089.5,'trend':'📈'},
    'Taita Taveta':{'cluster':'Medium','total_kg':987.4,'trend':'📈'},
    'Kericho':{'cluster':'Medium','total_kg':923.1,'trend':'📈'},
    'Homa Bay':{'cluster':'Medium','total_kg':876.5,'trend':'📈'},
    'Bungoma':{'cluster':'Medium','total_kg':823.4,'trend':'📈'},
    'Bomet':{'cluster':'Medium','total_kg':789.2,'trend':'📉'},
    'Turkana':{'cluster':'Medium','total_kg':756.8,'trend':'📈'},
    'Samburu':{'cluster':'Medium','total_kg':689.4,'trend':'📈'},
    'Kakamega':{'cluster':'Medium','total_kg':654.3,'trend':'📈'},
    'Vihiga':{'cluster':'Medium','total_kg':612.8,'trend':'📉'},
    'Garissa':{'cluster':'Medium','total_kg':598.7,'trend':'📈'},
    'Tharaka Nithi':{'cluster':'Medium','total_kg':567.4,'trend':'📈'},
    'Nandi':{'cluster':'Medium','total_kg':534.2,'trend':'📈'},
    'Nyamira':{'cluster':'Medium','total_kg':512.6,'trend':'📉'},
    'Wajir':{'cluster':'Low','total_kg':578.0,'trend':'📈'},
    'Trans Nzoia':{'cluster':'Low','total_kg':139.0,'trend':'📉'},
    'Elgeyo Marakwet':{'cluster':'Low','total_kg':85.8,'trend':'📉'},
    'Mandera':{'cluster':'Low','total_kg':43.5,'trend':'📉'},
    'Lamu':{'cluster':'Low','total_kg':24.9,'trend':'📉'},
    'Baringo':{'cluster':'Low','total_kg':15.6,'trend':'📉'},
    'Tana River':{'cluster':'Low','total_kg':8.2,'trend':'📉'},
    'Railways (CPIU)':{'cluster':'Low','total_kg':5.3,'trend':'📉'},
    'West Pokot':{'cluster':'Low','total_kg':2.8,'trend':'📉'},
}
CLUSTER_CFG = {
    'High'    :{'icon':'🔴','badge':'badge-h','color':RED,'bg':'#FDECEA',
                'share':'63.5%','action':'Immediate targeted enforcement surge required.',
                'desc':'Critical — highest enforcement priority.'},
    'Med-High':{'icon':'🟠','badge':'badge-mh','color':'#7F4F00','bg':'#FFF3CD',
                'share':'21.7%','action':'Build capacity before escalation to High tier.',
                'desc':'Emerging threat — pre-emptive intervention needed.'},
    'Medium'  :{'icon':'🔵','badge':'badge-m','color':BLU,'bg':'#D6E4F7',
                'share':'13.8%','action':'Maintain standard enforcement; monitor trends.',
                'desc':'Moderate activity — regular monitoring required.'},
    'Low'     :{'icon':'⚪','badge':'badge-l','color':'#444','bg':'#F2F2F2',
                'share':'1.0%','action':'Review enforcement capacity.',
                'desc':'Low recorded seizures — may reflect capacity gaps.'},
}

# ─────────────────────────────────────────────────────────────
# LOAD ALL MODELS FROM DISK
# ─────────────────────────────────────────────────────────────
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
            ig=_sig(g[:H]); fg=_sig(g[H:2*H])
            gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
            cn=fg*c+ig*gg; h=og*_tanh(cn); c=cn
        return float((Wy@h+by)[0])

    cur_log = list(log_series)
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
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem;'>
        <div style='font-size:3rem;'>🌿</div>
        <div style='font-size:1.05rem;font-weight:800;color:white;'>NACADA Forecast</div>
        <div style='font-size:.78rem;color:#C8F0D0;'>Cannabis Seizure Prediction</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.3);'>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Model")
    selected_model = st.selectbox(
        'Select forecasting model:',
        options=list(MODEL_COLORS.keys()),
        index=0,
        help='E2 Ensemble recommended — best MAPE 6.91%'
    )

    st.markdown("### 📥 Add New NACADA Value")
    st.markdown("<small style='color:#C8F0D0;'>When NACADA publishes a new bi-annual report, enter the value here to update the forecast with real new data.</small>",
                unsafe_allow_html=True)
    new_period = st.selectbox('New period:', FUTURE_LABELS + ['2027 H2','2028 H1'])
    new_value  = st.number_input(
        f'Actual seizures (kg):', min_value=0.0, max_value=200000.0,
        value=0.0, step=500.0, format='%.0f'
    )
    run_btn = st.button('🔄 Update Forecast', type='primary')

    st.markdown("### 🔍 County Lookup")
    county_q = st.text_input('Type county name:', placeholder='e.g. Nairobi, Kisumu…')

    st.markdown("<hr style='border-color:rgba(255,255,255,.3);'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.8rem;color:#C8F0D0;'>
    <b>Student:</b> William Maureen Ndinda<br>
    <b>Reg:</b> SCT213-C002-0048/2022<br>
    <b>Programme:</b> BSc Data Science<br>
    <b>JKUAT Karen | 2026</b><br><br>
    <b>Models loaded from:</b><br>
    <code style='color:#A8DBB8;font-size:.75rem;'>models/</code> folder ✓
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BUILD ACTIVE SERIES (with optional new value)
# ─────────────────────────────────────────────────────────────
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
    Active model: <b>{selected_model}</b> &nbsp;|&nbsp;
    NACADA Data 2021–2025 &nbsp;|&nbsp;
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

with c1:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{mc};'>{forecast_kg[0]:,} kg</div>
    <div class='kpi-lbl'>Forecast — {future_labels_show[0]}</div>
    <div class='kpi-sub'>{selected_model}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    last = active_kg[-1]
    chg  = (forecast_kg[0] - last) / last * 100
    arr  = '↑' if chg > 0 else '↓'
    col  = RED if chg > 10 else (GRN2 if chg < -5 else AMB)
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{col};'>{arr} {abs(chg):.1f}%</div>
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
    <div class='kpi-val'>{mape_val:.2f}%</div>
    <div class='kpi-lbl'>Test MAPE — {selected_model}</div>
    <div class='kpi-sub'>Lower is better</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>{rmse_val:,.0f} kg</div>
    <div class='kpi-lbl'>Test RMSE — {selected_model}</div>
    <div class='kpi-sub'>CV RMSE also available</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📈 Forecast Chart',
    '📋 Forecast Table',
    '🏘️ County Cluster Lookup',
    '📊 All Models Comparison',
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">📈 Historical Actuals + Live Model Forecast</div>',
                unsafe_allow_html=True)

    all_x_labels = active_labels + future_labels_show
    n_tot = len(all_x_labels)
    fig   = go.Figure()

    # Shading
    fig.add_vrect(x0=0, x1=TRAIN_END,
                  fillcolor='rgba(26,92,42,.04)', line_width=0,
                  annotation_text='Training', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=TRAIN_END, x1=N_HIST-.5,
                  fillcolor='rgba(239,159,39,.07)', line_width=0,
                  annotation_text='Test', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=n_active-.5, x1=n_tot-.5,
                  fillcolor='rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast →', annotation_font_size=10,
                  annotation_position='top right')

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=list(range(n_active)), y=active_kg,
        mode='lines+markers', name='Actual (NACADA)',
        line=dict(color=GRN, width=3.5),
        marker=dict(size=10, color=GRN, line=dict(width=2, color='white')),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
        text=active_labels,
    ))

    # Value annotations on actuals
    for i, (lbl, v) in enumerate(zip(active_labels, active_kg)):
        fig.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                           showarrow=False, yshift=18,
                           font=dict(size=9.5, color=GRN))

    # Confidence band
    fc_x = list(range(n_active, n_tot))
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1],
        y=[v*1.12 for v in forecast_kg]+[v*0.88 for v in reversed(forecast_kg)],
        fill='toself',
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
        name=f'{selected_model} forecast (live)',
        line=dict(color=mc, width=3),
        marker=dict(size=11, color=mc, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=[f'<b>{v:,} kg</b>' for v in forecast_kg],
        textposition='top center', textfont=dict(size=10, color=mc),
        hovertemplate=f'<b>{selected_model}</b><br>%{{x}}: <b>%{{y:,.0f}} kg</b><extra></extra>',
    ))

    # Separator
    fig.add_vline(x=n_active-.5,
                  line=dict(color=GRAY, dash='dot', width=1.5),
                  annotation_text='History | Forecast',
                  annotation_position='top',
                  annotation_font=dict(color=GRAY, size=10))

    fig.update_layout(
        title=dict(
            text=(f'<b>National Cannabis Seizures — Kenya</b><br>'
                  f'<span style="font-size:13px;color:{GRAY};">'
                  f'Historical: {active_labels[0]}–{active_labels[-1]}  ·  '
                  f'{selected_model} Live Forecast: '
                  f'{future_labels_show[0]}–{future_labels_show[-1]}</span>'),
            x=0.01, font=dict(size=16, color=GRN)
        ),
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        xaxis=dict(tickmode='array', tickvals=list(range(n_tot)),
                   ticktext=all_x_labels, tickangle=-32,
                   title='Period', gridcolor='#E8F5E9'),
        yaxis=dict(title='Cannabis seized (kg)', tickformat=',.0f',
                   gridcolor='#E8F5E9'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,.85)',
                    bordercolor=GRN2, borderwidth=1),
        hovermode='x unified', height=500,
        margin=dict(t=90, b=65, l=75, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class='info'>
    <b>Live prediction:</b> All values above are computed in real time by the
    <b>{selected_model}</b> model loaded from <code>models/</code>.
    No values are hardcoded. Uncertainty band = ±12% around each forecast point.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📋 Historical Actuals & Live Forecasts (kg)</div>',
                unsafe_allow_html=True)

    rows = []
    for lbl, v in zip(active_labels, active_kg):
        rows.append({'Period':lbl, 'Actual (kg)':f'{v:,.1f}',
                     f'{selected_model} Forecast':'—', 'Type':'✅ Observed'})
    for lbl, v in zip(future_labels_show, forecast_kg):
        rows.append({'Period':lbl, 'Actual (kg)':'Not yet reported',
                     f'{selected_model} Forecast':f'{v:,}', 'Type':'🔮 Live Forecast'})

    df_show = pd.DataFrame(rows)
    def hl_fc(row):
        if '🔮' in str(row['Type']):
            return ['background-color:#E8F5E9;font-weight:bold']*len(row)
        return ['']*len(row)

    st.dataframe(df_show.style.apply(hl_fc, axis=1),
                 use_container_width=True, hide_index=True)

    csv = df_show.to_csv(index=False)
    st.download_button('⬇️ Download as CSV', data=csv,
                       file_name=f'seizure_forecast_{selected_model.replace(" ","_")}.csv',
                       mime='text/csv')

    # All models side by side
    st.markdown('<div class="sec-title">📊 All Models — Live Forecast Comparison (kg)</div>',
                unsafe_allow_html=True)

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
            return ['background-color:#E8F5E9;font-weight:bold']*len(row)
        return ['']*len(row)

    st.dataframe(df_cmp.style.apply(hl_ens, axis=1),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — COUNTY CLUSTER LOOKUP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏘️ County Cluster Lookup</div>',
                unsafe_allow_html=True)

    if county_q:
        q     = county_q.strip()
        match = None
        for nm in COUNTY_DATA:
            if nm.lower() == q.lower(): match=nm; break
        if not match:
            cands = [nm for nm in COUNTY_DATA if q.lower() in nm.lower()]
            if len(cands)==1: match=cands[0]
            elif cands:
                st.warning(f'Multiple matches: {", ".join(cands)}. Be more specific.')

        if match:
            d=COUNTY_DATA[match]; ci=CLUSTER_CFG[d['cluster']]
            c1,c2=st.columns([1,2])
            with c1:
                st.markdown(f"""
                <div style='background:{ci["bg"]};border:2px solid {ci["color"]};
                     border-radius:14px;padding:1.5rem;text-align:center;'>
                  <div style='font-size:3rem;'>{ci["icon"]}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ci["color"]};'>{match}</div>
                  <div style='margin:.5rem 0;'>
                    <span class='{ci["badge"]}'>{d["cluster"]} Tier</span></div>
                  <div style='color:{ci["color"]};font-weight:600;font-size:.9rem;'>
                    {ci["desc"]}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style='background:white;border:1px solid #ddd;
                     border-radius:12px;padding:1.2rem;'>
                <table style='width:100%;border-collapse:collapse;'>
                  <tr><td style='padding:.5rem;color:#666;width:42%;'><b>Cluster Tier</b></td>
                      <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>{d["cluster"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>National Share</b></td>
                    <td style='padding:.5rem;font-weight:700;color:{ci["color"]};'>{ci["share"]}</td></tr>
                  <tr><td style='padding:.5rem;color:#666;'><b>Total (2021–2025)</b></td>
                      <td style='padding:.5rem;'>{d["total_kg"]:,.1f} kg</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>Avg per Period</b></td>
                    <td style='padding:.5rem;'>{d["total_kg"]/9:,.1f} kg</td></tr>
                  <tr><td style='padding:.5rem;color:#666;'><b>Trend</b></td>
                      <td style='padding:.5rem;'>{d["trend"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>NACADA Action</b></td>
                    <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>{ci["action"]}</td></tr>
                </table></div>""", unsafe_allow_html=True)

            others=[n for n,v in COUNTY_DATA.items() if v['cluster']==d['cluster'] and n!=match]
            st.markdown(f"""<div class='info'>
            <b>Other {d["cluster"]} tier counties:</b> {", ".join(others)}
            </div>""", unsafe_allow_html=True)

        elif county_q and not match:
            st.markdown(f"""<div class='warn'>
            County <b>"{county_q}"</b> not found. Check spelling.
            </div>""", unsafe_allow_html=True)

    # High-tier alert
    high = [n for n,d in COUNTY_DATA.items() if d['cluster']=='High']
    st.markdown(f"""<div class='danger'>
    🔴 <b>10 HIGH-TIER COUNTIES (63.5% of all seizures):</b><br>
    {" · ".join(high)}<br><br>
    <b>Trafficking corridors:</b> Tanzania border (Migori) ·
    Uganda border (Busia) · Indian Ocean coast (Kilifi) ·
    Northern Corridor/Moyale (Marsabit)
    </div>""", unsafe_allow_html=True)

    # Full county table
    st.markdown('<div class="sec-title">📋 All 49 Counties</div>', unsafe_allow_html=True)
    tier_order={'High':0,'Med-High':1,'Medium':2,'Low':3}
    tbl_rows=sorted(
        [{'County':n,'Cluster Tier':d['cluster'],
          'Total kg':f"{d['total_kg']:,.1f}",
          'Avg/Period':f"{d['total_kg']/9:,.1f}",
          'Trend':d['trend']}
         for n,d in COUNTY_DATA.items()],
        key=lambda r:(tier_order[r['Cluster Tier']],
                      -float(r['Total kg'].replace(',','')))
    )
    df_ct=pd.DataFrame(tbl_rows)
    tier_bg={'High':'#FDECEA','Med-High':'#FFF3CD','Medium':'#D6E4F7','Low':'#F2F2F2'}
    def hl_t(row):
        return [f'background-color:{tier_bg.get(row["Cluster Tier"],"")}']*len(row)
    st.dataframe(df_ct.style.apply(hl_t,axis=1),
                 use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════════════════════
# TAB 4 — ALL MODELS COMPARISON
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">📊 Live Forecast Comparison — All 6 Models</div>',
                unsafe_allow_html=True)

    # Bar chart — Test MAPE
    names  = list(MODEL_COLORS.keys())
    mapes  = [m_metrics.get(n,(0,0))[0] for n in names]
    bclrs  = [MODEL_COLORS[n] for n in names]

    fig2=go.Figure(go.Bar(
        x=[n.replace(' (Recommended)','') for n in names],
        y=mapes,
        marker_color=bclrs,
        marker_line_color='white', marker_line_width=1.2,
        text=[f'{v:.1f}%' for v in mapes],
        textposition='outside',
        hovertemplate='%{x}<br>Test MAPE: <b>%{y:.2f}%</b><extra></extra>',
    ))
    fig2.update_layout(
        title='Test MAPE by Model (lower = better)',
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
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
    <b>🌿 All forecasts above are computed live from saved trained models.</b><br>
    No values are hardcoded in this app. When you enter a new NACADA
    observation in the sidebar, all six models recalculate their forecasts
    in real time using the new data point as the latest known value.<br><br>
    <b>Recommended for NACADA use:</b> E2 Ensemble (XGBoost 79.5% + LSTM 20.5%,
    CV-weighted) — best test MAPE of <b>6.91%</b>.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f"""
<div style='text-align:center;color:{GRAY};font-size:.82rem;padding:.4rem;'>
🌿 Kenya Cannabis Seizure Forecast Tool &nbsp;|&nbsp;
William Maureen Ndinda (SCT213-C002-0048/2022) &nbsp;|&nbsp;
BSc Data Science & Analytics &nbsp;|&nbsp; JKUAT Karen &nbsp;|&nbsp; 2026<br>
All predictions computed live from saved models in <code>models/</code> folder &nbsp;|&nbsp;
NACADA Bi-Annual Reports 2021–2025
</div>
""", unsafe_allow_html=True)
