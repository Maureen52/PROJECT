# ============================================================
# app.py  —  Kenya Cannabis Seizure Forecast Tool
# ============================================================
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
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Kenya Cannabis Seizure Forecast',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F0F7F1; }

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
.kpi-sub { color: #555; font-size:0.76rem; margin:0; }

/* Info/danger boxes — dark text forced for visibility on light backgrounds */
.info {
    background: #E8F5E9; border-left: 4px solid #2D8B47;
    padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
    color: #1A3320 !important; font-size: 0.9rem;
}
.info * { color: #1A3320 !important; }

.danger {
    background: #FDECEA; border-left: 4px solid #C00000;
    padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
    color: #5C0000 !important; font-size: 0.9rem;
}
.danger * { color: #5C0000 !important; }

.warn {
    background: #FFF8E1; border-left: 4px solid #EF9F27;
    padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0;
    color: #4A3000 !important; font-size: 0.9rem;
}
.warn * { color: #4A3000 !important; }

.result {
    background: #E8F5E9; border: 2px solid #2D8B47;
    border-radius: 12px; padding: 1.2rem; margin: .8rem 0;
    color: #1A3320 !important;
}
.result * { color: #1A3320 !important; }

.sec-title {
    color: #1A5C2A; font-size: 1.1rem; font-weight: 700;
    border-bottom: 2px solid #2D8B47; padding-bottom: .3rem; margin: 1rem 0 .8rem;
}

.badge-h  { background:#C00000; color:#fff; border:2px solid #C00000;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-mh { background:#EF9F27; color:#fff; border:2px solid #EF9F27;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-m  { background:#2E5FA3; color:#fff; border:2px solid #2E5FA3;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-l  { background:#555; color:#fff; border:2px solid #555;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }

/* County detail card text — force dark on coloured backgrounds */
.county-card { color: #111 !important; }
.county-card * { color: #111 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────
GRN  = '#1A5C2A'; GRN2 = '#2D8B47'
AMB  = '#EF9F27'; BLU  = '#2E5FA3'; RED  = '#C00000'; GRAY = '#888780'
MODEL_COLOR = '#B5268B'   # E2 Ensemble only

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
    'High'    :{'icon':'🔴','badge':'badge-h','color':'#ffffff','text_color':'#5C0000',
                'bg':'#C00000','share':'63.5%',
                'action':'Immediate targeted enforcement surge required.',
                'desc':'Critical — highest enforcement priority.'},
    'Med-High':{'icon':'🟠','badge':'badge-mh','color':'#ffffff','text_color':'#4A3000',
                'bg':'#EF9F27','share':'21.7%',
                'action':'Build capacity before escalation to High tier.',
                'desc':'Emerging threat — pre-emptive intervention needed.'},
    'Medium'  :{'icon':'🔵','badge':'badge-m','color':'#ffffff','text_color':'#0D2340',
                'bg':'#2E5FA3','share':'13.8%',
                'action':'Maintain standard enforcement; monitor trends.',
                'desc':'Moderate activity — regular monitoring required.'},
    'Low'     :{'icon':'⚪','badge':'badge-l','color':'#ffffff','text_color':'#222',
                'bg':'#555555','share':'1.0%',
                'action':'Review enforcement capacity.',
                'desc':'Low recorded seizures — may reflect capacity gaps.'},
}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

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
    ens      = _load('ensemble_weights.pkl')
    return {'series': series, 'xgb': xgb_data,
            'lstm_w': lstm_w, 'lstm_sc': lstm_sc, 'ensemble': ens}

try:
    MODELS = load_all_models()
    SERIES = MODELS['series']
except Exception as e:
    st.error(f"❌ Could not load models from `models/` folder.\n\nError: {e}")
    st.stop()

HIST_LABELS   = SERIES['hist_labels']
HIST_KG       = np.array(SERIES['hist_kg'])
HIST_LOG      = np.array(SERIES['hist_log'])
TRAIN_END     = SERIES['train_end']
FUTURE_LABELS = SERIES['future_labels']
N_HIST        = len(HIST_LABELS)

# ─────────────────────────────────────────────────────────────
# PREDICTION — E2 ENSEMBLE ONLY
# ─────────────────────────────────────────────────────────────
def _sig(x):  return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x): return np.tanh(np.clip(x, -15, 15))


def xgb_build_row(log_series, t_idx):
    s = np.array(log_series)
    lag1=s[-1]; lag2=s[-2]; lag3=s[-3]
    rm2=s[-2:].mean(); rm3=s[-3:].mean()
    ldiff=lag1-lag2; tup=1 if ldiff>0 else 0
    abvm=1 if lag1>s.mean() else 0
    year=2021+(t_idx//2); half=(t_idx%2)+1
    return np.array([[t_idx, year, half,
                      np.sin(2*np.pi*half/2), np.cos(2*np.pi*half/2),
                      lag1, lag2, lag3, rm2, rm3, ldiff, tup, abvm]])


def predict_ensemble(log_series, n_ahead=4):
    w_xgb  = MODELS['ensemble']['xgb_weight']
    w_lstm = MODELS['ensemble']['lstm_weight']
    gbr    = MODELS['xgb']['model']
    Wih    = MODELS['lstm_w']['Wih']; Whh = MODELS['lstm_w']['Whh']
    bh     = MODELS['lstm_w']['bh'];  Wy  = MODELS['lstm_w']['Wy']
    by     = MODELS['lstm_w']['by']
    H      = MODELS['lstm_sc']['H']
    sc_min = MODELS['lstm_sc']['sc_min']
    sc_rng = MODELS['lstm_sc']['sc_rng']

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
        t         = len(HIST_LOG) + i if len(log_series)==N_HIST else len(log_series)+i
        xgb_log   = float(gbr.predict(xgb_build_row(np.array(cur_log), t))[0])
        cur_sc    = (np.array(cur_log) - sc_min) / sc_rng
        lstm_log  = lstm_fwd(cur_sc[-2:]) * sc_rng + sc_min
        e2_log    = w_xgb * xgb_log + w_lstm * lstm_log
        preds.append(int(np.expm1(e2_log)))
        cur_log.append(e2_log)
    return preds


# ─────────────────────────────────────────────────────────────
# BUILD ACTIVE SERIES
# ─────────────────────────────────────────────────────────────
active_log    = HIST_LOG.copy()
active_labels = list(HIST_LABELS)
active_kg     = list(HIST_KG)
new_added     = False

# Sidebar — only county lookup remains
with st.sidebar:
    st.markdown("### 🔍 County Lookup")
    county_q = st.text_input('Type county name:', placeholder='e.g. Nairobi, Kisumu…')

n_active = len(active_labels)

# ─────────────────────────────────────────────────────────────
# RUN E2 ENSEMBLE LIVE
# ─────────────────────────────────────────────────────────────
with st.spinner('🌿 Running E2 Ensemble forecast…'):
    forecast_kg = predict_ensemble(active_log, n_ahead=4)

future_labels_show = FUTURE_LABELS.copy()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <h1>🌿 Kenya Cannabis Seizure Forecast Tool</h1>
  <p>
    E2 Ensemble Model (XGBoost 79.5% + LSTM 20.5%) &nbsp;|&nbsp;
    NACADA Bi-Annual Data 2021–2025 &nbsp;|&nbsp;
    4-Period Forecast: 2025 H2 – 2027 H1
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{MODEL_COLOR};'>{forecast_kg[0]:,} kg</div>
    <div class='kpi-lbl'>Forecast — {future_labels_show[0]}</div>
    <div class='kpi-sub'>E2 Ensemble</div>
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
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>6.91%</div>
    <div class='kpi-lbl'>Test MAPE — E2 Ensemble</div>
    <div class='kpi-sub'>Lowest of all 6 models evaluated</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>~12,600 kg</div>
    <div class='kpi-lbl'>Projected Stable Level</div>
    <div class='kpi-sub'>Plateau forecast through 2027 H1</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS — 3 only
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    '📈 Forecast Chart',
    '📋 Forecast Table',
    '🏘️ County Risk Zones',
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">📈 Historical Actuals + E2 Ensemble Forecast</div>',
                unsafe_allow_html=True)

    all_x_labels = active_labels + future_labels_show
    n_tot = len(all_x_labels)
    fig   = go.Figure()

    # Region shading
    fig.add_vrect(x0=0, x1=TRAIN_END,
                  fillcolor='rgba(26,92,42,.04)', line_width=0,
                  annotation_text='Training', annotation_font_size=10,
                  annotation_font_color='#1A5C2A',
                  annotation_position='top left')
    fig.add_vrect(x0=TRAIN_END, x1=N_HIST-.5,
                  fillcolor='rgba(239,159,39,.07)', line_width=0,
                  annotation_text='Test', annotation_font_size=10,
                  annotation_font_color='#7F4F00',
                  annotation_position='top left')
    fig.add_vrect(x0=n_active-.5, x1=n_tot-.5,
                  fillcolor='rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast →', annotation_font_size=10,
                  annotation_font_color='#1A5C2A',
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
    for i, (lbl, v) in enumerate(zip(active_labels, active_kg)):
        fig.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                           showarrow=False, yshift=18,
                           font=dict(size=9.5, color=GRN))

    # Confidence band
    fc_x = list(range(n_active, n_tot))
    r, g_c, b = int(MODEL_COLOR[1:3],16), int(MODEL_COLOR[3:5],16), int(MODEL_COLOR[5:],16)
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1],
        y=[v*1.12 for v in forecast_kg]+[v*0.88 for v in reversed(forecast_kg)],
        fill='toself', fillcolor=f'rgba({r},{g_c},{b},0.12)',
        line=dict(width=0), showlegend=True, hoverinfo='skip',
        name='±12% uncertainty band',
    ))

    # Connector
    fig.add_trace(go.Scatter(
        x=[n_active-1, n_active], y=[active_kg[-1], forecast_kg[0]],
        mode='lines', line=dict(color=MODEL_COLOR, width=2, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_x, y=forecast_kg,
        mode='lines+markers+text',
        name='E2 Ensemble forecast',
        line=dict(color=MODEL_COLOR, width=3),
        marker=dict(size=11, color=MODEL_COLOR, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=[f'<b>{v:,} kg</b>' for v in forecast_kg],
        textposition='top center', textfont=dict(size=10, color=MODEL_COLOR),
        hovertemplate='<b>E2 Ensemble</b><br>%{x}: <b>%{y:,.0f} kg</b><extra></extra>',
    ))

    fig.add_vline(x=n_active-.5,
                  line=dict(color=GRAY, dash='dot', width=1.5),
                  annotation_text='History | Forecast',
                  annotation_position='top',
                  annotation_font=dict(color='#333333', size=10))

    fig.update_layout(
        title=dict(
            text=('<b>National Cannabis Seizures — Kenya</b><br>'
                  f'<span style="font-size:13px;color:#555555;">'
                  f'Historical: {active_labels[0]}–{active_labels[-1]}  ·  '
                  f'E2 Ensemble Forecast: '
                  f'{future_labels_show[0]}–{future_labels_show[-1]}</span>'),
            x=0.01, font=dict(size=16, color=GRN)
        ),
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        xaxis=dict(tickmode='array', tickvals=list(range(n_tot)),
                   ticktext=all_x_labels, tickangle=-32,
                   title='Period', gridcolor='#E8F5E9',
                   tickfont=dict(color='#222222')),
        yaxis=dict(title='Cannabis seized (kg)', tickformat=',.0f',
                   gridcolor='#E8F5E9', tickfont=dict(color='#222222')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,.9)',
                    bordercolor=GRN2, borderwidth=1,
                    font=dict(color='#222222')),
        hovermode='x unified', height=500,
        margin=dict(t=90, b=65, l=75, r=20),
        font=dict(color='#222222'),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='info'>
    <strong>Shaded band</strong> = ±12% uncertainty range around each forecast point.
    The <strong>E2 Ensemble</strong> combines XGBoost (79.5% weight) and LSTM (20.5% weight),
    achieving the lowest Test MAPE of <strong>6.91%</strong> across all six models evaluated.
    All predictions are computed live from trained models stored in the
    <code>models/</code> folder.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📋 Historical Actuals & E2 Ensemble Forecasts (kg)</div>',
                unsafe_allow_html=True)

    rows = []
    for lbl, v in zip(active_labels, active_kg):
        rows.append({'Period': lbl,
                     'Actual (kg)': f'{v:,.1f}',
                     'E2 Ensemble Forecast (kg)': '—',
                     'Status': 'Observed'})
    for lbl, v in zip(future_labels_show, forecast_kg):
        rows.append({'Period': lbl,
                     'Actual (kg)': 'Not yet reported',
                     'E2 Ensemble Forecast (kg)': f'{v:,}',
                     'Status': 'Forecast'})

    df_show = pd.DataFrame(rows)

    # All rows get dark text on black-tinted background
    def style_table(row):
        if row['Status'] == 'Forecast':
            # Forecast rows: dark green background, white text
            return ['background-color:#1A5C2A; color:#ffffff; font-weight:bold'] * len(row)
        else:
            # Observed rows: very dark background, white text
            return ['background-color:#1a1a1a; color:#ffffff;'] * len(row)

    st.dataframe(df_show.style.apply(style_table, axis=1),
                 use_container_width=True, hide_index=True)

    csv = df_show.to_csv(index=False)
    st.download_button('⬇️ Download as CSV', data=csv,
                       file_name='seizure_forecast_E2_Ensemble.csv',
                       mime='text/csv')


# ══════════════════════════════════════════════════════════════
# TAB 3 — COUNTY RISK ZONES
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏘️ County Cluster Risk Zones</div>',
                unsafe_allow_html=True)

    county_search = st.text_input('Search county:', placeholder='e.g. Nairobi, Kisumu…',
                                  key='county_tab3')

    if county_search:
        q     = county_search.strip()
        match = None
        for nm in COUNTY_DATA:
            if nm.lower() == q.lower():
                match = nm; break
        if not match:
            cands = [nm for nm in COUNTY_DATA if q.lower() in nm.lower()]
            if len(cands) == 1:
                match = cands[0]
            elif cands:
                st.markdown(f"""<div class='warn'>
                Multiple matches found: <strong>{", ".join(cands)}</strong>. Please be more specific.
                </div>""", unsafe_allow_html=True)

        if match:
            d  = COUNTY_DATA[match]
            ci = CLUSTER_CFG[d['cluster']]
            c1, c2 = st.columns([1, 2])
            with c1:
                # Solid colour background with white text — always readable
                st.markdown(f"""
                <div style='background:{ci["bg"]};border:2px solid {ci["bg"]};
                     border-radius:14px;padding:1.5rem;text-align:center;'>
                  <div style='font-size:3rem;'>{ci["icon"]}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:#ffffff;'>
                    {match}</div>
                  <div style='margin:.5rem 0;'>
                    <span class='{ci["badge"]}'>{d["cluster"]} Tier</span></div>
                  <div style='color:#ffffff;font-weight:600;font-size:.9rem;margin-top:.5rem;'>
                    {ci["desc"]}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                # White card, all text explicitly #111
                st.markdown(f"""
                <div style='background:white;border:2px solid {ci["bg"]};
                     border-radius:12px;padding:1.2rem;'>
                <table style='width:100%;border-collapse:collapse;'>
                  <tr>
                    <td style='padding:.5rem;color:#111;width:42%;font-weight:700;'>Cluster Tier</td>
                    <td style='padding:.5rem;color:#111;font-weight:800;'>{d["cluster"]}</td>
                  </tr>
                  <tr style='background:#f5f5f5;'>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>National Share</td>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>{ci["share"]} of all seizures</td>
                  </tr>
                  <tr>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>Total (2021–2025)</td>
                    <td style='padding:.5rem;color:#111;'>{d["total_kg"]:,.1f} kg</td>
                  </tr>
                  <tr style='background:#f5f5f5;'>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>Avg per Period</td>
                    <td style='padding:.5rem;color:#111;'>{d["total_kg"]/9:,.1f} kg</td>
                  </tr>
                  <tr>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>Trend</td>
                    <td style='padding:.5rem;color:#111;'>{d["trend"]}</td>
                  </tr>
                  <tr style='background:#f5f5f5;'>
                    <td style='padding:.5rem;color:#111;font-weight:700;'>NACADA Action</td>
                    <td style='padding:.5rem;color:#111;font-weight:600;'>{ci["action"]}</td>
                  </tr>
                </table></div>""", unsafe_allow_html=True)

            others = [n for n, v in COUNTY_DATA.items()
                      if v['cluster'] == d['cluster'] and n != match]
            st.markdown(f"""
            <div class='info'>
            <strong>Other {d["cluster"]} tier counties:</strong><br>
            {", ".join(others)}
            </div>""", unsafe_allow_html=True)

        elif county_search and not match:
            st.markdown(f"""<div class='warn'>
            County <strong>"{county_search}"</strong> not found. Check spelling or browse the table below.
            </div>""", unsafe_allow_html=True)

    # High-tier alert
    high = [n for n, d in COUNTY_DATA.items() if d['cluster'] == 'High']
    st.markdown(f"""
    <div class='danger'>
    🔴 <strong>10 HIGH-TIER COUNTIES (63.5% of national seizures):</strong><br>
    {" · ".join(high)}<br><br>
    <strong>Key trafficking corridors:</strong> Tanzania border (Migori) ·
    Uganda border (Busia) · Indian Ocean coast (Kilifi) ·
    Northern Corridor / Moyale highway (Marsabit)
    </div>""", unsafe_allow_html=True)

    # Full county table — all rows dark
    st.markdown('<div class="sec-title">📋 All 49 Counties — Cluster Assignments</div>',
                unsafe_allow_html=True)

    tier_order = {'High': 0, 'Med-High': 1, 'Medium': 2, 'Low': 3}
    tbl_rows = sorted(
        [{'County': n, 'Cluster Tier': d['cluster'],
          'Total kg (2021–2025)': f"{d['total_kg']:,.1f}",
          'Avg per Period (kg)': f"{d['total_kg']/9:,.1f}",
          'Trend': d['trend']}
         for n, d in COUNTY_DATA.items()],
        key=lambda r: (tier_order[r['Cluster Tier']],
                       -float(r['Total kg (2021–2025)'].replace(',', '')))
    )
    df_ct = pd.DataFrame(tbl_rows)

    # Tier-matched background colours, white text throughout
    tier_style = {
        'High'    : 'background-color:#C00000; color:#ffffff;',
        'Med-High': 'background-color:#9B6800; color:#ffffff;',
        'Medium'  : 'background-color:#1D3F6E; color:#ffffff;',
        'Low'     : 'background-color:#333333; color:#ffffff;',
    }

    def style_county_table(row):
        style = tier_style.get(row['Cluster Tier'], 'background-color:#1a1a1a; color:#ffffff;')
        return [style] * len(row)

    st.dataframe(df_ct.style.apply(style_county_table, axis=1),
                 use_container_width=True, hide_index=True, height=440)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f"""
<div style='text-align:center;color:#555555;font-size:.82rem;padding:.4rem;'>
🌿 Kenya Cannabis Seizure Forecast Tool &nbsp;|&nbsp;
E2 Ensemble Model (XGBoost + LSTM) &nbsp;|&nbsp;
NACADA Bi-Annual Reports 2021–2025<br>
All predictions computed live from saved models in <code>models/</code> folder
</div>
""", unsafe_allow_html=True)
