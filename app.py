# ============================================================
# app.py  —  Kenya Cannabis Seizure Forecast Tool
# William Maureen Ndinda | SCT213-C002-0048/2022 | JKUAT Karen
# ============================================================
#
# HOW TO RUN ON YOUR MACHINE:
#   pip install streamlit plotly scikit-learn numpy pandas
#   streamlit run app.py
#
# WHAT THIS APP DOES:
#   1. Shows the full historical seizure series (2021–2025)
#   2. Lets user input the latest NACADA observed value
#   3. Forecasts the next 4 periods using XGBoost, LSTM, or E2 Ensemble
#   4. Displays forecast values and confidence range
#   5. Shows county High-tier risk zone
#   6. User can switch between all five models + ensemble
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
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
# CSS — GREEN PRIMARY THEME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F0F7F1; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A5C2A 0%, #2D8B47 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label { color: #C8F0D0 !important; }

.header {
    background: linear-gradient(135deg, #1A5C2A 0%, #2D8B47 60%, #52A96A 100%);
    padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 18px rgba(26,92,42,0.25);
}
.header h1 { color: white !important; margin: 0; font-size: 1.85rem; font-weight: 800; }
.header p  { color: #C8F0D0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

.kpi {
    background: white; border: 2px solid #2D8B47;
    border-radius: 12px; padding: 1.1rem 0.8rem;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.kpi-val  { color: #1A5C2A; font-size: 2rem; font-weight: 800; margin: 0; }
.kpi-lbl  { color: #2D8B47; font-size: 0.82rem; font-weight: 600; margin: 0.2rem 0 0; }
.kpi-sub  { color: #888; font-size: 0.76rem; margin: 0; }

.info  { background:#E8F5E9; border-left:4px solid #2D8B47; color:#000;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.warn  { background:#FFF8E1; border-left:4px solid #EF9F27;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.danger{ background:#FDECEA; border-left:4px solid #C00000;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.result{ background:#E8F5E9; border:2px solid #2D8B47; border-radius:12px;
         padding:1.2rem; margin:.8rem 0; }

.sec-title {
    color:#1A5C2A; font-size:1.1rem; font-weight:700;
    border-bottom:2px solid #2D8B47; padding-bottom:.3rem; margin:1rem 0 .8rem;
}

div.stButton > button {
    background:#2D8B47 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:.55rem 1.8rem !important; width:100%;
}
div.stButton > button:hover { background:#1A5C2A !important; }

/* Tabs: uppercase labels and black font */
div[role="tablist"] > button, div[role="tab"] button {
    text-transform:uppercase !important;
    color:#000000 !important;
    font-weight:700 !important;
}

table.dark-tbl, table.dark-tbl td, table.dark-tbl th {
    background:#2b2b2b !important;
    color:#ffffff !important;
@@ -431,10 +438,6 @@
    st.markdown("<hr style='border-color:rgba(255,255,255,.3);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.8rem;color:#C8F0D0;'>
    <b>Student:</b> William Maureen Ndinda<br>
    <b>Reg:</b> SCT213-C002-0048/2022<br>
    <b>Programme:</b> BSc Data Science<br>
    <b>JKUAT Karen | 2026</b><br><br>
    <b>Data:</b> NACADA 2021–2025<br>
    <b>Models:</b> XGBoost · LSTM · E2 Ensemble
    </div>
@@ -479,414 +482,413 @@
    Model: <b>{selected_model}</b> &nbsp;|&nbsp;
    NACADA Data 2021–2025 &nbsp;|&nbsp;
    4-Period Forecast (2025 H2 – 2027 H1) &nbsp;|&nbsp;
    BSc Data Science, JKUAT Karen
  </p>
</div>
""", unsafe_allow_html=True)

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
    </div>""", unsafe_allow_html=True)

with c2:
    last = active_kg[-1]
    chg  = (forecast_kg[0]-last)/last*100
    arr  = '↑' if chg>0 else '↓'
    col  = RED if chg>10 else (GRN if chg<-5 else AMB)
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{col};'>{arr} {abs(chg):.1f}%</div>
    <div class='kpi-lbl'>Change vs Last Period</div>
    <div class='kpi-sub'>Last: {last:,.0f} kg ({active_labels[-1]})</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>{met.get('mape','—')}%</div>
    <div class='kpi-lbl'>Model Test MAPE</div>
    <div class='kpi-sub'>Rank #{met.get('rank','—')} of 6 models</div>
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
    </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📈 Forecast Chart',
    '📋 Forecast Table',
    '🏘️ County Risk Map',
    '📊 Model Comparison',
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">📈 Full Historical Series + 4-Period Forecast</div>',
                unsafe_allow_html=True)

    mc = MODEL_COLORS[selected_model]
    all_x_labels = active_labels + future_labels_used
    n_tot        = len(all_x_labels)

    fig = go.Figure()

    # Region shading
    fig.add_vrect(x0=0, x1=7,
                  fillcolor='rgba(26,92,42,.04)', line_width=0,
                  annotation_text='Training', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=7, x1=N_HIST-.5,
                  fillcolor='rgba(239,159,39,.07)', line_width=0,
                  annotation_text='Test', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=n_active-.5, x1=n_tot-.5,
                  fillcolor=f'rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast', annotation_font_size=10,
                  annotation_position='top right')

    # Historical actuals — thick green line
    fig.add_trace(go.Scatter(
        x=list(range(n_active)), y=active_kg,
        mode='lines+markers',
        name='Actual seizures',
        line=dict(color=GRN, width=3.5),
        marker=dict(size=10, color=GRN, line=dict(width=2, color='white')),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
        text=active_labels,
    ))

    # Value labels on actual points
    for i, (lbl, v) in enumerate(zip(active_labels, active_kg)):
        fig.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                           showarrow=False, yshift=16,
                           font=dict(size=9.5, color=GRN))

    # Connector from last actual to first forecast
    fig.add_trace(go.Scatter(
        x=[n_active-1, n_active], y=[active_kg[-1], forecast_kg[0]],
        mode='lines', line=dict(color=mc, width=2.2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Forecast line + confidence band
    fc_x = list(range(n_active, n_tot))
    fig.add_trace(go.Scatter(
        x=fc_x+fc_x[::-1],
        y=[v*1.12 for v in forecast_kg]+[v*0.88 for v in reversed(forecast_kg)],
        fill='toself',
        fillcolor=f'rgba({int(mc[1:3],16)},{int(mc[3:5],16)},{int(mc[5:],16)},0.13)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
        name='95% confidence band',
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=forecast_kg,
        mode='lines+markers+text',
        name=f'{selected_model} forecast',
        line=dict(color=mc, width=3),
        marker=dict(size=11, color=mc, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=[f'<b>{v:,}</b>' for v in forecast_kg],
        textposition='top center',
        textfont=dict(size=10, color=mc),
        hovertemplate=(f'<b>{selected_model}</b><br>'
                       '%{text} — %{x}<extra></extra>'),
    ))

    # Vertical separator
    fig.add_vline(x=n_active-.5,
                  line=dict(color=GRAY, dash='dot', width=1.5),
                  annotation_text='← History | Forecast →',
                  annotation_position='top', annotation_font=dict(color=GRAY, size=10))

    fig.update_layout(
        title=dict(
            text=(f'<b>National Cannabis Seizures — Kenya</b><br>'
                  f'<span style="font-size:13px;color:{GRAY};">'
                  f'Historical 2021–{active_labels[-1]}  ·  '
                  f'{selected_model} Forecast 2025 H2–2027 H1</span>'),
            x=0.01, font=dict(size=16, color=GRN)
        ),
        plot_bgcolor='white', paper_bgcolor='#F0F7F1',
        xaxis=dict(tickmode='array', tickvals=list(range(n_tot)),
                   ticktext=all_x_labels, tickangle=-32,
                   title='Period', gridcolor='#E8F5E9',
                   showline=True, linecolor=GRN2),
        yaxis=dict(title='Cannabis seized (kg)', tickformat=',.0f',
                   gridcolor='#E8F5E9', showline=True, linecolor=GRN2),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,.85)',
                    bordercolor=GRN2, borderwidth=1),
        hovermode='x unified', height=510,
        margin=dict(t=90, b=65, l=75, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Confidence note
    st.markdown(f"""<div class='info'>
    <b>Shaded band</b> = ±12% uncertainty range around the {selected_model} forecast.
    The actual confidence interval is derived from the model's training residuals.
    <b>E2 Ensemble</b> (XGBoost 79.5% + LSTM 20.5%) achieved the best test MAPE
    of <b>6.91%</b> — the lowest of all models evaluated.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📋 Complete Forecast Values (kg)</div>',
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

    df_show = pd.DataFrame(rows)

    def hl(row):
        # Dark theme for all rows
        return ['background-color:#2b2b2b; color:#ffffff']*len(row)

    st.dataframe(df_show.style.apply(hl, axis=1),
                 use_container_width=True, hide_index=True)

    # Download
    csv = df_show.to_csv(index=False)
    st.download_button(
        label='⬇️ Download as CSV',
        data=csv,
        file_name=f'kenya_seizure_forecast_{selected_model.replace(" ","_")}.csv',
        mime='text/csv'
    )

    # All-model future forecasts
    st.markdown('<div class="sec-title">📊 All Models — Future Forecast Comparison (kg)</div>',
                unsafe_allow_html=True)

    cmp_rows = []
    for m, d in STATIC_FC.items():
        row = {'Model':m, 'Type':'⭐ Ensemble' if 'Ensemble' in m else 'Individual'}
        for lbl, v in zip(future_labels_used, d['future']):
            row[lbl] = f'{v:,}'
        row['Test MAPE'] = f"{MODEL_METRICS.get(m.replace(' (Recommended)',''),{}).get('mape','—')}%"
        cmp_rows.append(row)

    df_cmp = pd.DataFrame(cmp_rows)

    def hl2(row):
        # Dark theme for all rows
        return ['background-color:#2b2b2b; color:#ffffff']*len(row)

    st.dataframe(df_cmp.style.apply(hl2, axis=1),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — COUNTY RISK MAP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏘️ County Cluster Risk Zones</div>',
                unsafe_allow_html=True)

    # County lookup
    if county_query:
        q = county_query.strip()
        match = None
        for nm in COUNTY_DATA:
            if nm.lower() == q.lower(): match=nm; break
        if not match:
            cands = [nm for nm in COUNTY_DATA if q.lower() in nm.lower()]
            if len(cands)==1: match=cands[0]
            elif cands:
                st.warning(f'Multiple matches: {", ".join(cands)}. Be more specific.')

        if match:
            d  = COUNTY_DATA[match]
            ci = CLUSTER_CFG[d['cluster']]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div style='background:{ci["bg"]};border:2px solid {ci["color"]};
                     border-radius:14px;padding:1.5rem;text-align:center;'>
                  <div style='font-size:3rem;'>{ci["icon"]}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ci["color"]};'>
                    {match}</div>
                  <div style='margin:.5rem 0;'>
                    <span class='{ci["badge"]}'>{d["cluster"]} Tier</span></div>
                  <div style='color:{ci["color"]};font-weight:600;font-size:.9rem;'>
                    {ci["desc"]}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style='background:white;border:1px solid #ddd;
                     border-radius:12px;padding:1.2rem;'>
                <table class='dark-tbl' style='width:100%;border-collapse:collapse;'>
                  <tr><td style='padding:.5rem;width:42%;'><b>Cluster Tier</b></td>
                      <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>
                        {d["cluster"]}</td></tr>
                  <tr>
                    <td style='padding:.5rem;'><b>National Share</b></td>
                    <td style='padding:.5rem;font-weight:700;color:{ci["color"]};'>
                      {ci["share"]} of all seizures</td></tr>
                  <tr><td style='padding:.5rem;'><b>Total (2021–2025)</b></td>
                      <td style='padding:.5rem;'>{d["total_kg"]:,.1f} kg</td></tr>
                  <tr>
                    <td style='padding:.5rem;'><b>Avg per Period</b></td>
                    <td style='padding:.5rem;'>{d["total_kg"]/9:,.1f} kg</td></tr>
                  <tr><td style='padding:.5rem;'><b>Trend</b></td>
                      <td style='padding:.5rem;'>{d["trend"]}</td></tr>
                  <tr>
                    <td style='padding:.5rem;'><b>NACADA Action</b></td>
                    <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>
                      {ci["action"]}</td></tr>
                </table></div>""", unsafe_allow_html=True)

            others = [n for n,v in COUNTY_DATA.items()
                      if v['cluster']==d['cluster'] and n!=match]
            st.markdown(f"""<div class='info'>
            <b>Other counties in the {d["cluster"]} tier:</b><br>
            {", ".join(others)}
            </div>""", unsafe_allow_html=True)

        elif county_query and not match:
            st.markdown(f"""<div class='warn'>
            County <b>"{county_query}"</b> not found. Check spelling or use the table below.
            </div>""", unsafe_allow_html=True)

    # High-tier alert box
    high_counties = [n for n,d in COUNTY_DATA.items() if d['cluster']=='High']
    st.markdown(f"""<div class='danger'>
    🔴 <b>10 HIGH-TIER COUNTIES (63.5% of national seizures):</b><br>
    {" · ".join(high_counties)}<br><br>
    <b>Trafficking corridors:</b> Tanzania border (Migori) ·
    Uganda border (Busia) · Indian Ocean coast (Kilifi) ·
    Northern Corridor/Moyale highway (Marsabit)
    </div>""", unsafe_allow_html=True)

    # Full county table
    st.markdown('<div class="sec-title">📋 All 49 Counties — Cluster Assignments</div>',
                unsafe_allow_html=True)

    tier_order = {'High':0,'Med-High':1,'Medium':2,'Low':3}
    tbl_rows   = sorted(
        [{'County':n, 'Cluster Tier':d['cluster'],
          'Total kg (2021–2025)':f"{d['total_kg']:,.1f}",
          'Avg per Period (kg)':f"{d['total_kg']/9:,.1f}",
          'Trend':d['trend']}
         for n,d in COUNTY_DATA.items()],
        key=lambda r:(tier_order[r['Cluster Tier']], -float(r['Total kg (2021–2025)'].replace(',','')))
    )
    df_ct = pd.DataFrame(tbl_rows)

    tier_bg = {'High':'#FDECEA','Med-High':'#FFF3CD',
               'Medium':'#D6E4F7','Low':'#F2F2F2'}

    def hl_tier(row):
        # Dark theme for all rows
        return ['background-color:#2b2b2b; color:#ffffff']*len(row)

    st.dataframe(df_ct.style.apply(hl_tier, axis=1),
                 use_container_width=True, hide_index=True, height=440)


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">📊 Model Performance Comparison</div>',
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
        # Dark theme for all rows
        return ['background-color:#2b2b2b; color:#ffffff']*len(row)

    st.dataframe(df_cmp2.style.apply(hl3, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown(f"""<div class='result' style='color:#000000;'>
    <b>🌿 Why E2 Ensemble is recommended:</b><br>
    The E2 Ensemble combines XGBoost (weight 79.5%) and LSTM (weight 20.5%) using
    inverse cross-validation weighting. It achieves the best overall Test MAPE of
    <b>6.91%</b> — beating XGBoost alone (7.08%) — and forecasts a stable plateau
    of approximately <b>12,600–12,800 kg per half-year</b> through 2027 H1.<br><br>
    <b>Model selection principle:</b> Cross-validation RMSE is the primary criterion
    because the holdout test set contains only 2 data points — too few for reliable
    ranking by test RMSE alone.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f"""
<div style='text-align:center;color:{GRAY};font-size:.82rem;padding:.4rem;'>
🌿 Kenya Cannabis Seizure Forecast Tool &nbsp;|&nbsp;
William Maureen Ndinda (SCT213-C002-0048/2022) &nbsp;|&nbsp;
BSc Data Science &amp; Analytics &nbsp;|&nbsp; JKUAT Karen &nbsp;|&nbsp; 2026<br>
Data: NACADA Bi-Annual Reports 2021–2025 &nbsp;|&nbsp;
Models: E2 Ensemble · XGBoost · LSTM · SARIMA · ARIMA · Prophet
</div>
""", unsafe_allow_html=True)
