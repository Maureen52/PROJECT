# ============================================================
# app.py  —  Kenya Cannabis Seizure Forecast Tool
# William Maureen Ndinda | SCT213-C002-0048/2022 | JKUAT Karen
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

.info  { background:#E8F5E9; border-left:4px solid #2D8B47;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }
.danger{ background:#FDECEA; border-left:4px solid #C00000;
         padding:.8rem 1rem; border-radius:0 8px 8px 0; margin:.6rem 0; }

.sec-title {
    color:#1A5C2A; font-size:1.1rem; font-weight:700;
    border-bottom:2px solid #2D8B47; padding-bottom:.3rem; margin:1rem 0 .8rem;
}

.badge-h  { background:#FDECEA; color:#C00000;  border:2px solid #C00000;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-mh { background:#FFF3CD; color:#7F4F00;  border:2px solid #EF9F27;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-m  { background:#D6E4F7; color:#1F3864;  border:2px solid #2E5FA3;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
.badge-l  { background:#F2F2F2; color:#444;     border:2px solid #888;
            padding:.35rem 1rem; border-radius:20px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
GRN  = '#1A5C2A'; GRN2 = '#2D8B47'; AMB = '#EF9F27'
BLU  = '#2E5FA3'; RED  = '#C00000'; GRAY = '#888780'
MODEL_COLOR = '#B5268B'   # E2 Ensemble colour

HIST_LABELS = ['2021 H1','2021 H2','2022 H1','2022 H2','2023 H1',
               '2023 H2','2024 H1','2024 H2','2025 H1']
HIST_KG     = np.array([6932.09, 4781.42, 3621.41, 7249.73, 11866.28,
                         13714.12, 13415.20, 14737.70, 12752.52])
HIST_LOG    = np.log1p(HIST_KG)

NEXT_PERIODS = ['2025 H2','2026 H1','2026 H2','2027 H1']
N_HIST       = len(HIST_LABELS)

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
    'High'    :{'icon':'🔴','badge':'badge-h', 'color':RED, 'bg':'#FDECEA',
                'share':'63.5%','action':'Immediate targeted enforcement surge required.',
                'desc':'Critical — highest enforcement priority.'},
    'Med-High':{'icon':'🟠','badge':'badge-mh','color':'#7F4F00','bg':'#FFF3CD',
                'share':'21.7%','action':'Build capacity before escalation to High tier.',
                'desc':'Emerging threat — pre-emptive intervention needed.'},
    'Medium'  :{'icon':'🔵','badge':'badge-m', 'color':BLU, 'bg':'#D6E4F7',
                'share':'13.8%','action':'Maintain standard enforcement; monitor trends.',
                'desc':'Moderate activity — regular monitoring required.'},
    'Low'     :{'icon':'⚪','badge':'badge-l', 'color':'#444','bg':'#F2F2F2',
                'share':'1.0%','action':'Review enforcement capacity.',
                'desc':'Low recorded seizures — may reflect capacity gaps.'},
}

# E2 Ensemble static forecasts (XGBoost 79.5% + LSTM 20.5%)
FORECAST_VALUES = [12790, 12680, 12627, 12607]   # kg per future period

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <h1>🌿 Kenya Cannabis Seizure Forecast Tool</h1>
  <p>
    E2 Ensemble Model (XGBoost + LSTM) &nbsp;|&nbsp;
    NACADA Data 2021–2025 &nbsp;|&nbsp;
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
    <div class='kpi-val' style='color:{MODEL_COLOR};'>{FORECAST_VALUES[0]:,} kg</div>
    <div class='kpi-lbl'>Next Period Forecast</div>
    <div class='kpi-sub'>2025 H2 · E2 Ensemble</div>
    </div>""", unsafe_allow_html=True)

with c2:
    last = HIST_KG[-1]
    chg  = (FORECAST_VALUES[0] - last) / last * 100
    arr  = '↑' if chg > 0 else '↓'
    col  = RED if chg > 10 else (GRN if chg < -5 else AMB)
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val' style='color:{col};'>{arr} {abs(chg):.1f}%</div>
    <div class='kpi-lbl'>Change vs Last Period</div>
    <div class='kpi-sub'>Last observed: {last:,.0f} kg (2025 H1)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class='kpi'>
    <div class='kpi-val'>6.91%</div>
    <div class='kpi-lbl'>Model Test MAPE</div>
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
# TABS
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
    st.markdown('<div class="sec-title">📈 Historical Seizures + 4-Period E2 Ensemble Forecast</div>',
                unsafe_allow_html=True)

    all_x_labels = list(HIST_LABELS) + NEXT_PERIODS
    n_tot        = len(all_x_labels)
    forecast_kg  = FORECAST_VALUES

    fig = go.Figure()

    # Region shading
    fig.add_vrect(x0=0, x1=6.5,
                  fillcolor='rgba(26,92,42,.04)', line_width=0,
                  annotation_text='Training', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=6.5, x1=N_HIST - .5,
                  fillcolor='rgba(239,159,39,.07)', line_width=0,
                  annotation_text='Test', annotation_font_size=10,
                  annotation_position='top left')
    fig.add_vrect(x0=N_HIST - .5, x1=n_tot - .5,
                  fillcolor='rgba(45,139,71,.06)', line_width=0,
                  annotation_text='Forecast', annotation_font_size=10,
                  annotation_position='top right')

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=list(range(N_HIST)), y=list(HIST_KG),
        mode='lines+markers',
        name='Actual seizures',
        line=dict(color=GRN, width=3.5),
        marker=dict(size=10, color=GRN, line=dict(width=2, color='white')),
        hovertemplate='<b>%{text}</b><br>Actual: <b>%{y:,.0f} kg</b><extra></extra>',
        text=HIST_LABELS,
    ))

    for i, (lbl, v) in enumerate(zip(HIST_LABELS, HIST_KG)):
        fig.add_annotation(x=i, y=v, text=f'<b>{v:,.0f}</b>',
                           showarrow=False, yshift=16,
                           font=dict(size=9.5, color=GRN))

    # Connector
    fig.add_trace(go.Scatter(
        x=[N_HIST - 1, N_HIST], y=[HIST_KG[-1], forecast_kg[0]],
        mode='lines', line=dict(color=MODEL_COLOR, width=2.2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Confidence band
    fc_x = list(range(N_HIST, n_tot))
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1],
        y=[v * 1.12 for v in forecast_kg] + [v * 0.88 for v in reversed(forecast_kg)],
        fill='toself',
        fillcolor='rgba(181,38,139,0.13)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
        name='±12% uncertainty band',
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_x, y=forecast_kg,
        mode='lines+markers+text',
        name='E2 Ensemble forecast',
        line=dict(color=MODEL_COLOR, width=3),
        marker=dict(size=11, color=MODEL_COLOR, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=[f'<b>{v:,}</b>' for v in forecast_kg],
        textposition='top center',
        textfont=dict(size=10, color=MODEL_COLOR),
        hovertemplate='<b>E2 Ensemble</b><br>%{text} kg — %{x}<extra></extra>',
    ))

    fig.add_vline(x=N_HIST - .5,
                  line=dict(color=GRAY, dash='dot', width=1.5),
                  annotation_text='← History | Forecast →',
                  annotation_position='top',
                  annotation_font=dict(color=GRAY, size=10))

    fig.update_layout(
        title=dict(
            text=('<b>National Cannabis Seizures — Kenya</b><br>'
                  f'<span style="font-size:13px;color:{GRAY};">'
                  'Historical 2021–2025 H1  ·  E2 Ensemble Forecast 2025 H2–2027 H1</span>'),
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

    st.markdown("""<div class='info'>
    <b>Shaded band</b> = ±12% uncertainty range. The <b>E2 Ensemble</b> combines
    XGBoost (79.5% weight) and LSTM (20.5% weight) and achieved the lowest Test MAPE
    of <b>6.91%</b> across all six models evaluated. The forecast indicates a
    <b>stable plateau of approximately 12,600–12,800 kg per half-year</b> through 2027 H1.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📋 Historical & Forecast Values (kg)</div>',
                unsafe_allow_html=True)

    rows = []
    for lbl, val in zip(HIST_LABELS, HIST_KG):
        rows.append({'Period': lbl,
                     'Actual (kg)': f'{val:,.1f}',
                     'Forecast (kg)': '—',
                     'Status': '✅ Observed'})

    for lbl, val in zip(NEXT_PERIODS, FORECAST_VALUES):
        rows.append({'Period': lbl,
                     'Actual (kg)': 'Not yet reported',
                     'Forecast (kg)': f'{val:,}',
                     'Status': '🔮 E2 Ensemble Forecast'})

    df_show = pd.DataFrame(rows)

    def hl(row):
        if 'Forecast' in str(row['Status']):
            return ['background-color:#E8F5E9; font-weight:bold'] * len(row)
        return [''] * len(row)

    st.dataframe(df_show.style.apply(hl, axis=1),
                 use_container_width=True, hide_index=True)

    csv = df_show.to_csv(index=False)
    st.download_button(
        label='⬇️ Download as CSV',
        data=csv,
        file_name='kenya_seizure_forecast_E2_Ensemble.csv',
        mime='text/csv'
    )


# ══════════════════════════════════════════════════════════════
# TAB 3 — COUNTY RISK ZONES
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏘️ County Cluster Risk Zones</div>',
                unsafe_allow_html=True)

    county_query = st.text_input('Search county:', placeholder='e.g. Nairobi, Kisumu…')

    if county_query:
        q     = county_query.strip()
        match = None
        for nm in COUNTY_DATA:
            if nm.lower() == q.lower():
                match = nm; break
        if not match:
            cands = [nm for nm in COUNTY_DATA if q.lower() in nm.lower()]
            if len(cands) == 1:
                match = cands[0]
            elif cands:
                st.warning(f'Multiple matches: {", ".join(cands)}. Please be more specific.')

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
                <table style='width:100%;border-collapse:collapse;'>
                  <tr><td style='padding:.5rem;color:#666;width:42%;'><b>Cluster Tier</b></td>
                      <td style='padding:.5rem;color:{ci["color"]};font-weight:700;'>
                        {d["cluster"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>National Share</b></td>
                    <td style='padding:.5rem;font-weight:700;color:{ci["color"]};'>
                      {ci["share"]} of all seizures</td></tr>
                  <tr><td style='padding:.5rem;color:#666;'><b>Total (2021–2025)</b></td>
                      <td style='padding:.5rem;'>{d["total_kg"]:,.1f} kg</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>Avg per Period</b></td>
                    <td style='padding:.5rem;'>{d["total_kg"]/9:,.1f} kg</td></tr>
                  <tr><td style='padding:.5rem;color:#666;'><b>Trend</b></td>
                      <td style='padding:.5rem;'>{d["trend"]}</td></tr>
                  <tr style='background:#f9f9f9;'>
                    <td style='padding:.5rem;color:#666;'><b>NACADA Action</b></td>
                    <td style='padding:.5rem;font-weight:600;color:{ci["color"]};'>
                      {ci["action"]}</td></tr>
                </table></div>""", unsafe_allow_html=True)

            others = [n for n, v in COUNTY_DATA.items()
                      if v['cluster'] == d['cluster'] and n != match]
            st.markdown(f"""<div class='info'>
            <b>Other counties in the {d["cluster"]} tier:</b><br>
            {", ".join(others)}
            </div>""", unsafe_allow_html=True)

        elif not match:
            st.markdown(f"""<div style='background:#FFF8E1;border-left:4px solid #EF9F27;
            padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;'>
            County <b>"{county_query}"</b> not found. Check spelling or browse the table below.
            </div>""", unsafe_allow_html=True)

    # High-tier alert
    high_counties = [n for n, d in COUNTY_DATA.items() if d['cluster'] == 'High']
    st.markdown(f"""<div class='danger'>
    🔴 <b>10 HIGH-TIER COUNTIES (63.5% of national seizures):</b><br>
    {" · ".join(high_counties)}<br><br>
    <b>Key trafficking corridors:</b> Tanzania border (Migori) ·
    Uganda border (Busia) · Indian Ocean coast (Kilifi) ·
    Northern Corridor / Moyale highway (Marsabit)
    </div>""", unsafe_allow_html=True)

    # Full county table
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

    tier_bg = {'High': '#FDECEA', 'Med-High': '#FFF3CD',
               'Medium': '#D6E4F7', 'Low': '#F2F2F2'}

    def hl_tier(row):
        bg = tier_bg.get(row['Cluster Tier'], '')
        return [f'background-color:{bg}'] * len(row)

    st.dataframe(df_ct.style.apply(hl_tier, axis=1),
                 use_container_width=True, hide_index=True, height=440)


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
Model: E2 Ensemble (XGBoost 79.5% + LSTM 20.5%)
</div>
""", unsafe_allow_html=True)
