import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page Configuration
st.set_page_config(page_title='Kenya Cannabis Seizure Analytics', layout='wide')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Exploratory Data Analysis", "Forecasting Models", "Spatial Hotspots"])

# Load Data functions
@st.cache_data
def load_national_data():
    return pd.read_csv('national_features.csv')

@st.cache_data
def load_county_data():
    return pd.read_csv('county_features.csv')

@st.cache_data
def load_cluster_data():
    return pd.read_csv('county_clusters.csv')

@st.cache_data
def load_forecast_data():
    return pd.read_csv('forecast_all_models.csv')

# --- Page: Overview ---
if page == "Overview":
    st.title("Kenya Cannabis Seizure Dashboard (2021-2027)")
    st.markdown("""This application presents an end-to-end analysis of cannabis seizure data in Kenya, from initial cleaning to future forecasting using Machine Learning and Deep Learning ensembles.""")
    
    nf = load_national_data()
    total_seized = nf['kg_seized'].sum()
    st.metric("Grand Total Seized (2021-2025 H1)", f"{total_seized:,.2f} kg")
    
    fig = px.line(nf, x='period_label', y='kg_seized', title='National Seizure Trend',
                 labels={'kg_seized': 'Quantity (kg)', 'period_label': 'Half-Year Period'})
    st.plotly_chart(fig, use_container_width=True)

# --- Page: EDA ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    nf = load_national_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Log-Transformed Distribution")
        fig_hist = px.histogram(nf, x='log_kg_seized', nbins=10, title="National Log Seizures")
        st.plotly_chart(fig_hist)
        
    with col2:
        st.subheader("Period-on-Period Change")
        fig_pct = px.bar(nf, x='period_label', y='pct_change', title="Growth Rate (%)")
        st.plotly_chart(fig_pct)

# --- Page: Forecasting ---
elif page == "Forecasting Models":
    st.title("Time-Series Forecasting Leaderboard")
    df_fc = load_forecast_data()
    
    model_choice = st.selectbox("Select Model to View Forecast", df_fc['Entry'].unique())
    
    st.subheader(f"Results for {model_choice}")
    selected_row = df_fc[df_fc['Entry'] == model_choice]
    st.table(selected_row)
    
    # Plotting Forecast vs Actuals
    nf = load_national_data()
    actuals = nf['kg_seized'].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nf['period_label'], y=actuals, name='Actual', line=dict(color='black', width=3)))
    
    # Extract numeric forecast values from the row
    fc_cols = ['2024 H2', '2025 H1', '2025 H2', '2026 H1', '2026 H2', '2027 H1']
    y_fc = selected_row[fc_cols].values.flatten()
    
    fig.add_trace(go.Scatter(x=fc_cols, y=y_fc, name=f'Forecast ({model_choice})', 
                             line=dict(dash='dash', width=2)))
    
    st.plotly_chart(fig, use_container_width=True)

# --- Page: Spatial ---
elif page == "Spatial Hotspots":
    st.title("Spatial Hotspot Analysis (K-means)")
    clusters = load_cluster_data()
    
    fig_cluster = px.bar(clusters, x='county', y='total_kg', color='cluster', 
                         title="Counties Grouped by Seizure Tiers",
                         labels={'total_kg': 'Cumulative kg (2021-2025)'})
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.subheader("Cluster Assignments")
    st.dataframe(clusters[['county', 'cluster', 'total_kg', 'trend_direction']])
