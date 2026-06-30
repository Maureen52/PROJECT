This is my final year BSC PROJECT, Where I modeled data to make prections and created am app

What this app does:

An interactive web application that forecasts national cannabis seizure
quantities in Kenya using three machine learning models:

- E2 Ensemble (XGBoost 79.5% + LSTM 20.5%) — best MAPE 6.91% ✅ Recommended
- XGBoost — best individual model RMSE (1,046 kg)
- LSTM— most consistent generaliser (overfit ratio 1.04×)

Features
1. Shows full historical NACADA seizure series (2021–2025)
2. Produces 4-period forecast (2025 H2 – 2027 H1)
3. Lets you enter a new NACADA observation to update forecasts in real time
4. Switch between all 6 models (ARIMA, SARIMA, Prophet, LSTM, XGBoost, E2 Ensemble)
5. County cluster lookup — type any county name to see its risk tier
6. High-tier county risk zone display (10 counties = 63.5% of national seizures)
7. Download forecast table as CSV

DASHOARD LINK
https://ml-for-time-series-data-rkgdtynglkenyrlappq6ntu.streamlit.app/
