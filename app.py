"""
House Price Prediction — Streamlit demo
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# =========================================================
# LOAD MODEL (cached so it loads once, not on every interaction)
# =========================================================
@st.cache_resource
def load_model():
    pipeline = joblib.load('house_price_model.joblib')
    metadata = joblib.load('model_metadata.joblib')
    return pipeline, metadata

pipeline, meta = load_model()

# =========================================================
# SUBURB DEFAULTS — lat/long + CBD distance per suburb
# =========================================================
SUBURB_DEFAULTS = {
    'Oakleigh':  {'lat': -37.899, 'lon': 145.094, 'cbd_m': 14600},
    'Reservoir': {'lat': -37.718, 'lon': 145.007, 'cbd_m': 12700},
    'Thornbury': {'lat': -37.759, 'lon': 145.001, 'cbd_m':  7000},
}

# =========================================================
# HEADER
# =========================================================
st.title("🏠 Melbourne House Price Predictor")
st.markdown(
    "Predicts sale price for properties in **Oakleigh, Reservoir, and Thornbury** "
    "based on property attributes. Trained on ~1,800 historical sales."
)
st.divider()

# =========================================================
# INPUT WIDGETS
# =========================================================
st.subheader("Property details")

col1, col2 = st.columns(2)

with col1:
    suburb    = st.selectbox("Suburb", meta['suburbs'])
    prop_type = st.selectbox("Property type", meta['types'])
    bedrooms  = st.number_input("Bedrooms",       min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms",       min_value=1, max_value=8,  value=2, step=1)
    car_parks = st.number_input("Car park spaces", min_value=0, max_value=10, value=2, step=1)

with col2:
    land_size = st.number_input(
        "Land size (m²)", min_value=50, max_value=5000, value=500, step=10
    )
    distance_to_cbd = st.number_input(
        "Distance to CBD (m)",
        min_value=5000, max_value=20000,
        value=SUBURB_DEFAULTS[suburb]['cbd_m'],
        step=500,
        help="Auto-filled from suburb centre. Adjust for your specific street."
    )
    train_stations = st.number_input(
        "Train stations within ~1km",
        min_value=0, max_value=10,
        value=meta['defaults']['train_station'],
        step=1
    )
    sale_date = st.date_input(
        "Intended sale date",
        value=date.today(),
        min_value=date(2024, 1, 1),
        max_value=date(2030, 12, 31)
    )

# =========================================================
# PREDICT BUTTON
# =========================================================
st.divider()

if st.button("Predict price", type="primary", use_container_width=True):

    # ---- Derived features ----
    log_land_size  = np.log1p(land_size)
    ref            = pd.Timestamp(meta['reference_date'])
    days_since_ref = (pd.Timestamp(sale_date) - ref).days
    latitude       = SUBURB_DEFAULTS[suburb]['lat']
    longitude      = SUBURB_DEFAULTS[suburb]['lon']

    # ---- Build input row (all columns default to 0, then fill in) ----
    row = {col: 0 for col in meta['numeric_features'] + meta['dummy_features']}

    # Numeric features
    row['Latitude']         = latitude
    row['Longitude']        = longitude
    row['Bedroom']          = bedrooms
    row['Bathroom']         = bathrooms
    row['Car Park space']   = car_parks
    row['train_station']    = train_stations
    row['distance_to_cbd']  = distance_to_cbd
    row['days_since_ref']   = days_since_ref
    row['log_land_size']    = log_land_size

    # Suburb dummy (Oakleigh is the baseline — both dummies = 0)
    if suburb == 'Reservoir':
        row['Suburb_Reservoir'] = 1
    elif suburb == 'Thornbury':
        row['Suburb_Thornbury'] = 1

    # Type dummy (apartment is the baseline — all type dummies = 0)
    type_col = f'Type_{prop_type}'
    if type_col in row:
        row[type_col] = 1

    # ---- Predict ----
    input_df      = pd.DataFrame([row])[meta['numeric_features'] + meta['dummy_features']]
    log_price_pred = pipeline.predict(input_df)[0]
    price_pred     = np.expm1(log_price_pred)

    # ---- Display result ----
    st.success(f"### Predicted price: **${price_pred:,.0f}**")

    # Confidence range — ±1 RMSE in log space (model RMSE ≈ 0.15 for XGBoost)
    rmse_log = 0.15
    low  = np.expm1(log_price_pred - rmse_log)
    high = np.expm1(log_price_pred + rmse_log)
    st.caption(
        f"Typical 68% confidence range: **${low:,.0f} – ${high:,.0f}**  \n"
        f"_Based on the model's RMSE of ~{rmse_log:.2f} log-price units (≈{round((np.exp(rmse_log)-1)*100)}%)._"
    )

    # Show the inputs used (useful for debugging and transparency)
    with st.expander("See model inputs"):
        st.dataframe(input_df.T.rename(columns={0: 'value'}))

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Trained on Melbourne property data (~1,800 sales across Oakleigh, Reservoir, Thornbury). "
    "Model: XGBoost with one-hot encoded categoricals, log-transformed price and land size. "
    "R² ≈ 0.83 | Median error ≈ 8.7%."
)