"""
House Price Prediction — Streamlit demo
Self-contained: retrains model on startup so no .joblib files needed.
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
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
# TRAIN MODEL ON STARTUP (cached — only runs once per session)
# =========================================================
@st.cache_resource
def train_model():
    """
    Load data, clean, and train the XGBoost pipeline.
    @st.cache_resource means this only runs ONCE when the app starts,
    then the result is reused for every subsequent interaction.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    # ---- Clean ----
    df = pd.read_csv("housing_enriched.csv")
    df.columns = df.columns.str.strip()
    df = df[df['Y = Sold price'] != 'Contact agent'].copy()
    df['price'] = (
        df['Y = Sold price']
          .str.replace('$', '', regex=False)
          .str.replace(',', '', regex=False)
          .astype(float)
    )
    df = df.drop(columns=['Y = Sold price'])
    df['Last sold date'] = pd.to_datetime(df['Last sold date'])
    df['days_since_ref'] = (
        df['Last sold date'] - pd.Timestamp('2000-01-01')
    ).dt.days
    df = df.drop(columns=['URL', 'Address', 'Postcode', 'Last sold date'])
    df['log_price']     = np.log1p(df['price'])
    df['log_land_size'] = np.log1p(df['Land size'])
    df = df.drop(columns=['price', 'Land size'])
    df = pd.get_dummies(df, columns=['Suburb', 'Type'], drop_first=True, dtype=int)

    y = df['log_price']
    X = df.drop(columns=['log_price'])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Pipeline ----
    numeric_features = [
        'Latitude', 'Longitude',
        'Bedroom', 'Bathroom', 'Car Park space',
        'primary_school', 'secondary_school', 'preschool',
        'train_station', 'distance_to_cbd',
        'days_since_ref', 'log_land_size'
    ]
    dummy_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('dum', 'passthrough', dummy_features),
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('model', XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42,
            verbosity=0
        ))
    ])
    pipeline.fit(X_train, y_train)

    metadata = {
        'numeric_features': numeric_features,
        'dummy_features':   dummy_features,
        'suburbs':  ['Oakleigh', 'Reservoir', 'Thornbury'],
        'types':    ['apartment', 'house', 'townhouse', 'unit', 'unitblock', 'villa'],
        'reference_date': '2000-01-01',
        'defaults': {
            'train_station': int(X_train['train_station'].median()),
        }
    }
    return pipeline, metadata

# Show a spinner while training on first load
with st.spinner("Loading model... (first load takes ~15 seconds)"):
    pipeline, meta = train_model()

# =========================================================
# SUBURB DEFAULTS
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

    # ---- Build input row ----
    row = {col: 0 for col in meta['numeric_features'] + meta['dummy_features']}

    row['Latitude']        = latitude
    row['Longitude']       = longitude
    row['Bedroom']         = bedrooms
    row['Bathroom']        = bathrooms
    row['Car Park space']  = car_parks
    row['train_station']   = train_stations
    row['distance_to_cbd'] = distance_to_cbd
    row['days_since_ref']  = days_since_ref
    row['log_land_size']   = log_land_size

    # Suburb dummy (Oakleigh = baseline, both dummies = 0)
    if suburb == 'Reservoir':
        row['Suburb_Reservoir'] = 1
    elif suburb == 'Thornbury':
        row['Suburb_Thornbury'] = 1

    # Type dummy (apartment = baseline, all type dummies = 0)
    type_col = f'Type_{prop_type}'
    if type_col in row:
        row[type_col] = 1

    # ---- Predict ----
    input_df       = pd.DataFrame([row])[meta['numeric_features'] + meta['dummy_features']]
    log_price_pred = pipeline.predict(input_df)[0]
    price_pred     = np.expm1(log_price_pred)

    # ---- Display ----
    st.success(f"### Predicted price: **${price_pred:,.0f}**")

    rmse_log = 0.15
    low  = np.expm1(log_price_pred - rmse_log)
    high = np.expm1(log_price_pred + rmse_log)
    st.caption(
        f"Typical 68% confidence range: **${low:,.0f} – ${high:,.0f}**  \n"
        f"_Based on model RMSE ≈ {rmse_log:.2f} log-price units (~{round((np.exp(rmse_log)-1)*100)}%)._"
    )

    with st.expander("See model inputs"):
        st.dataframe(input_df.T.rename(columns={0: 'value'}))

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Model: XGBoost | R² ≈ 0.83 | Median error ≈ 8.7% | "
    "Trained on ~1,800 Melbourne property sales (Oakleigh, Reservoir, Thornbury)."
)
