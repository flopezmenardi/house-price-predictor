"""
AMBA Rental Price Predictor
Professional Machine Learning Application for Real Estate Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AMBA Rental Price Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional, techy appearance
st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        animation: fadeInDown 0.8s ease-out;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Section Headers */
    .section-header {
        color: #3b82f6;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
        animation: scaleIn 0.5s ease-out;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        margin: 0;
    }
    
    .prediction-label {
        font-size: 1rem;
        color: #d1fae5;
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Checkboxes */
    .stCheckbox {
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        transform: scale(1.05);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(59, 130, 246, 0.5);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Progress and loading animations */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    p, label {
        color: #cbd5e1;
    }
    
    /* Divider */
    hr {
        border-color: rgba(59, 130, 246, 0.2);
        margin: 2rem 0;
    }
    
    /* Data Display */
    .stMetric {
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #3b82f6 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(59, 130, 246, 0.5);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main Header
st.markdown(
    """
<div class="main-header">
    <h1 class="main-title">AMBA Rental Price Predictor</h1>
    <p class="main-subtitle">Advanced Machine Learning Model for Metropolitan Buenos Aires Real Estate Valuation</p>
</div>
""",
    unsafe_allow_html=True,
)


# Load model and metadata (with caching)
@st.cache_data
def load_model():
    """Load trained model and metadata"""
    model_path = "models/rental_price_model.pkl"
    metadata_path = "models/model_metadata.json"
    preprocessing_path = "models/preprocessing_info.json"

    if not os.path.exists(model_path):
        st.error("Model not found. Please run the modeling notebook first.")
        st.stop()

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(preprocessing_path, "r", encoding="utf-8") as f:
        preprocessing_info = json.load(f)

    return model, metadata, preprocessing_info


# Load data for category values
@st.cache_data
def load_data_for_categories():
    """Load data to get unique categorical values"""
    try:
        df = pd.read_csv("output/alquiler_AMBA_clean.csv")
        return df
    except:
        return None


# Load model and data
try:
    model, metadata, preprocessing_info = load_model()
    df_data = load_data_for_categories()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Sidebar with model information
with st.sidebar:
    st.markdown(
        "<h3 style='color: #3b82f6;'>Model Information</h3>", unsafe_allow_html=True
    )
    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Type:** {metadata['model_type']}")
    st.write(f"**Trained:** {metadata['date_trained']}")
    st.write(f"**Training samples:** {metadata['training_samples']:,}")

    st.markdown("---")
    st.markdown(
        "<h3 style='color: #3b82f6;'>Performance Metrics</h3>", unsafe_allow_html=True
    )
    val_metrics = metadata["metrics"]["validation"]
    st.metric("RMSE", f"${val_metrics['RMSE']:,.2f}")
    st.metric("MAE", f"${val_metrics['MAE']:,.2f}")
    st.metric("R²", f"{val_metrics['R²']:.4f}")
    st.metric("MAPE", f"{val_metrics['MAPE']:.2f}%")

    st.markdown("---")
    st.markdown("**Developed by:**")
    st.markdown("• Ignacio Bruzone")
    st.markdown("• Felix Lopez Menardi")
    st.markdown("• Christian Ijjas")

# Main form
st.markdown(
    "<h2 class='section-header'>Property Characteristics</h2>", unsafe_allow_html=True
)

# Divide into columns for better organization
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Physical Characteristics**")

    superficie = st.number_input(
        "Total Area (m²)",
        min_value=10.0,
        max_value=1000.0,
        value=50.0,
        step=5.0,
        help="Total property area in square meters",
    )

    dormitorios = st.number_input("Bedrooms", min_value=0, max_value=8, value=1, step=1)

    banos = st.number_input("Bathrooms", min_value=1, max_value=6, value=1, step=1)

    ambientes = st.number_input("Rooms", min_value=1, max_value=10, value=2, step=1)

    antiguedad = st.number_input(
        "Age (years)", min_value=0, max_value=200, value=10, step=1
    )

    cocheras = st.number_input(
        "Parking Spaces", min_value=0, max_value=10, value=0, step=1
    )

with col2:
    st.markdown("**Location**")

    if df_data is not None:
        ciudades = sorted(df_data["ITE_ADD_CITY_NAME"].unique().tolist())
        provincias = sorted(df_data["ITE_ADD_STATE_NAME"].unique().tolist())
        barrios = sorted(df_data["ITE_ADD_NEIGHBORHOOD_NAME"].unique().tolist())
    else:
        ciudades = ["Capital Federal", "Escobar", "La Matanza", "Tigre"]
        provincias = ["Capital Federal", "Bs.As. G.B.A. Norte"]
        barrios = ["Palermo", "Belgrano", "Recoleta"]

    ciudad = st.selectbox(
        "City", ciudades, index=0 if "Capital Federal" in ciudades else 0
    )

    provincia = st.selectbox(
        "State", provincias, index=0 if "Capital Federal" in provincias else 0
    )

    barrio = st.selectbox("Neighborhood", barrios, index=0)

    st.markdown("**Coordinates (optional)**")
    use_coords = st.checkbox("Use specific coordinates", value=False)

    if use_coords:
        longitude = st.number_input(
            "Longitude",
            min_value=-60.0,
            max_value=-58.0,
            value=-58.4375,
            step=0.0001,
            format="%.4f",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-35.0,
            max_value=-34.0,
            value=-34.6037,
            step=0.0001,
            format="%.4f",
        )
    else:
        if df_data is not None:
            barrio_data = df_data[df_data["ITE_ADD_NEIGHBORHOOD_NAME"] == barrio]
            if len(barrio_data) > 0:
                longitude = barrio_data["LONGITUDE"].mean()
                latitude = barrio_data["LATITUDE"].mean()
            else:
                longitude = -58.4375
                latitude = -34.6037
        else:
            longitude = -58.4375
            latitude = -34.6037

# Amenities
st.markdown(
    "<h2 class='section-header'>Amenities & Services</h2>", unsafe_allow_html=True
)
amenities_col1, amenities_col2, amenities_col3 = st.columns(3)

with amenities_col1:
    amoblado = st.checkbox("Furnished", value=False)
    acceso_internet = st.checkbox("Internet Access", value=False)
    gimnasio = st.checkbox("Gym", value=False)
    laundry = st.checkbox("Laundry", value=False)

with amenities_col2:
    calefaccion = st.checkbox("Heating", value=False)
    salon_usos_multiples = st.checkbox("Multipurpose Room", value=False)
    aire_ac = st.checkbox("Air Conditioning", value=False)
    jacuzzi = st.checkbox("Jacuzzi", value=False)

with amenities_col3:
    ascensor = st.checkbox("Elevator", value=False)
    seguridad = st.checkbox("Security", value=False)
    pileta = st.checkbox("Pool", value=False)
    area_parrillas = st.checkbox("BBQ Area", value=False)

# Temporal information
st.markdown("<h2 class='section-header'>Time Period</h2>", unsafe_allow_html=True)
temp_col1, temp_col2 = st.columns(2)

with temp_col1:
    year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025], index=3)

with temp_col2:
    mes = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=11,
        format_func=lambda x: [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][x - 1],
    )


# Determine season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return "verano"
    elif month in [3, 4, 5]:
        return "otoño"
    elif month in [6, 7, 8]:
        return "invierno"
    else:
        return "primavera"


estacion = get_season(mes)

# Prediction button
st.markdown("---")
predict_button = st.button(
    "Generate Prediction", type="primary", use_container_width=True
)

if predict_button:
    # Validations
    if dormitorios > ambientes:
        st.error("Number of bedrooms cannot exceed total rooms.")
        st.stop()

    if superficie <= 0:
        st.error("Area must be greater than 0.")
        st.stop()

    # Prepare data for prediction
    try:
        with st.spinner("Processing prediction..."):
            # Create DataFrame with input data
            input_data = {
                "STotalM2": superficie,
                "Dormitorios": dormitorios,
                "Banos": banos,
                "Ambientes": ambientes,
                "Antiguedad": antiguedad,
                "Cocheras": cocheras,
                "LONGITUDE": longitude,
                "LATITUDE": latitude,
                "year": year,
                "mes": mes,
                "Amoblado": 1 if amoblado else 0,
                "AccesoInternet": 1 if acceso_internet else 0,
                "Gimnasio": 1 if gimnasio else 0,
                "Laundry": 1 if laundry else 0,
                "Calefaccion": 1 if calefaccion else 0,
                "SalonDeUsosMul": 1 if salon_usos_multiples else 0,
                "AireAC": 1 if aire_ac else 0,
                "Jacuzzi": 1 if jacuzzi else 0,
                "Ascensor": 1 if ascensor else 0,
                "Seguridad": 1 if seguridad else 0,
                "Pileta": 1 if pileta else 0,
                "AreaParrillas": 1 if area_parrillas else 0,
                "ITE_ADD_CITY_NAME": ciudad,
                "ITE_ADD_STATE_NAME": provincia,
                "ITE_ADD_NEIGHBORHOOD_NAME": barrio,
                "ITE_TIPO_PROD": "U",
                "estacion": estacion,
            }

            # Calculate derived features
            densidad_ambientes = ambientes / superficie if superficie > 0 else 0
            total_amenities = sum(
                [
                    amoblado,
                    acceso_internet,
                    gimnasio,
                    laundry,
                    calefaccion,
                    salon_usos_multiples,
                    aire_ac,
                    jacuzzi,
                    ascensor,
                    seguridad,
                    pileta,
                    area_parrillas,
                ]
            )
            tiene_cochera = 1 if cocheras > 0 else 0

            input_data["densidad_ambientes"] = densidad_ambientes
            input_data["total_amenities"] = total_amenities
            input_data["tiene_cochera"] = tiene_cochera

            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])

            # Apply One-Hot Encoding
            categorical_cols = [
                "ITE_ADD_CITY_NAME",
                "ITE_ADD_STATE_NAME",
                "ITE_ADD_NEIGHBORHOOD_NAME",
                "ITE_TIPO_PROD",
                "estacion",
            ]

            df_encoded = pd.get_dummies(
                df_input,
                columns=categorical_cols,
                prefix=categorical_cols,
                drop_first=False,
            )

            # Get feature names from model
            feature_names = preprocessing_info["feature_names"]

            # Ensure all necessary columns are present
            missing_features = [f for f in feature_names if f not in df_encoded.columns]
            if missing_features:
                df_missing = pd.DataFrame(0, index=df_encoded.index, columns=missing_features)
                df_encoded = pd.concat([df_encoded, df_missing], axis=1)

            # Reorder columns to match training
            df_encoded = df_encoded[feature_names]

            # Make prediction
            prediction = model.predict(df_encoded)[0]

            # Show result
            st.markdown("---")
            st.success("Prediction completed successfully")

            # Main result
            st.markdown(
                f"""
            <div class="prediction-result">
                <p class="prediction-value">${prediction:,.2f}</p>
                <p class="prediction-label">Monthly Rental Price (Constant Pesos)</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Additional metrics
            col_result1, col_result2, col_result3 = st.columns(3)

            precio_por_m2 = prediction / superficie if superficie > 0 else 0

            with col_result1:
                st.metric("Price per m²", f"${precio_por_m2:,.2f}")

            with col_result2:
                st.metric("Total Area", f"{superficie} m²")

            with col_result3:
                st.metric("Total Amenities", f"{total_amenities}")

            # Additional information
            with st.expander("Detailed Information"):
                st.write(f"**Location:** {barrio}, {ciudad}, {provincia}")
                st.write(f"**Coordinates:** ({latitude:.4f}, {longitude:.4f})")
                st.write(f"**Room Density:** {densidad_ambientes:.4f}")
                st.write(f"**Season:** {estacion.capitalize()}")
                st.write(
                    f"**Period:** {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][mes - 1]} {year}"
                )

            # Comparative analysis
            st.markdown(
                "<h2 class='section-header'>Comparative Analysis</h2>",
                unsafe_allow_html=True,
            )

            if df_data is not None:
                # Filtrar propiedades similares
                similar_props = df_data[
                    (df_data['STotalM2'] >= superficie * 0.8) &
                    (df_data['STotalM2'] <= superficie * 1.2) &
                    (df_data['Dormitorios'] == dormitorios) &
                    (df_data['Banos'] == banos) &
                    (df_data['ITE_ADD_CITY_NAME'] == ciudad) &
                    (df_data['ITE_ADD_NEIGHBORHOOD_NAME'] == barrio)
                ]

                if len(similar_props) > 0:
                    precio_medio_similar = similar_props[
                        "precio_pesos_constantes"
                    ].mean()
                    precio_mediano_similar = similar_props[
                        "precio_pesos_constantes"
                    ].median()

                    col_comp1, col_comp2, col_comp3 = st.columns(3)

                    with col_comp1:
                        st.metric("Model Prediction", f"${prediction:,.0f}")

                    with col_comp2:
                        diff_mean = (
                            (prediction - precio_medio_similar)
                            / precio_medio_similar
                            * 100
                        )
                        st.metric(
                            "Similar Properties (Mean)",
                            f"${precio_medio_similar:,.0f}",
                            delta=f"{diff_mean:+.1f}%",
                        )

                    with col_comp3:
                        diff_median = (
                            (prediction - precio_mediano_similar)
                            / precio_mediano_similar
                            * 100
                        )
                        st.metric(
                            "Similar Properties (Median)",
                            f"${precio_mediano_similar:,.0f}",
                            delta=f"{diff_median:+.1f}%",
                        )

                    st.caption(
                        f"Comparison based on {len(similar_props)} similar properties"
                    )
                else:
                    st.info("No similar properties found for comparison.")

    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p>AMBA Rental Price Predictor | Machine Learning for Real Estate</p>
    <p style='font-size: 0.85rem; margin-top: 0.5rem;'>Powered by Random Forest Regression</p>
</div>
""",
    unsafe_allow_html=True,
)
