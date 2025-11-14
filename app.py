"""
Aplicación Streamlit para Predicción de Precios de Alquiler en AMBA
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Alquiler - AMBA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏠 Predictor de Precios de Alquiler en AMBA")
st.markdown("---")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning entrenado con datos de propiedades 
en el Área Metropolitana de Buenos Aires (AMBA) para predecir el precio de alquiler mensual.
""")

# Cargar modelo y metadatos (con caching)
@st.cache_data
def load_model():
    """Carga el modelo entrenado y sus metadatos"""
    model_path = 'models/rental_price_model.pkl'
    metadata_path = 'models/model_metadata.json'
    preprocessing_path = 'models/preprocessing_info.json'
    
    if not os.path.exists(model_path):
        st.error("❌ Modelo no encontrado. Por favor, ejecuta primero el notebook de modelado.")
        st.stop()
    
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open(preprocessing_path, 'r', encoding='utf-8') as f:
        preprocessing_info = json.load(f)
    
    return model, metadata, preprocessing_info

# Cargar datos para obtener valores únicos de categorías
@st.cache_data
def load_data_for_categories():
    """Carga datos para obtener valores únicos de categorías"""
    try:
        df = pd.read_csv('output/alquiler_AMBA_clean.csv')
        return df
    except:
        return None

# Cargar modelo y datos
try:
    model, metadata, preprocessing_info = load_model()
    df_data = load_data_for_categories()
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Sidebar con información del modelo
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.write(f"**Modelo:** {metadata['model_name']}")
    st.write(f"**Tipo:** {metadata['model_type']}")
    st.write(f"**Fecha de entrenamiento:** {metadata['date_trained']}")
    st.write(f"**Muestras de entrenamiento:** {metadata['training_samples']:,}")
    
    st.markdown("---")
    st.header("📊 Métricas del Modelo")
    val_metrics = metadata['metrics']['validation']
    st.write(f"**RMSE:** ${val_metrics['RMSE']:,.2f}")
    st.write(f"**MAE:** ${val_metrics['MAE']:,.2f}")
    st.write(f"**R²:** {val_metrics['R²']:.4f}")
    st.write(f"**MAPE:** {val_metrics['MAPE']:.2f}%")
    
    st.markdown("---")
    st.markdown("**Desarrollado por:**")
    st.markdown("- Ignacio Bruzone")
    st.markdown("- Felix Lopez Menardi")
    st.markdown("- Christian Ijjas")

# Formulario principal
st.header("📝 Ingrese las características de la propiedad")

# Dividir en columnas para mejor organización
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Físicas")
    
    # Superficie total
    superficie = st.number_input(
        "Superficie Total (m²)",
        min_value=10.0,
        max_value=1000.0,
        value=50.0,
        step=5.0,
        help="Superficie total de la propiedad en metros cuadrados"
    )
    
    # Dormitorios
    dormitorios = st.number_input(
        "Dormitorios",
        min_value=0,
        max_value=8,
        value=1,
        step=1,
        help="Número de dormitorios"
    )
    
    # Baños
    banos = st.number_input(
        "Baños",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
        help="Número de baños"
    )
    
    # Ambientes
    ambientes = st.number_input(
        "Ambientes",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Número total de ambientes"
    )
    
    # Antigüedad
    antiguedad = st.number_input(
        "Antigüedad (años)",
        min_value=0,
        max_value=200,
        value=10,
        step=1,
        help="Antigüedad de la propiedad en años"
    )
    
    # Cocheras
    cocheras = st.number_input(
        "Cocheras",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Número de cocheras"
    )

with col2:
    st.subheader("Ubicación")
    
    # Obtener valores únicos para dropdowns
    if df_data is not None:
        ciudades = sorted(df_data['ITE_ADD_CITY_NAME'].unique().tolist())
        provincias = sorted(df_data['ITE_ADD_STATE_NAME'].unique().tolist())
        barrios = sorted(df_data['ITE_ADD_NEIGHBORHOOD_NAME'].unique().tolist())
    else:
        # Valores por defecto si no se pueden cargar
        ciudades = ['Capital Federal', 'Escobar', 'La Matanza', 'Lanús', 'Morón', 'Quilmes', 'San Fernando', 'San Isidro', 'Tigre']
        provincias = ['Capital Federal', 'Bs.As. G.B.A. Norte', 'Bs.As. G.B.A. Oeste', 'Bs.As. G.B.A. Sur']
        barrios = ['Palermo', 'Belgrano', 'Villa Crespo', 'Puerto Madero', 'Recoleta']
    
    ciudad = st.selectbox(
        "Ciudad",
        ciudades,
        index=0 if 'Capital Federal' in ciudades else 0
    )
    
    provincia = st.selectbox(
        "Provincia",
        provincias,
        index=0 if 'Capital Federal' in provincias else 0
    )
    
    barrio = st.selectbox(
        "Barrio",
        barrios,
        index=0
    )
    
    # Coordenadas (opcionales)
    st.markdown("**Coordenadas (opcional)**")
    use_coords = st.checkbox("Usar coordenadas específicas", value=False)
    
    if use_coords:
        longitude = st.number_input(
            "Longitud",
            min_value=-60.0,
            max_value=-58.0,
            value=-58.4375,
            step=0.0001,
            format="%.4f"
        )
        latitude = st.number_input(
            "Latitud",
            min_value=-35.0,
            max_value=-34.0,
            value=-34.6037,
            step=0.0001,
            format="%.4f"
        )
    else:
        # Usar coordenadas promedio del barrio si está disponible
        if df_data is not None:
            barrio_data = df_data[df_data['ITE_ADD_NEIGHBORHOOD_NAME'] == barrio]
            if len(barrio_data) > 0:
                longitude = barrio_data['LONGITUDE'].mean()
                latitude = barrio_data['LATITUDE'].mean()
            else:
                longitude = -58.4375
                latitude = -34.6037
        else:
            longitude = -58.4375
            latitude = -34.6037

# Amenities
st.subheader("🏢 Amenities y Servicios")
amenities_col1, amenities_col2, amenities_col3 = st.columns(3)

with amenities_col1:
    amoblado = st.checkbox("Amoblado", value=False)
    acceso_internet = st.checkbox("Acceso a Internet", value=False)
    gimnasio = st.checkbox("Gimnasio", value=False)
    laundry = st.checkbox("Laundry", value=False)

with amenities_col2:
    calefaccion = st.checkbox("Calefacción", value=False)
    salon_usos_multiples = st.checkbox("Salón de Usos Múltiples", value=False)
    aire_ac = st.checkbox("Aire Acondicionado", value=False)
    jacuzzi = st.checkbox("Jacuzzi", value=False)

with amenities_col3:
    ascensor = st.checkbox("Ascensor", value=False)
    seguridad = st.checkbox("Seguridad", value=False)
    pileta = st.checkbox("Pileta", value=False)
    area_parrillas = st.checkbox("Área de Parrillas", value=False)

# Información temporal
st.subheader("📅 Información Temporal")
temp_col1, temp_col2 = st.columns(2)

with temp_col1:
    year = st.selectbox(
        "Año",
        [2021, 2022, 2023, 2024],
        index=2
    )

with temp_col2:
    mes = st.selectbox(
        "Mes",
        list(range(1, 13)),
        index=11,
        format_func=lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1]
    )

# Determinar estación basada en el mes
def get_season(month):
    if month in [12, 1, 2]:
        return 'verano'
    elif month in [3, 4, 5]:
        return 'otoño'
    elif month in [6, 7, 8]:
        return 'invierno'
    else:
        return 'primavera'

estacion = get_season(mes)

# Botón de predicción
st.markdown("---")
predict_button = st.button("🔮 Predecir Precio de Alquiler", type="primary", use_container_width=True)

if predict_button:
    # Validaciones
    if dormitorios > ambientes:
        st.error("⚠️ El número de dormitorios no puede ser mayor que el número de ambientes.")
        st.stop()
    
    if superficie <= 0:
        st.error("⚠️ La superficie debe ser mayor a 0.")
        st.stop()
    
    # Preparar datos para predicción
    try:
        # Crear DataFrame con los datos ingresados
        input_data = {
            'STotalM2': superficie,
            'Dormitorios': dormitorios,
            'Banos': banos,
            'Ambientes': ambientes,
            'Antiguedad': antiguedad,
            'Cocheras': cocheras,
            'LONGITUDE': longitude,
            'LATITUDE': latitude,
            'year': year,
            'mes': mes,
            'Amoblado': 1 if amoblado else 0,
            'AccesoInternet': 1 if acceso_internet else 0,
            'Gimnasio': 1 if gimnasio else 0,
            'Laundry': 1 if laundry else 0,
            'Calefaccion': 1 if calefaccion else 0,
            'SalonDeUsosMul': 1 if salon_usos_multiples else 0,
            'AireAC': 1 if aire_ac else 0,
            'Jacuzzi': 1 if jacuzzi else 0,
            'Ascensor': 1 if ascensor else 0,
            'Seguridad': 1 if seguridad else 0,
            'Pileta': 1 if pileta else 0,
            'AreaParrillas': 1 if area_parrillas else 0,
            'ITE_ADD_CITY_NAME': ciudad,
            'ITE_ADD_STATE_NAME': provincia,
            'ITE_ADD_NEIGHBORHOOD_NAME': barrio,
            'ITE_TIPO_PROD': 'U',  # Asumimos tipo U (Unidad) por defecto
            'estacion': estacion
        }
        
        # Calcular features derivadas
        densidad_ambientes = ambientes / superficie if superficie > 0 else 0
        total_amenities = sum([
            amoblado, acceso_internet, gimnasio, laundry, calefaccion,
            salon_usos_multiples, aire_ac, jacuzzi, ascensor, seguridad,
            pileta, area_parrillas
        ])
        tiene_cochera = 1 if cocheras > 0 else 0
        
        input_data['densidad_ambientes'] = densidad_ambientes
        input_data['total_amenities'] = total_amenities
        input_data['tiene_cochera'] = tiene_cochera
        
        # Convertir a DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Aplicar One-Hot Encoding como en el entrenamiento
        # Obtener todas las columnas categóricas codificadas
        categorical_cols = ['ITE_ADD_CITY_NAME', 'ITE_ADD_STATE_NAME', 
                          'ITE_ADD_NEIGHBORHOOD_NAME', 'ITE_TIPO_PROD', 'estacion']
        
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
        
        # Obtener feature names del modelo
        feature_names = preprocessing_info['feature_names']
        
        # Asegurar que todas las columnas necesarias estén presentes
        for feature in feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Reordenar columnas para que coincidan con el orden del entrenamiento
        df_encoded = df_encoded[feature_names]
        
        # Hacer predicción
        prediction = model.predict(df_encoded)[0]
        
        # Mostrar resultado
        st.markdown("---")
        st.success("✅ Predicción completada")
        
        # Resultado principal
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            st.markdown(f"### 💰 Precio Predicho de Alquiler Mensual")
            st.markdown(f"# ${prediction:,.2f}")
            st.caption("Precio en pesos constantes")
        
        with col_result2:
            precio_por_m2 = prediction / superficie if superficie > 0 else 0
            st.metric("Precio por m²", f"${precio_por_m2:,.2f}")
        
        # Información adicional
        with st.expander("📊 Información Adicional"):
            st.write(f"**Superficie:** {superficie} m²")
            st.write(f"**Precio por m²:** ${precio_por_m2:,.2f}")
            st.write(f"**Ubicación:** {barrio}, {ciudad}, {provincia}")
            st.write(f"**Total de amenities:** {total_amenities}")
            st.write(f"**Densidad de ambientes:** {densidad_ambientes:.4f}")
        
        # Visualización (opcional)
        st.markdown("---")
        st.subheader("📈 Análisis Comparativo")
        
        # Comparar con propiedades similares si tenemos datos
        if df_data is not None:
            # Filtrar propiedades similares
            similar_props = df_data[
                (df_data['STotalM2'] >= superficie * 0.8) & 
                (df_data['STotalM2'] <= superficie * 1.2) &
                (df_data['Dormitorios'] == dormitorios) &
                (df_data['Banos'] == banos)
            ]
            
            if len(similar_props) > 0:
                precio_medio_similar = similar_props['precio_pesos_constantes'].mean()
                precio_mediano_similar = similar_props['precio_pesos_constantes'].median()
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.metric("Predicción del Modelo", f"${prediction:,.0f}")
                
                with col_comp2:
                    st.metric("Precio Medio Similar", f"${precio_medio_similar:,.0f}")
                
                with col_comp3:
                    st.metric("Precio Mediano Similar", f"${precio_mediano_similar:,.0f}")
                
                # Gráfico de comparación
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(['Predicción', 'Media Similar', 'Mediana Similar'], 
                       [prediction, precio_medio_similar, precio_mediano_similar],
                       color=['steelblue', 'coral', 'lightgreen'])
                ax.set_xlabel('Precio (pesos)', fontweight='bold')
                ax.set_title('Comparación con Propiedades Similares', fontweight='bold')
                ax.grid(alpha=0.3, axis='x')
                st.pyplot(fig)
            else:
                st.info("No se encontraron propiedades similares para comparar.")
        
    except Exception as e:
        st.error(f"❌ Error al realizar la predicción: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Predictor de Precios de Alquiler en AMBA | Desarrollado con Streamlit</p>
</div>
""", unsafe_allow_html=True)

