# 🏠 Predictor de Precios de Alquiler en AMBA

Sistema de predicción de precios de alquiler mensual para propiedades en el Área Metropolitana de Buenos Aires (AMBA) utilizando Machine Learning.

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para predecir precios de alquiler de propiedades en el AMBA. El sistema incluye:

- **Análisis exploratorio de datos** (`analysis-pipeline.ipynb`)
- **Pipeline de limpieza y transformación** (`transformation-pipeline.ipynb`)
- **Pipeline de modelado y entrenamiento** (`modeling-pipeline.ipynb`)
- **Aplicación web interactiva** (`app.py`) con Streamlit

## 👥 Integrantes

- Ignacio Bruzone
- Felix Lopez Menardi
- Christian Ijjas

## 📁 Estructura del Proyecto

```
house-price-predictor/
├── data/
│   ├── alquiler_AMBA_dev.csv          # Dataset de desarrollo
│   ├── alquiler_AMBA_test.csv          # Dataset de prueba
│   └── mapa_amba.html                  # Mapa del AMBA
├── output/
│   └── alquiler_AMBA_clean.csv         # Datos limpios y transformados
├── models/                             # Modelos entrenados
│   ├── rental_price_model.pkl          # Modelo final
│   ├── model_metadata.json             # Metadatos del modelo
│   └── preprocessing_info.json         # Información de preprocesamiento
├── notebooks/
│   ├── analysis-pipeline.ipynb          # Análisis exploratorio
│   ├── transformation-pipeline.ipynb    # Limpieza y transformación
│   ├── modeling-pipeline.ipynb          # Entrenamiento completo (exploratorio)
│   └── train-model.ipynb                # Entrenamiento rápido del modelo final
├── app.py                              # Aplicación Streamlit
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Uso del Proyecto

### 1. Análisis Exploratorio

Ejecutar el notebook `analysis-pipeline.ipynb` para explorar los datos originales.

### 2. Limpieza y Transformación

Ejecutar el notebook `transformation-pipeline.ipynb` para:
- Eliminar duplicados y outliers
- Separar propiedades de alquiler de ventas
- Limpiar y transformar variables
- Crear features derivadas
- Guardar datos limpios en `output/alquiler_AMBA_clean.csv`

### 3. Entrenamiento del Modelo

#### Opción Rápida (Recomendada para uso rápido)

Para obtener el modelo entrenado rápidamente y poner la aplicación en funcionamiento, ejecutar el notebook `train-model.ipynb`:
- Entrena directamente el modelo Random Forest con los mejores hiperparámetros encontrados
- Proceso optimizado sin exploración ni comparación de modelos
- Guarda el modelo listo para usar en `models/`
- **Tiempo estimado**: ~2-5 minutos

Luego puedes pasar directamente al paso 4 para ejecutar la aplicación.

#### Opción Completa (Análisis Exploratorio)

Ejecutar el notebook `modeling-pipeline.ipynb` para:
- Cargar datos limpios
- Aplicar feature engineering (encoding de variables categóricas)
- Dividir datos en train/validation
- Entrenar múltiples modelos (Linear Regression, Random Forest, XGBoost, Gradient Boosting)
- Evaluar y comparar modelos
- Optimizar hiperparámetros
- Guardar el mejor modelo en `models/`

**Nota:** Este proceso puede tardar 30 minutos o más dependiendo del hardware, ya que incluye la exploración completa de modelos y optimización de hiperparámetros.

### 4. Aplicación Web

Ejecutar la aplicación Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

#### Características de la App

- **Formulario interactivo** para ingresar características de la propiedad:
  - Características físicas (superficie, dormitorios, baños, etc.)
  - Ubicación (ciudad, provincia, barrio)
  - Amenities y servicios
  - Información temporal
  
- **Predicción en tiempo real** del precio de alquiler mensual

- **Visualizaciones comparativas** con propiedades similares

- **Información del modelo** en el sidebar (métricas, fecha de entrenamiento)

## 🔧 Dependencias Principales

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: Machine Learning y preprocesamiento
- **xgboost**: Modelo de boosting avanzado
- **streamlit**: Framework para aplicaciones web
- **matplotlib/seaborn**: Visualizaciones
- **joblib**: Guardado y carga de modelos

Ver `requirements.txt` para la lista completa.

## 📈 Modelos Implementados

El pipeline prueba y compara los siguientes modelos:

1. **Linear Regression** - Modelo baseline
2. **Random Forest Regressor** - Modelo ensemble basado en árboles
3. **XGBoost Regressor** - Gradient boosting optimizado
4. **Gradient Boosting Regressor** - Boosting estándar

El mejor modelo se selecciona según el RMSE (Root Mean Squared Error) en el conjunto de validación.

## 📊 Métricas de Evaluación

Los modelos se evalúan usando:

- **RMSE** (Root Mean Squared Error): Error cuadrático medio en pesos
- **MAE** (Mean Absolute Error): Error absoluto medio en pesos
- **R²** (Coeficiente de determinación): Proporción de varianza explicada
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio

## 🎯 Características del Dataset

- **Tamaño original**: 278,725 registros
- **Tamaño después de limpieza**: ~95,785 registros
- **Período**: 2021-2022
- **Fuente**: Mercado Libre Argentina
- **Área**: Área Metropolitana de Buenos Aires (AMBA)

### Variables Principales

- **Físicas**: Superficie, dormitorios, baños, ambientes, antigüedad
- **Ubicación**: Ciudad, provincia, barrio, coordenadas geográficas
- **Amenities**: Amoblado, internet, gimnasio, pileta, ascensor, etc.
- **Temporales**: Año, mes, estación
- **Target**: Precio de alquiler mensual en pesos constantes

## 📝 Notas Importantes

1. **Preprocesamiento**: El test set debe pasar por el mismo pipeline de transformación que el conjunto de desarrollo antes de hacer predicciones.

2. **Modelo**: El modelo se guarda en formato pickle (.pkl) junto con metadatos en JSON.

3. **Features**: Las variables categóricas se codifican usando One-Hot Encoding durante el entrenamiento. La aplicación aplica el mismo proceso.

4. **Coordenadas**: Si no se especifican coordenadas en la app, se usan las coordenadas promedio del barrio seleccionado.

---

