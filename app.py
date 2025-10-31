

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import requests
import base64
# import torch
import joblib
import subprocess
import gdown
import matplotlib.pyplot as plt
# --------------------------------------------------------------
# from sklearn.base import BaseEstimator, TransformerMixin
# from sentence_transformers import SentenceTransformer
# ------------------------------------------------------------
# clase BertTransformer

# Para tratar el texto
# class BertTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         # Asegurar que X es iterable de strings
#         if hasattr(X, "iloc"):
#             X = X.iloc[:, 0]
#         elif isinstance(X, list):
#             X = pd.Series(X)
#         X = X.fillna('').astype(str)
#         embeddings = self.model.encode(X.tolist(), show_progress_bar=False)
#         return embeddings

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(page_title="App Innovapay", layout="wide")

# ------------------------------------------------------------
# INSTALACIÓN Y CONFIGURACIÓN DE LIBRERÍAS
# ------------------------------------------------------------
subprocess.run(["pip", "install", "gdown"], check=True)

# ------------------------------------------------------------
# RUTAS Y PARÁMETROS FIJOS
# ------------------------------------------------------------
EDA_NOTEBOOK_GITHUB_URL = "https://github.com/marco11235813/Retencion-de-clientes---NoCountry/blob/main/doc/informe_eda.pdf"
LOOKER_DASHBOARD_URL = "https://lookerstudio.google.com/embed/reporting/3a55f164-2eb4-4b21-983b-08cdffef6786/page/p_12345"
DRIVE_FILE_ID = "1xI5TieWDkdS4j5yT5umMrjFtUE_jCm4a"
MODEL_LOCAL_PATH = "modelo_churn.joblib"

# ------------------------------------------------------------
# DESCARGA Y CARGA DEL MODELO DESDE GOOGLE DRIVE (comentado temporalmente)
# ------------------------------------------------------------
# """
# @st.cache_resource
# def load_model_from_drive():
#     try:
#         if not os.path.exists(MODEL_LOCAL_PATH):
#             url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
#             st.info("📥 Descargando modelo desde Google Drive (esto puede tardar unos segundos)...")
#             gdown.download(url, MODEL_LOCAL_PATH, quiet=False)

#         try:
#             model = joblib.load(MODEL_LOCAL_PATH)
#         except Exception:
#             model = torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu'))

#         return model

#     except Exception as e:
#         st.error(f"❌ No se pudo cargar el modelo. Verificá el enlace o el archivo en Drive.\n\nDetalles: {e}")
#         return None
# """

# ------------------------------------------------------------
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# ------------------------------------------------------------
def plot_histograms(df, ncols=2, max_vars=6):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:max_vars]
    nrows = (len(numeric) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3*nrows))
    axs = np.array(axs).reshape(-1)
    for i, col in enumerate(numeric):
        axs[i].hist(df[col].dropna(), bins=20, color="#0078ff", alpha=0.7)
        axs[i].set_title(col)
    for j in range(len(numeric), len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("Innovapay")
# 👇 Eliminamos la opción de predicción del menú
selection = st.sidebar.radio("Navegar", ["Informe EDA", "Dashboard"])


# ------------------------------------------------------------
# INFORME EDA
# ------------------------------------------------------------
# if selection == "Informe EDA":
#     st.header("📄 Informe — Análisis Exploratorio (EDA)")
#     st.markdown("""
#     **Introducción**  
#     Este informe muestra el análisis exploratorio del dataset de clientes,
#     incluyendo análisis descriptivo, y métricas mas importantes asociadas al churn.
#     """)

#     pdf_path = os.path.join("doc", "informe_eda.pdf")

#     if os.path.exists(pdf_path):
#         with open(pdf_path, "rb") as f:
#             base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#         # Mostrar el PDF directamente en la app
#         pdf_display = f"""
#             <iframe src="data:application/pdf;base64,{base64_pdf}"
#                     width="100%" height="850" type="application/pdf"></iframe>
#         """
#         st.markdown(pdf_display, unsafe_allow_html=True)

#         # Opción para descargarlo
#         with open(pdf_path, "rb") as f:
#             st.download_button(
#                 label="⬇️ Descargar informe EDA (PDF)",
#                 data=f,
#                 file_name="informe_eda.pdf",
#                 mime="application/pdf"
#             )

#     else:
#         st.warning("⚠️ No se encontró el archivo local 'informe_eda.pdf' en la carpeta 'doc/'. Verificá su ubicación.")

#     # Enlace al repositorio (versión online del informe)
#     st.markdown("---")
#     st.markdown(
#         f"📘 [Abrir informe completo en GitHub]({EDA_NOTEBOOK_GITHUB_URL})"
#     )

st.header("📊 Informe EDA")

pdf_path = "doc/Informe_EDA.pdf"
github_url = "https://github.com/usuario/repositorio/blob/main/doc/Informe_EDA.pdf"

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.markdown(f"🔗 [Abrir Informe EDA en GitHub]({github_url})")
# ------------------------------------------------------------
# DASHBOARD (LOOKER STUDIO)
# ------------------------------------------------------------
elif selection == "Dashboard":
    st.header("📊 Dashboard Interactivo — Looker Studio")
    st.markdown("""
    Este dashboard permite visualizar métricas clave sobre el comportamiento de los clientes,
    su evolución temporal y las tasas de retención.
    """)
    st.components.v1.iframe(LOOKER_DASHBOARD_URL, width=1200, height=800, scrolling=True)

# ------------------------------------------------------------
# PREDICCIÓN DE RIESGO DE CHURN (OCULTO EN LA APP)
# ------------------------------------------------------------
# """
# elif selection == "Predicción Riesgo Churn":
#     st.header("⚠️ Predicción de Riesgo de Churn")
#     st.markdown("Ingresá los datos del cliente para obtener una predicción sobre su probabilidad de churn.")

#     model = load_model_from_drive()
#     if model is None:
#         st.stop()

#     with st.form("churn_form"):
#         st.subheader("🧩 Datos del cliente")

#         col1, col2 = st.columns(2)

#         with col1:
#             points_in_wallet = st.number_input("Puntos en la billetera", min_value=0.0, step=0.01, format="%.2f")
#             avg_transaction_value = st.number_input("Valor promedio de transacción", min_value=0.0, step=0.01, format="%.2f")
#             avg_frequency_login_days = st.number_input("Frecuencia promedio de login (días)", min_value=0.0, step=0.01, format="%.2f")

#         with col2:
#             avg_tx_amount = st.number_input("Monto promedio de transacciones", min_value=0.0, step=0.01, format="%.2f")
#             days_since_last_login = st.number_input("Días desde el último login", min_value=0, step=1)
#             membership_category = st.selectbox(
#                 "Categoría de membresía",
#                 ["Platinum Membership", "Premium Membership", "No Membership",
#                  "Gold Membership", "Silver Membership", "Basic Membership"]
#             )

#         feedback = st.text_input("Feedback del cliente", "")

#         submitted = st.form_submit_button("🔍 Predecir")

#     if submitted:
#         try:
#             X_input = pd.DataFrame({
#                 "points_in_wallet": [points_in_wallet],
#                 "avg_transaction_value": [avg_transaction_value],
#                 "avg_frequency_login_days": [avg_frequency_login_days],
#                 "avg_tx_amount": [avg_tx_amount],
#                 "days_since_last_login": [days_since_last_login],
#                 "membership_category": [membership_category],
#                 "feedback": [feedback],
#             })

#             prediction = model.predict(X_input)[0]
#             proba = model.predict_proba(X_input)[0][1]

#             if prediction == 1:
#                 st.error(f"🚨 El cliente tiene **alto riesgo de churn** ({proba:.2%})")
#             else:
#                 st.success(f"✅ El cliente **no presenta riesgo de churn** ({proba:.2%})")

#         except Exception as e:
#             st.error(f"⚠️ Error al realizar la predicción: {e}")
# """

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado por **Datalogic Data Team** 💡")







