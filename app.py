# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="App Innovapay", layout="wide")

# ----- Helpers -----
@st.cache_data
def load_local_data(path="data/synthetic/churn_data.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def train_simple_model(df, target_col="churn"):
    """
    Entrena un modelo logístico simple usando todas las columnas numéricas
    (salvo target) y devuelve (model, scaler, X_test, y_test).
    """
    df2 = df.copy()
    if target_col not in df2.columns:
        raise ValueError(f"Target column '{target_col}' no encontrada en el dataframe.")
    # seleccionar columnas numéricas (excluye target)
    X = df2.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df2[target_col].astype(int)
    if X.shape[1] == 0:
        raise ValueError("No hay columnas numéricas para entrenar el modelo. Preprocesamiento requerido.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    return model, scaler, X_train, X_test, y_train, y_test

def plot_histograms(df, ncols=2, max_vars=6):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = numeric[:max_vars]
    n = len(numeric)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3*nrows))
    axs = np.array(axs).reshape(-1)
    for i, col in enumerate(numeric):
        axs[i].hist(df[col].dropna())
        axs[i].set_title(col)
    # hide unused axes
    for j in range(n, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    return fig

def plot_corr_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, interpolation='nearest')
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    # annotate
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    return fig

# ----- Layout: SIDEBAR -----
st.sidebar.title("Innovapay")
selection = st.sidebar.radio("Navegar", ["Informe", "Dashboard", "Riesgo de Churn"])

# Useful file helpers in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Datos / Modelo**")
uploaded = st.sidebar.file_uploader("Subí un CSV (data.csv) para EDA / entrenamiento", type=["csv"])
model_file = st.sidebar.file_uploader("Subí un modelo (model.pkl) opcional", type=["pkl"])

# load local data if exists, else use uploaded
data = None
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Error leyendo CSV subido: {e}")
else:
    data = load_local_data("data.csv")

# if model uploaded, load it
uploaded_model = None
if model_file is not None:
    try:
        uploaded_model = pickle.load(model_file)
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar el modelo: {e}")
        uploaded_model = None

# ----- Informe (EDA) -----
if selection == "Informe":
    st.header("📄 Informe — Análisis Exploratorio (EDA)")
    st.markdown(
        """
        **Introducción**  
        Este informe muestra un análisis exploratorio rápido del dataset disponible. 
        Si no hay datos, podés subir un CSV en la barra lateral (sidebar).
        """
    )

    if data is None:
        st.info("No se detectó un dataset local ni se subió uno. Subí un archivo CSV en la barra lateral para ver el EDA.")
    else:
        st.subheader("Resumen rápido")
        st.write("Dimensiones:", data.shape)
        st.write("Columnas:", list(data.columns))
        st.write("Primeras filas:")
        st.dataframe(data.head())

        st.subheader("Estadísticas descriptivas")
        st.write(data.describe(include="all").transpose())

        # Histograms
        st.subheader("Histogramas (variables numéricas)")
        fig_hist = plot_histograms(data, ncols=2, max_vars=6)
        st.pyplot(fig_hist)

        # Corr
        st.subheader("Matriz de correlación (numéricas)")
        fig_corr = plot_corr_heatmap(data)
        if fig_corr is not None:
            st.pyplot(fig_corr)
        else:
            st.info("No hay suficientes variables numéricas para calcular correlación.")

        # Insights area (simple heuristics)
        st.subheader("Observaciones automáticas (breves)")
        obs = []
        if data.select_dtypes(include=[np.number]).shape[1] > 0:
            obs.append("- Hay variables numéricas: revisar outliers y distribuciones.")
        if "churn" in data.columns:
            churn_rate = data["churn"].mean() if data["churn"].dtype.kind in "biufc" else None
            if churn_rate is not None:
                obs.append(f"- Tasa de churn (promedio de `churn`): {churn_rate:.3f}")
        if len(obs) == 0:
            st.write("No se detectaron observaciones automáticas. Revisá los datos manualmente.")
        else:
            for o in obs:
                st.write(o)

# ----- Dashboard (Looker Studio embed) -----
elif selection == "Dashboard":
    st.header("📊 Dashboard (Looker Studio)")
    st.markdown(
        """
        Aquí podés embeber un informe de Looker Studio.  
        Pegar la URL pública de tu informe abajo (compartir → habilitar acceso).
        """
    )
    report_url = st.text_input("https://lookerstudio.google.com/s/phHIKGaTlyQ", value="")
    if report_url:
        # guardá la URL en un iframe
        st.markdown("**Informe embebido:**")
        # iframe-friendly URL handling could be required; el usuario debe proporcionar la URL embebible
        iframe = f'<iframe src="{report_url}" width="100%" height="800" frameborder="0" style="border:0" allowfullscreen></iframe>'
        st.components.v1.html(iframe, height=800, scrolling=True)
    else:
        st.info("Pegar aquí la URL pública de tu informe de Looker Studio para embeberlo.")

# ----- Riesgo de Churn -----
elif selection == "Riesgo de Churn":
    st.header("⚠️ Riesgo de Churn — Predicción")
    st.markdown("Ingresá las características del cliente para predecir su probabilidad de churn.")

    model = None
    scaler = None
    X_test = None
    y_test = None

    # if a model pickle was uploaded and it includes scaler, use it
    if uploaded_model is not None:
        # esperar que el pickle sea un dict {'model':..., 'scaler':..., 'columns':...}
        if isinstance(uploaded_model, dict) and "model" in uploaded_model:
            model = uploaded_model["model"]
            scaler = uploaded_model.get("scaler", None)
            model_cols = uploaded_model.get("columns", None)
            st.success("Modelo cargado desde archivo .pkl")
        else:
            st.warning("El modelo subido no sigue el formato esperado (dict con clave 'model'). Intentá entrenar con datos.")
            uploaded_model = None

    # si hay data pero no modelo cargado, ofrecer entrenar uno simple
    if model is None and data is not None:
        st.info("Se detectó dataset. Podés entrenar un modelo simple (logistic regression) usando columnas numéricas.")
        if st.button("Entrenar modelo simple con dataset detectado"):
            try:
                model, scaler, X_train, X_test, y_train, y_test = train_simple_model(data, target_col="churn")
                st.success("Modelo entrenado correctamente.")
                # mostrar métricas
                y_pred = model.predict(scaler.transform(X_test))
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("Reporte de clasificación (test set):")
                st.dataframe(pd.DataFrame(report).transpose())
                auc = roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:,1])
                st.write(f"AUC ROC (test): {auc:.3f}")
            except Exception as e:
                st.error(f"No se pudo entrenar el modelo: {e}")

    st.markdown("---")
    st.subheader("Formulario de características (valores de ejemplo)")

    # Si hay un modelo y scaler, intentamos usar las columnas del X_train o inferir variables numéricas
    example_num_cols = []
    if data is not None:
        example_num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if "churn" in example_num_cols:
            example_num_cols.remove("churn")

    # if no numeric columns, allow user to define a simple numeric vector
    if example_num_cols:
        st.info("Si entrenaste o cargaste un modelo, los campos esperados son las columnas numéricas del dataset.")
        inputs = {}
        with st.form(key="predict_form"):
            for col in example_num_cols:
                # default value = median
                default = float(data[col].median()) if col in data.columns else 0.0
                inputs[col] = st.number_input(col, value=default, format="%.4f")
            submit = st.form_submit_button("Predecir riesgo")
        if submit:
            if model is None:
                st.error("No hay un modelo disponible. Entrenalo o subí un model.pkl en la sidebar.")
            else:
                X_new = np.array([inputs[c] for c in example_num_cols]).reshape(1, -1)
                if scaler is not None:
                    X_new_s = scaler.transform(X_new)
                else:
                    X_new_s = X_new
                proba = model.predict_proba(X_new_s)[0,1]
                st.success(f"Probabilidad estimada de churn: {proba:.3f}")
    else:
        st.info("No se detectan columnas numéricas en el dataset. Podés subir un dataset con variables numéricas o definir manualmente las features.")
        with st.form("manual"):
            n = st.number_input("Cantidad de features (numéricas) a ingresar", min_value=1, max_value=20, value=3)
            inputs = {}
            for i in range(int(n)):
                inputs[f"x{i+1}"] = st.number_input(f"x{i+1}", value=0.0, format="%.4f")
            submit2 = st.form_submit_button("Predecir con valores manuales")
        if submit2:
            if model is None:
                st.error("No hay un modelo disponible. Entrenalo o subí un model.pkl en la sidebar.")
            else:
                X_new = np.array([inputs[k] for k in sorted(inputs.keys())]).reshape(1, -1)
                if scaler is not None:
                    X_new_s = scaler.transform(X_new)
                else:
                    X_new_s = X_new
                proba = model.predict_proba(X_new_s)[0,1]
                st.success(f"Probabilidad estimada de churn: {proba:.3f}")

# ----- Footer -----
st.sidebar.markdown("---")
st.sidebar.markdown("App creada para Innovapay")





