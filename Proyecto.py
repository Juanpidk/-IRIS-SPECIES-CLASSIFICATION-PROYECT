import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Iris Classification", layout="wide")

st.title("🌼 Iris Species Classification — Proyecto Final")
st.markdown("Este dashboard utiliza **Iris.csv** para entrenar un modelo Random Forest y predecir la especie de iris.")

@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    df = df.rename(columns={
        "SepalLengthCm": "sepal_length",
        "SepalWidthCm": "sepal_width",
        "PetalLengthCm": "petal_length",
        "PetalWidthCm": "petal_width",
        "Species": "species"
    })
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df

df = load_data()

# Sidebar
st.sidebar.header("Configuración del modelo")
test_size = st.sidebar.slider("Tamaño del test (%)", 10, 40, 20)
n_estimators = st.sidebar.slider("Árboles (n_estimators)", 10, 200, 50, 10)
max_depth = st.sidebar.slider("Profundidad máxima (0 = None)", 0, 20, 0)

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=(None if max_depth == 0 else max_depth),
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Métricas
st.subheader("📊 Métricas del Modelo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
c2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")
c3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.4f}")
c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.4f}")

with st.expander("🔎 Reporte de Clasificación y Matriz de Confusión"):
    st.text(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    cm_df = pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_)
    st.dataframe(cm_df)


# Visualizaciones
st.subheader("📈 Visualizaciones del Dataset")

fig = px.histogram(df, x="sepal_length", color="species",
                   nbins=20, title="Histograma de Sepal Length")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter_matrix(
    df,
    dimensions=["sepal_length","sepal_width","petal_length","petal_width"],
    color="species",
    title="Scatter Matrix"
)
fig2.update_traces(diagonal_visible=False)
st.plotly_chart(fig2, use_container_width=True)

# Panel de Predicción
st.subheader("🧠 Predicción de una Nueva Muestra")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        sl = st.number_input("Sepal Length", 0.0, 10.0, 5.1, 0.1)
        sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5, 0.1)
    with c2:
        pl = st.number_input("Petal Length", 0.0, 10.0, 1.4, 0.1)
        pw = st.number_input("Petal Width", 0.0, 10.0, 0.2, 0.1)

    submit = st.form_submit_button("Predecir")

    if submit:
        X_new = np.array([[sl, sw, pl, pw]])
        pred = rf.predict(X_new)[0]
        proba = rf.predict_proba(X_new)[0]

        st.success(f"🌼 Especie predicha: **{pred}**")

        proba_df = pd.DataFrame({
            "species": rf.classes_,
            "probability": proba
        })
        st.table(proba_df)

        fig3 = px.scatter_3d(
            df,
            x="petal_length",
            y="petal_width",
            z="sepal_length",
            color="species",
            title="Ubicación de la nueva muestra"
        )
        fig3.add_trace(go.Scatter3d(
            x=[pl], y=[pw], z=[sl],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Nueva Muestra"
        ))
        st.plotly_chart(fig3, use_container_width=True)
