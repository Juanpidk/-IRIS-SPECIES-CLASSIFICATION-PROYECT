# 🌼 Iris Species Classification — Proyecto Final de Minería de Datos

Este proyecto implementa un modelo de **clasificación supervisada** para predecir la especie de una flor Iris usando el dataset **Iris.csv**.  
El usuario puede **entrenar el modelo**, **visualizar métricas**, explorar el dataset con gráficos interactivos y **predecir nuevas muestras** directamente desde un dashboard desarrollado en **Streamlit**.

---

## 🚀 ¿Qué verás al ejecutar la aplicación?

Al iniciar el programa (`Proyecto.py`) en Streamlit, encontrarás un **dashboard organizado, intuitivo y completamente interactivo**, dividido en estas secciones:

---

### ## 1️⃣ Barra lateral — “Configuración del modelo”
Desde la barra lateral podrás:

- Ajustar el porcentaje del conjunto de prueba (test size).
- Modificar la cantidad de árboles del Random Forest (`n_estimators`).
- Cambiar la profundidad máxima del árbol (`max_depth`).
- Visualizar cómo cambian las métricas al entrenar el modelo con nuevos parámetros.

Estos controles permiten experimentar con el modelo de forma fácil.

---

### ## 2️⃣ Métricas del modelo (después de entrenar)

La primera sección principal muestra:

- ✔ **Accuracy**
- ✔ **Precision**
- ✔ **Recall**
- ✔ **F1-Score**

Estas métricas se calculan automáticamente cada vez que ajustas parámetros.

También puedes abrir un panel adicional que contiene:

- 📄 **Reporte de Clasificación Completo**  
  (precision, recall, f1 por clase)
- 🔢 **Matriz de Confusión**  
  mostrada en tabla con colores para facilitar la interpretación.

---

### ## 3️⃣ Visualizaciones del dataset (“Dashboard visual”)

La aplicación incluye gráficos interactivos que ayudan a entender la estructura del dataset:

- 📊 **Histograma** de `sepal_length` agrupado por especie  
- 🌐 **Scatter Matrix (matriz de dispersión)**  
  para ver cómo se relacionan las 4 características entre sí
- 🔺 **Gráfico 3D interactivo**  
  que muestra las flores en un espacio tridimensional según sus medidas

Estas visualizaciones permiten identificar patrones entre las distintas especies.

---

### ## 4️⃣ Panel de predicción (muy fácil de usar)

Podrás ingresar manualmente los valores de una nueva flor:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

Al presionar **"Predecir"**, la app mostrará:

- 🌼 **La especie predicha**
- 📈 **Las probabilidades para cada clase**

Además, verás un **gráfico 3D colocando tu nueva flor dentro del dataset real**, lo que permite ver visualmente a qué tipo se parece más.

---

### ## 5️⃣ Guardar modelo entrenado (opcional)
En la barra lateral encontrarás un botón:

> **Guardar modelo**

Esto crea un archivo `rf_iris_model.joblib` con el modelo ya entrenado.

---

## 📁 Archivos incluidos en este repositorio
