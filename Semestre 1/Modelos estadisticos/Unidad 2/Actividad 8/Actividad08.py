# Actividad08.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# 
# Código Actividad : Implementación de Modelos Predictivos con Incertidumbre

# Importamos la libreria pandas y el dataset Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# Aquí cargamos los datos
iris = load_iris()
X, y = iris.data, iris.target


# División 70% entrenamiento / 30% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Verificación
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Prueba:", X_test.shape, y_test.shape)
print("Clases en entrenamiento:", np.bincount(y_train))
print("Clases en prueba:", np.bincount(y_test))


# Crear y entrenar el modelo
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Predicciones de clase y probabilísticas
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)

# Verificación
print("Forma de y_proba:", y_proba.shape)
print("Exactitud:", accuracy_score(y_test, y_pred))

# Ver las primeras 5 flores de prueba
np.set_printoptions(precision=3, suppress=True)
print("Probabilidades (primeras 5):\n", y_proba[:5])
print("Clase predicha:", y_pred[:5])
print("Clase real:    ", y_test[:5])



nombres = iris.target_names

# --- Confianza y margen ---
confianza = y_proba.max(axis=1)
ordenadas = np.sort(y_proba, axis=1)
margen = ordenadas[:, -1] - ordenadas[:, -2]

# --- Casos inciertos: confianza máxima entre 0.4 y 0.7 ---
mask_05 = (confianza >= 0.4) & (confianza <= 0.7)
print(f"Casos con confianza entre 0.4 y 0.7: {mask_05.sum()}")

# Si hay pocos, ampliar el umbral
UMBRAL = 0.9
mask_inc = confianza < UMBRAL
print(f"Casos con confianza < {UMBRAL}: {mask_inc.sum()}")

# --- Tabla de casos inciertos ---
for i in np.where(mask_inc)[0]:
    print(f"\nFlor de prueba #{i}")
    print("  Real:    ", nombres[y_test[i]])
    print("  Predicha:", nombres[y_pred[i]])
    print("  Probabilidades:", dict(zip(nombres, y_proba[i].round(3))))
    print("  Margen:", margen[i].round(3))

# --- Histogramas ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].hist(confianza, bins=10, edgecolor="black")
ax[0].set_title("Confianza máxima por predicción")
ax[0].set_xlabel("Probabilidad máxima")
ax[0].set_ylabel("Número de flores")

for j, color in zip([1, 2], ["tab:orange", "tab:green"]):
    ax[1].hist(y_proba[:, j], bins=10, alpha=0.6,
               label=nombres[j], color=color, edgecolor="black")
ax[1].set_title("P(versicolor) y P(virginica)")
ax[1].set_xlabel("Probabilidad predicha")
ax[1].legend()

ax[2].hist(y_proba[:, 0], bins=10, color="tab:blue", edgecolor="black")
ax[2].set_title("P(setosa)")
ax[2].set_xlabel("Probabilidad predicha")

plt.tight_layout()
plt.show()
