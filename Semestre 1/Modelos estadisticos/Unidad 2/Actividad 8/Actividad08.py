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
