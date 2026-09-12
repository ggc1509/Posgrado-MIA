# Act1Tarea1.py
# Maestría en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Actividad 1: Proyecto "Explorador avanzado del Dataset Iris"
# Tarea 1: Extensión del código base
# Programa de exploración del dataset "iris" extendido


# La librería "pandas" sirve para la manipulación y analisis de datos
# "sklearn.datasets" contiene los datos  
import pandas as pd
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
data = load_iris()

# Se crea un dataframe con las caracteristicas
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Primeras 10 filas
print("Primeras 10 filas:")
print(df.head(10))

# Últimas 8 filas
print("\nÚltimas 8 filas:")
print(df.tail(8))

# Muestra en consola la información relevante del dataset
df.info()

# Estadísticas descriptivas del dataset
print("\nEstadísticas descriptivas:")
print(df.describe())