# Act1Tarea2.py
# Maestría en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Actividad 1: Proyecto "Explorador avanzado del Dataset Iris"
# Tarea 2: Extensión del código base
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

# Se toma del dataframe unicamente la columna species, 
# posteriormente con "values_count" se cuenta el número
# de veces que existe el valor en la columna, para 
# posteriormente se regresan los resultados.

print("\n\nSe muestran el numero de elementos de cada variedad ")
print(df['species'].value_counts())

print("\n\nEn este apartado se observan los promedios de las tres variedades")
print(df.groupby('species', observed=True).mean())

print("\n\nAhora presentamos un resumen estadistico de cada especie")
print(df.groupby('species', observed=True).describe())