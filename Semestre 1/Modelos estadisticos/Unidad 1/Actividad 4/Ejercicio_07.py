# Ejercicio_07.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 7: Visualización de datos con un histograma

import matplotlib.pyplot as plt
from sklearn. datasets import load_iris
import pandas as pd
import numpy as np
from scipy import stats

# Cargar el conjunto de datos Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Crear histograma para la longitud del sépalo
plt.hist(df['sepal length (cm)'], bins=10, alpha=0.5, color='blue')
plt.title("Histograma de Longitud de Sépalo")
plt.xlabel("Longitud de Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.show()

sepal_length = df['sepal length (cm)']
datos = (sepal_length)

print("\nLos datos son: ")
print (datos)

media = np.mean(datos)
print("\nLa media es: ")
print(media)

mediana = np.median(datos)
print("\nLa mediana es: ")
print(mediana)


resultado = stats.mode(datos, keepdims=False)
moda = resultado.mode
frecuencia = resultado.count

print("\nLa moda es:")
print(moda)
print("Se repite", frecuencia, "veces")

var_poblacion = np.var(datos)
var_muestral = np.var(datos, ddof=1)
print("\nLa varianza poblacional es: ")
print(var_poblacion)
print("La varianza muestral es: ")
print(var_muestral)

desv_pob = np.std(datos)
desv_mues = np.std(datos, ddof=1)
print("\nLa desviación estandard poblacional es: ")
print(desv_pob)
print("La desviación estandard muestral es: ")
print(desv_mues)


p25, p50, p75 = np.percentile(datos,[25,50,75])

print("\nEl percentil 25 es: ")
print(p25)
print("El percentil 50 es: ")
print(p50)
print("El percentil 75 es: ")
print(p75)
