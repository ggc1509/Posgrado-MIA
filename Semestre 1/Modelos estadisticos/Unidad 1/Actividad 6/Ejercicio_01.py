# Ejercicio_01.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 1: Estatura de estudiantes
# Dataset desarrollado con ayuda de Claude 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("estaturas.csv")

print(df.head())
print(df["estatura_cm"].describe())

estatura = df['estatura_cm']

media = np.mean(estatura)
print("\nLa media es: ")
print(media)

mediana = np.median(estatura)
print("\nLa mediana es: ")
print(mediana)

# Crear histograma para la estaturas de estudiantes
plt.hist(df['estatura_cm'], bins=15, alpha=0.5, color='blue')
plt.title("Histograma de estaturas de estudiantes")
plt.xlabel("Estatura (cm)")
plt.ylabel("Frecuencia")
plt.show()
