# Ejercicio_02.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 2: Sueldos mensuales
# Dataset desarrollado con ayuda de Claude 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("salarios.csv")

print(df.head())
print(df["salario"].describe())

salario = df['salario']

media = np.mean(salario)
print("\nLa media es: ")
print(media)

mediana = np.median(salario)
print("\nLa mediana es: ")
print(mediana)

# Crear histograma para la estaturas de estudiantes
plt.hist(df['salario'], bins=15, alpha=0.5, color='blue')
plt.title("Histograma de salarios")
plt.xlabel("Salario en pesos MXN")
plt.ylabel("Frecuencia")
plt.show()

# Se eliminan los datos de los directivos
sin_directivos = df[df["salario"] <= 30000]

media2 = np.mean(sin_directivos["salario"])
print("\nLa media quitando el sueldo de los directivos es: ")
print(media2)

mediana2 = np.median(sin_directivos["salario"])
print("\nLa mediana quitando el sueldo de los directivos es: ")
print(mediana2)

# Crear histograma para la estaturas de estudiantes
plt.hist(sin_directivos["salario"], bins=15, alpha=0.5, color='blue')
plt.title("Histograma de salarios")
plt.xlabel("Salario en pesos MXN")
plt.ylabel("Frecuencia")
plt.show()