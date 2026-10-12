# Ejercicio_03.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: Ingresos mensuales
# Dataset desarrollado con ayuda de Claude 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("salarios-15.csv")

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

# Se eliminan los datos vacios
limpio = df[df["salario"] > 0]

media2 = np.mean(limpio["salario"])
print("\nLa media quitando los valores vacios: ")
print(media2)

mediana2 = np.median(limpio["salario"])
print("\nLa mediana quitando los valores vacios: ")
print(mediana2)

# Crear histograma para la estaturas de estudiantes
plt.hist(limpio["salario"], bins=15, alpha=0.5, color='blue')
plt.title("Histograma de salarios sin valores faltantes")
plt.xlabel("Salario en pesos MXN")
plt.ylabel("Frecuencia")
plt.show()