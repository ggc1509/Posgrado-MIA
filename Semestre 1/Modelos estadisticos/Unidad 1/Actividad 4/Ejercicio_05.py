# Ejercicio_05.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 5: Cálculo de la desviación estándar.

import numpy as np

data =  [15, 17, 19, 13, 20]
datos = np.array(data)

datos = (data)
desv_pob = np.std(datos)
desv_mues = np.std(datos, ddof=1)
print("Los datos son: ")
print (data)
print("La desviación estandard poblacional es: ")
print(desv_pob)
print("La desviación estandard muestral es: ")
print(desv_mues)