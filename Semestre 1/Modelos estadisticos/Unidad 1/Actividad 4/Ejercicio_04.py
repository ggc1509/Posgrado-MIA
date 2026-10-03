# Ejercicio_04.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 4: Cálculo de la varianza.

import numpy as np

data =  [5, 7, 9, 6, 10]
datos = np.array(data)

var_poblacion = np.var(datos)
var_muestral = np.var(datos, ddof=1)
print("Los datos son: ")
print (data)
print("La varianza poblacional es: ")
print(var_poblacion)
print("La varianza muestral es: ")
print(var_muestral)