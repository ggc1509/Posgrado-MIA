# Ejercicio_06.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 6: Cálculo de percentiles

import numpy as np

data =  [8, 14, 10, 12, 18]
datos = np.array(data)

datos = (data)

p25, p50, p75 = np.percentile(datos,[25,50,75])

print("Los datos son: ")
print (data)
print("El percentil 25 es: ")
print(p25)
print("El percentil 50 es: ")
print(p50)
print("El percentil 75 es: ")
print(p75)