# Ejercicio_03.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: Cálculo de la moda.

import numpy as np
from scipy import stats

data = [5, 7, 5, 10, 5]
datos = np.array(data)
resultado = stats.mode(datos, keepdims=False)
moda = resultado.mode
frecuencia = resultado.count

print("Los datos son:")
print(data)
print("La moda es:")
print(moda)
print("Se repite", frecuencia, "veces")