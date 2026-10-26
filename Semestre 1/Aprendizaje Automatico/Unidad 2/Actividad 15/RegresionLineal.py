# RegresionLineal.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Aprendizaje Automático
# Germán Godínez Cardoza
# Código Actividad 15: 
# Aplicación de regresión lineal 

import matplotlib.pyplot as plt
import numpy as np

# 1. Datos de la tabla
x = np.array([5, 7, 10, 12])       # Publicidad (miles)
y = np.array([52, 58, 65, 70])     # Ventas

# 2. Calcular la regresión lineal (grado 1)
# np.polyfit devuelve la pendiente (m) y la intersección en y (b)
m, b = np.polyfit(x, y, 1)

# 3. Crear el gráfico de dispersión (los puntos originales)
plt.scatter(x, y, color='blue', s=60, zorder=5, label='Datos reales')

# 4. Generar los puntos para dibujar la línea de regresión suavemente
# np.linspace crea un rango de valores entre el mínimo y máximo de x
x_linea = np.linspace(min(x), max(x), 100)
y_linea = m * x_linea + b

# 5. Graficar la línea de regresión
plt.plot(x_linea, y_linea, color='green', linewidth=2, 
         label=f'Regresión: y = {m:.2f}x + {b:.2f}')

# 6. Añadimos etiquetas, título, leyenda y cuadrícula
plt.xlabel('Publicidad (miles)')
plt.ylabel('Ventas')
plt.title('Relación entre Publicidad y Ventas')
plt.legend()  # Muestra la leyenda con las etiquetas que definimos
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

# 7. Mostramos el gráfico
plt.show()
