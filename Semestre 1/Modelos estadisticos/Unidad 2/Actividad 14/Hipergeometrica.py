# Hipergeometrica.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 14: 
# Distribución de Hipergeométrica

# Importación de librerías necesarias
import matplotlib.pyplot as plt      # Generación de gráficas
from scipy import stats              # Distribuciones estadísticas
import numpy as np                   # Operaciones numéricas y arreglos

# ==============================
# Gráfica de la distribución hipergeométrica
# ==============================

# Parámetros de la distribución:
# M = tamaño total de la población
# n = número de elementos exitosos en la población
# N = tamaño de la muestra extraída
M, n, N = 30, 10, 12

# Creación del objeto de distribución hipergeométrica
hipergeometrica = stats.hypergeom(M, n, N)

# Generación de los posibles valores de la variable aleatoria
x = np.arange(0, n + 1)

# Cálculo de la Función de Masa de Probabilidad (PMF)
fmp = hipergeometrica.pmf(x)

# Gráfica de la distribución
plt.plot(x, fmp, '--')                         # Línea punteada
plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)  # Barras verticales

# Etiquetas y título de la gráfica
plt.title('Distribución Hipergeométrica')
plt.ylabel('Probabilidad')
plt.xlabel('Valores')

# Mostrar la gráfica
plt.show()

# ==============================
# Histograma de datos aleatorios
# ==============================

# Generación de 1000 valores aleatorios
# que siguen la distribución hipergeométrica
aleatorios = hipergeometrica.rvs(1000)

# Creación del histograma
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)

# Etiquetas y título del histograma
plt.ylabel('Frecuencia')
plt.xlabel('Valores')
plt.title('Histograma Hipergeométrica')

# Mostrar el histograma
plt.show()