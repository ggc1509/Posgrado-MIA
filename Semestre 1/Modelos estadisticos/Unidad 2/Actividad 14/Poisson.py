# Poisson.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 14: 
# Distribución de Poisson

# Importa la biblioteca para generar gráficos
import matplotlib.pyplot as plt

# Importa el módulo de distribuciones estadísticas de SciPy
from scipy import stats

# Importa NumPy para trabajar con arreglos numéricos
import numpy as np

# -------------------------------
# Gráfica de una distribución Poisson
# -------------------------------

# Parámetro λ (lambda) o media de la distribución de Poisson
mu = 3.6

# Crea el objeto de la distribución de Poisson con λ = 3.6
poisson = stats.poisson(mu)

# Genera los valores del eje x comprendidos entre los percentiles
# 1% y 99% de la distribución
x = np.arange(
    poisson.ppf(0.01),
    poisson.ppf(0.99)
)

# Calcula la Función de Masa de Probabilidad (PMF)
# para cada valor de x
fmp = poisson.pmf(x)

# Dibuja una línea punteada que une las probabilidades
plt.plot(x, fmp, '--')

# Dibuja barras verticales para representar que la distribución
# de Poisson es una distribución discreta
plt.vlines(
    x,          # posiciones en el eje x
    0,          # inicio de las líneas
    fmp,        # altura de cada línea
    colors='b',# color azul
    lw=5,       # grosor de línea
    alpha=0.5  # transparencia
)

# Agrega un título al gráfico
plt.title('Distribución Poisson')

# Etiqueta del eje Y
plt.ylabel('Probabilidad')

# Etiqueta del eje X
plt.xlabel('Valores')

# Muestra la gráfica en pantalla
plt.show()


# ---------------------------------------
# Generación y visualización de un histograma
# para datos con distribución de Poisson
# ---------------------------------------

# Genera 1000 valores aleatorios que siguen
# la distribución de Poisson definida previamente
aleatorios = poisson.rvs(1000)

# Construye el histograma:
# cuenta: frecuencia de datos en cada intervalo
# cajas: límites de los intervalos (bins)
# ignorar: objetos gráficos creados por Matplotlib
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)

# Etiqueta del eje Y
plt.ylabel('Frecuencia')

# Etiqueta del eje X
plt.xlabel('Valores')

# Título de la gráfica
plt.title('Histograma Poisson')
