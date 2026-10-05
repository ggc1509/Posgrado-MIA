# Ejercicio_08.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 8: Gráfico de cajas (Boxplot) para la longitud del sépalo

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# Cargar el conjunto de datos Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#Crear gráfico de cajas
sns.boxplot(x=iris.target_names[iris.target], y=df['sepal length (cm)'])
plt.title("Gráfico de Cajas - Longitud del Sépalo")
plt.show()