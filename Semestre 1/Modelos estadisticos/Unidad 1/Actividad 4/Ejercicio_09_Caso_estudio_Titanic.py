# Ejercicio_09.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 9: Caso de Estudio: Análisis Descriptivo del Conjunto de Datos Titanic

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Cargamos el dataset "titanic"
titanic = sns.load_dataset('titanic')
#print(titanic)


edad = titanic['age'].dropna()
clase = titanic['pclass']

# Quitar los NaN de la columna edad
edad = titanic['age'].dropna()
print(edad)

#**********************************************
#   1.- Calculo de la media
#**********************************************

media = np.mean(edad)
print("\nLa media es: ")
print(media)

#**********************************************
#   2.- Calculo de la mediana
#**********************************************

mediana = np.median(edad)
print("\nLa mediana es: ")
print(mediana)


#**********************************************
#   3.- Calculo de la moda
#**********************************************

resultado = stats.mode(clase, keepdims=False)
moda = resultado.mode
frecuencia = resultado.count

print("\nLa moda es:")
print(moda)
print("Se repite", frecuencia, "veces")

#**********************************************
#   4.- Calculo de la varianza
#**********************************************

var_poblacion = np.var(edad)
var_muestral = np.var(edad, ddof=1)
print("\nLa varianza poblacional es: ")
print(var_poblacion)
print("La varianza muestral es: ")
print(var_muestral)

#**********************************************
#   5.- Calculo de la desviación estándar
#**********************************************

desv_pob = np.std(edad)
desv_mues = np.std(edad, ddof=1)
print("\nLa desviación estandard poblacional es: ")
print(desv_pob)
print("La desviación estandard muestral es: ")
print(desv_mues)

#**********************************************
#   6.- Calculo de la percentiles
#**********************************************

p25, p50, p75 = np.percentile(edad,[25,50,75])

print("\nEl percentil 25 es: ")
print(p25)
print("El percentil 50 es: ")
print(p50)
print("El percentil 75 es: ")
print(p75)


#**********************************************
#   7.- Histograma
#**********************************************

plt.hist(edad, bins=10, alpha=0.5, color='blue')
plt.title("Histograma de distribución de edades de los pasajeros del Titanic")
plt.xlabel("Edad (años)")
plt.ylabel("Frecuencia")
plt.show()

#**********************************************
#   8.- Crear gráfico de cajas
#**********************************************

sns.boxplot(x='survived', y='age', data=titanic)
plt.title("Gráfico de Cajas - Edad según Supervivencia")
plt.xlabel("Supervivencia (0 = No sobrevivió, 1 = Sobrevivió)")
plt.ylabel("Edad (años)")
plt.show()
