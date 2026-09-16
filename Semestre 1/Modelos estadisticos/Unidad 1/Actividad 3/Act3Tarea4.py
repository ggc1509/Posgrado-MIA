# Act3Tarea3.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadisticos para el Aprendizaje Automatico y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: 
# Ejercicio 1.4. Análisis de variables cualitativas (Análisis de frecuencia).

import pandas as pd

#Datos de ejemplo para la columna Satisfacción
satisfaccion = ['Alta','Baja', 'Media', 'Alta', 'Baja']

#Crear un dataframe
df = pd.DataFrame({'satisfaccion':satisfaccion})
print('\nDataFrame')
print(df)


#Calcular la frecuencia de cada categoría
frec_df = df['satisfaccion'].value_counts()
print('\nFrecuencia de cada categoría')
print(frec_df)
