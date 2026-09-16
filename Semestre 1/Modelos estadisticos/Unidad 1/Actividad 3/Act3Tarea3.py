# Act3Tarea3.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadisticos para el Aprendizaje Automatico y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: 
# Ejercicio 1.3. Normalización de datos cuantitativos.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#Datos de ejemplo para la columna Altura
altura = [170, 165, 180, 160, 175]

# Creamos un DataFrame de nombre df a partir de los
# datos de "altura".
df = pd.DataFrame({'Altura': altura})
print(df)
print('\n')

# Crear el objeto scaler
# MinMaxScaler() transforma los datos en una escala de
# 0 a 1, siendo 160 el inicio de la escala o 0 y 
# 180 el fin de la escala o 1.
scaler = MinMaxScaler()

#Normalizar los datos
df['Altura_normalizada'] = scaler.fit_transform(df[['Altura']])
print(df)