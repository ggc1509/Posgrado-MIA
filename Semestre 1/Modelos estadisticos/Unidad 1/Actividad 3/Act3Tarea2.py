# Act3Tarea2.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadisticos para el Aprendizaje Automatico y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: 
# Ejercicio 1.2. Codificación One-Hot en Python

import pandas as pd
# Datos de ejemplo
# Se crea un diccionario, "Color de ojos" es el nombre de la columna
# ['Azul', 'Verde', 'Marrón', 'Azul', 'Marrón'] es la lista con 5 filas
data = {'Color de Ojos': ['Azul', 'Verde', 'Marrón', 'Azul', 'Marrón']}


# Creamos un dataframe a partir de la lista
# pd.DataFrame ---> Convierte el diccionario creado
# anteriormente en un DataFrame de pandas
df = pd.DataFrame(data)

# Imprime el DataFrame creado 
print('Dataframe creado a partir de la lista')
print(df)
print('\n')

# Realizar la codificación One-Hot
# pd.get_dummies() Convierte la columna en varias columnas 
# binarias.
df_encoded = pd.get_dummies(df, columns=['Color de Ojos'])

# Imprime la tabla
print('Tabla creada')
print(df_encoded)
