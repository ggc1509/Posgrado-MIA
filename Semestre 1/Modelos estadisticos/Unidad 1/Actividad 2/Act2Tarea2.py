# Act2Tarea2.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadisticos para el Aprendizaje Automatico y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 2: 
# Tarea 2: Funciones de Análisis
# Proyecto "Constructor de Dataset Climático"
			
import pandas as pd

# Crear DataFrame vacío con la estructura del Weather Dataset
weather_data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
df_weather = pd.DataFrame(weather_data)

# a) Contar combinaciones por condición climática
def contar_combinaciones(df, columnas):
    """
    Cuenta cuántas veces aparece cada combinación de valores
    en las columnas indicadas.
    
    Parámetros:
        df (DataFrame): dataset a analizar
        columnas (str o list): columna(s) por las que se agrupa
    
    Retorna:
        Series con el conteo de cada combinación
    """
    return df.groupby(columnas).size().sort_values(ascending=False)


# b) Calcular probabilidades de juego por categoría
def probabilidad_juego(df, columna):
    """
    Calcula la proporción de 'yes'/'no' en 'play' para cada
    categoría de la columna indicada.
    
    Parámetros:
        df (DataFrame): dataset a analizar
        columna (str): columna categórica (ej. 'outlook', 'humidity')
    
    Retorna:
        DataFrame con la probabilidad de cada valor de 'play'
        por categoría
    """
    return df.groupby(columna)['play'].value_counts(normalize=True).unstack(fill_value=0)


# c) Filtrar datos por condiciones específicas
def filtrar_datos(df, **condiciones):
    """
    Filtra el DataFrame según condiciones exactas por columna.
    
    Uso:
        filtrar_datos(df_weather, outlook='sunny', windy=True)
    
    Parámetros:
        df (DataFrame): dataset a analizar
        **condiciones: pares columna=valor a filtrar
    
    Retorna:
        DataFrame filtrado
    """
    filtro = pd.Series(True, index=df.index)
    for columna, valor in condiciones.items():
        filtro &= (df[columna] == valor)
    return df[filtro]


# ------------------- Pruebas -------------------
if __name__ == '__main__':
    print("Dataset original:")
    print(df_weather)

    print("\na) Combinaciones outlook + play:")
    print(contar_combinaciones(df_weather, ['outlook', 'play']))

    print("\na) Combinaciones outlook + play:")
    print(contar_combinaciones(df_weather, ['outlook', 'play']))

    print("\nb) Probabilidad de jugar según 'outlook':")
    print(probabilidad_juego(df_weather, 'outlook'))

    print("\nc) Filtrado: outlook='sunny' y windy=True:")
    print(filtrar_datos(df_weather, outlook='sunny', windy=True))