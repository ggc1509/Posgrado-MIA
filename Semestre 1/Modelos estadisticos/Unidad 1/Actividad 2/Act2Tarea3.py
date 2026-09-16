# Act2Tarea3.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadisticos para el Aprendizaje Automatico y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 3: 
# Tarea 1: Generador de Reglas
# Proyecto "Constructor de Dataset Climático"

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Crear DataFrame con la estructura del Weather Dataset
weather_data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
df_weather = pd.DataFrame(weather_data)
print(df_weather)


# ============================================================
# Tarea 3: Generador de Reglas
# ============================================================

# a) Extraer reglas automáticamente del dataset
def extraer_reglas(df, columnas_condicion, columna_objetivo='play', min_soporte=1):
    """
    Genera reglas del tipo 'SI condiciones ENTONCES resultado' para
    combinaciones donde el resultado es 100% consistente (todas las
    filas de esa combinación tienen el mismo valor en columna_objetivo).

    Parámetros:
        df (DataFrame): dataset a analizar
        columnas_condicion (list): columnas a combinar como antecedente
        columna_objetivo (str): columna a predecir (ej. 'play')
        min_soporte (int): mínimo de casos requeridos para considerar la regla

    Retorna:
        list de strings con las reglas encontradas
    """
    reglas = []
    grupos = df.groupby(columnas_condicion)[columna_objetivo]

    for condiciones, valores in grupos:
        if len(valores) < min_soporte:
            continue
        proporciones = valores.value_counts(normalize=True)
        valor_mayoritario = proporciones.idxmax()
        confianza = proporciones.max()

        # Solo reglas 100% consistentes (o casi, según se necesite)
        if confianza == 1.0:
            if not isinstance(condiciones, tuple):
                condiciones = (condiciones,)
            antecedente = " AND ".join(
                f"{col}={val}" for col, val in zip(columnas_condicion, condiciones)
            )
            regla = f"SI {antecedente} ENTONCES {columna_objetivo}={valor_mayoritario} (soporte={len(valores)}, confianza={confianza:.0%})"
            reglas.append(regla)

    return reglas


# b) Generar un árbol de decisión básico
def generar_arbol_basico(df, columnas_features, columna_objetivo='play'):
    """
    Entrena un árbol de decisión con las columnas categóricas indicadas.

    Retorna:
        arbol: modelo DecisionTreeClassifier entrenado
        encoders: dict de LabelEncoder por columna (necesarios para predecir)
        texto_arbol: representación en texto del árbol
    """
    df_encoded = df.copy()
    encoders = {}

    # Codificar cada columna categórica (incluyendo el objetivo)
    for col in columnas_features + [columna_objetivo]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df_encoded[columnas_features]
    y = df_encoded[columna_objetivo]

    arbol = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arbol.fit(X, y)

    texto_arbol = export_text(arbol, feature_names=columnas_features)

    return arbol, encoders, texto_arbol


# c) Predecir "play" para nuevas condiciones
def predecir_play(arbol, encoders, columnas_features, columna_objetivo, nuevas_condiciones):
    """
    Predice el valor de 'play' para un nuevo conjunto de condiciones.

    Parámetros:
        arbol: modelo entrenado (de generar_arbol_basico)
        encoders: dict de LabelEncoder (de generar_arbol_basico)
        columnas_features (list): columnas usadas como entrada
        columna_objetivo (str): columna a predecir
        nuevas_condiciones (dict): ej. {'outlook': 'sunny', 'temperature': 'cool', ...}

    Retorna:
        la predicción ya decodificada (ej. 'yes' o 'no')
    """
    fila = pd.DataFrame([nuevas_condiciones])[columnas_features]
    for col in columnas_features:
        fila[col] = encoders[col].transform(fila[col])

    prediccion_codificada = arbol.predict(fila)
    prediccion = encoders[columna_objetivo].inverse_transform(prediccion_codificada)

    return prediccion[0]


# ------------------- Pruebas Tarea 3 -------------------
if __name__ == '__main__':
    columnas_features = ['outlook', 'temperature', 'humidity', 'windy']

    print("\n--- a) Reglas extraídas (outlook, humidity, windy -> play) ---")
    for regla in extraer_reglas(df_weather, ['outlook', 'humidity', 'windy']):
        print(" -", regla)

    print("\n--- b) Árbol de decisión ---")
    arbol, encoders, texto_arbol = generar_arbol_basico(df_weather, columnas_features)
    print(texto_arbol)

    print("\n--- c) Predicción para nuevas condiciones ---")
    nueva = {'outlook': 'sunny', 'temperature': 'cool', 'humidity': 'high', 'windy': True}
    resultado = predecir_play(arbol, encoders, columnas_features, 'play', nueva)
    print(f"Condiciones: {nueva}")
    print(f"Predicción de play: {resultado}")