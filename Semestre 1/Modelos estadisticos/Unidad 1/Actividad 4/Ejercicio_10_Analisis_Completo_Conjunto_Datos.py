# Ejercicio_10.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad 10: Análisis Completo del Conjunto de Datos Titanic
# Código desarrollado con ayuda de Claude 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Cargamos el dataset "titanic" (class -> clase para evitar la palabra reservada)
titanic = sns.load_dataset('titanic')
titanic = titanic.rename(columns={'class': 'clase'})


# Función para extraer una columna sin NaN
def extraer_serie(df, nombre):
    """Devuelve la serie sin NaN y la cantidad de NaN que tenía."""
    serie = df[nombre]
    n_nan = serie.isna().sum()
    return serie.dropna(), n_nan


encabezados = titanic.columns.tolist()
print(encabezados)

tendencia = []      # filas de la tabla de tendencia central
dispersion = []     # filas de la tabla de dispersión
categoricas = []    # filas de la tabla de variables no numéricas

for nombre in encabezados:
    serie, n_nan = extraer_serie(titanic, nombre)

    # ------------------------------------------------------------
    # Variables numéricas (se excluyen booleanas y categóricas)
    # ------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):

        # 1, 2, 3.- Media, mediana y moda
        resultado = stats.mode(serie.to_numpy(), keepdims=False)
        tendencia.append({
            'Variable': nombre,
            'N válidos': len(serie),
            'NaN': n_nan,
            'Media': np.mean(serie),
            'Mediana': np.median(serie),
            'Moda': resultado.mode,
            'Frec. moda': resultado.count,
        })

        # 4, 5, 6.- Varianza, desviación estándar y percentiles
        q1, q2, q3 = np.percentile(serie, [25, 50, 75])
        dispersion.append({
            'Variable': nombre,
            'Var. pob.': np.var(serie),
            'Var. mues.': np.var(serie, ddof=1),
            'Desv. pob.': np.std(serie),
            'Desv. mues.': np.std(serie, ddof=1),
            'Mínimo': serie.min(),
            'Máximo': serie.max(),
            'Rango': serie.max() - serie.min(),
            'P25': q1,
            'P50': q2,
            'P75': q3,
            'IQR': q3 - q1,
        })

        # 7 y 8.- Histograma y gráfico de cajas en una sola figura
        fig, ejes = plt.subplots(1, 2, figsize=(12, 4))

        ejes[0].hist(serie, bins=10, alpha=0.5, color='blue')
        ejes[0].set_title(f"Histograma de {nombre}")
        ejes[0].set_xlabel(nombre)
        ejes[0].set_ylabel("Frecuencia")

        if nombre == 'survived':
            sns.boxplot(y=serie, ax=ejes[1])
            ejes[1].set_title(f"Gráfico de cajas - {nombre}")
        else:
            sns.boxplot(x='survived', y=nombre, data=titanic, ax=ejes[1])
            ejes[1].set_title(f"Gráfico de cajas - {nombre} según supervivencia")
            ejes[1].set_xlabel("Supervivencia (0 = No, 1 = Sí)")

        plt.tight_layout()
        plt.savefig(f'graficas_{nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ------------------------------------------------------------
    # Variables categóricas o booleanas: solo moda y frecuencias
    # ------------------------------------------------------------
    else:
        conteo = serie.value_counts()
        categoricas.append({
            'Variable': nombre,
            'N válidos': len(serie),
            'NaN': n_nan,
            'Categorías': serie.nunique(),
            'Moda': conteo.index[0],
            'Frec. moda': conteo.iloc[0],
        })

# ----------------------------------------------------------------
# Tablas de resultados
# ----------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

tabla_tendencia = pd.DataFrame(tendencia).set_index('Variable').round(2)
tabla_dispersion = pd.DataFrame(dispersion).set_index('Variable').round(2)
tabla_categoricas = pd.DataFrame(categoricas).set_index('Variable')

print("\nMEDIDAS DE TENDENCIA CENTRAL (variables numéricas)")
print(tabla_tendencia)

print("\nMEDIDAS DE DISPERSIÓN Y PERCENTILES (variables numéricas)")
print(tabla_dispersion)

print("\nVARIABLES CATEGÓRICAS / BOOLEANAS")
print(tabla_categoricas)