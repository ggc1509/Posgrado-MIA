# Act1Tarea3.py
# Maestría en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Actividad 1: Proyecto "Explorador avanzado del Dataset Iris"
# Tarea 3: Extensión del código base
# Programa de exploración del dataset "iris" extendido

# La librería "pandas" sirve para la manipulación y analisis de datos
# "sklearn.datasets" contiene los datos  
import pandas as pd
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
data = load_iris()

# Se crea un dataframe con las caracteristicas
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Se toma del dataframe unicamente la columna species, 
# posteriormente con "values_count" se cuenta el número
# de veces que existe el valor en la columna, para 
# posteriormente se regresan los resultados.

# Exporta el dataframe completo a un archivo CSV dentro de la carpeta
# del proyecto
df.to_csv('Dataset_Iris.csv', index=False)
print("Dataset exportado exitosamente a 'iris_dataset.csv'")

#**********************************************************************
# Se guardan las estadísticas en un archivo de texto

with open('estadisticas_iris.txt', 'w', encoding='utf-8') as f:
    f.write("ESTADÍSTICAS DEL DATASET IRIS\n")
    f.write("=" * 50 + "\n\n")

    f.write("Estadísticas descriptivas generales:\n")
    f.write(df.describe().to_string())
    f.write("\n\n")

    f.write("Número de muestras por especie:\n")
    f.write(df['species'].value_counts().to_string())
    f.write("\n\n")

    f.write("Promedios por especie:\n")
    f.write(df.groupby('species', observed=True).mean().to_string())
    f.write("\n\n")

    f.write("Resumen estadístico agrupado por especie:\n")
    f.write(df.groupby('species', observed=True).describe().T.to_string())
    f.write("\n")

print("Estadísticas guardadas exitosamente en 'estadisticas_iris.txt'")

#**********************************************************************
# Se crea un reporte con los hallazgos

promedios = df.groupby('species', observed=True).mean()
especie_petalo_mayor = promedios['petal length (cm)'].idxmax()
especie_petalo_menor = promedios['petal length (cm)'].idxmin()
especie_sepalo_ancho = promedios['sepal width (cm)'].idxmax()

with open('reporte.txt', 'w', encoding='utf-8') as f:
    f.write("REPORTE AUTOMÁTICO DE HALLAZGOS - DATASET IRIS\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"1. El dataset contiene {df.shape[0]} muestras y {df.shape[1]} columnas ")
    f.write(f"({df.shape[1]-1} numéricas + 1 categórica).\n\n")

    f.write("2. Distribución de clases:\n")
    f.write(f"   El dataset está balanceado, con {df['species'].value_counts().iloc[0]} ")
    f.write("muestras por cada una de las 3 especies.\n\n")

    f.write("3. Especie con pétalos más grandes (en promedio):\n")
    f.write(f"   {especie_petalo_mayor} ")
    f.write(f"({promedios.loc[especie_petalo_mayor, 'petal length (cm)']:.2f} cm de longitud promedio)\n\n")

    f.write("4. Especie con pétalos más pequeños (en promedio):\n")
    f.write(f"   {especie_petalo_menor} ")
    f.write(f"({promedios.loc[especie_petalo_menor, 'petal length (cm)']:.2f} cm de longitud promedio)\n\n")

    f.write("5. Especie con sépalos más anchos (en promedio):\n")
    f.write(f"   {especie_sepalo_ancho} ")
    f.write(f"({promedios.loc[especie_sepalo_ancho, 'sepal width (cm)']:.2f} cm de ancho promedio)\n\n")

    f.write("6. Conclusión:\n")
    f.write("   Las medidas del pétalo (longitud y ancho) muestran mayor variación entre\n")
    f.write("   especies que las medidas del sépalo, lo que sugiere que son las variables\n")
    f.write("   más útiles para distinguir entre especies mediante modelos de clasificación.\n")

print("Reporte de hallazgos creado exitosamente en 'reporte.txt'")