# 02_No_Supervisado.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Aprendizaje Automático
# Germán Godínez Cardoza
# Código Actividad 2: 
# Aplicar un algoritmo no supervisado (K-Means o PCA).
# Código desarrollado usando Claude de Anthropic con modelo Sonnet 5

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

CARPETA = "graficas_no_supervisado"
sns.set_theme(style="whitegrid")

# ---------------- CONFIGURACIÓN ----------------
# MODO de variables:
#   "completo"    -> las 13 variables originales
#   "2_variables" -> solo 2 variables originales (las de VARIABLES_2)
#   "2_pca"       -> las 13 variables reducidas a 2 componentes principales
MODO = "completo"
VARIABLES_2 = ["flavanoids", "color_intensity"]
MOSTRAR = True     # True: abre una ventana por gráfica (ciérrala para ver la siguiente)
GUARDAR = False    # True: además guarda cada gráfica como PNG en CARPETA
# -----------------------------------------------

if GUARDAR:
    os.makedirs(CARPETA, exist_ok=True)


def salida(nombre):
    """Guarda y/o muestra la figura actual."""
    if GUARDAR:
        plt.savefig(f"{CARPETA}/{nombre}", dpi=150)
    if MOSTRAR:
        plt.show()
    plt.close()

# ------------------------------------------------------------------
# 1. Datos (MISMO dataset que el programa supervisado)
# ------------------------------------------------------------------
wine = load_wine(as_frame=True)
X, y_real = wine.data, wine.target          # y_real SOLO para interpretar al final
clases = list(wine.target_names)
if MODO == "2_variables":
    X = X[VARIABLES_2]
X_esc = StandardScaler().fit_transform(X)   # PCA y K-Means dependen de escalas -> estandarizar

# ------------------------------------------------------------------
# 2. PCA
# ------------------------------------------------------------------
pca = PCA().fit(X_esc)
Z = pca.transform(X_esc)
var = pca.explained_variance_ratio_
n = X.shape[1]
n95 = int(np.argmax(np.cumsum(var) >= 0.95) + 1)
print(f"PCA: PC1+PC2 explican {var[:2].sum():.1%} | componentes para 95%: {n95}")

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].bar(range(1, n + 1), var, label="Individual")
ax[0].plot(range(1, n + 1), np.cumsum(var), "ro-", label="Acumulada")
ax[0].axhline(0.95, ls="--", color="gray")
ax[0].set_xlabel("Componente"); ax[0].set_ylabel("Varianza explicada")
ax[0].set_title("Varianza explicada por PCA"); ax[0].legend()
sc = ax[1].scatter(Z[:, 0], Z[:, 1], c="gray", edgecolor="k", alpha=.8)
ax[1].set_xlabel(f"PC1 ({var[0]:.1%})"); ax[1].set_ylabel(f"PC2 ({var[1]:.1%})")
ax[1].set_title("Proyección PCA 2D (sin colores: PCA no conoce las clases)")
cargas = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
sns.heatmap(cargas, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax[2])
ax[2].set_title("Cargas: peso de cada variable en PC1 y PC2")
plt.tight_layout(); salida("01_pca.png")

# Biplot: muestras + vectores de las variables
plt.figure(figsize=(9, 8))
plt.scatter(Z[:, 0], Z[:, 1], c="lightgray", edgecolor="k", alpha=.7)
for nombre, (a, b) in zip(X.columns, pca.components_[:2].T):
    plt.arrow(0, 0, a * 4, b * 4, color="crimson", head_width=.08)
    plt.text(a * 4.3, b * 4.3, nombre, color="crimson", fontsize=9)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Biplot PCA: variables sobre las muestras")
plt.tight_layout(); salida("02_biplot.png")

# ------------------------------------------------------------------
# 3. K-Means: elegir k (codo + silueta)
# ------------------------------------------------------------------
# Entrada de K-Means: variables estandarizadas, o las 2 primeras componentes de PCA
X_km = Z[:, :2] if MODO == "2_pca" else X_esc
nombres = ["PC1", "PC2"] if MODO == "2_pca" else list(X.columns)
print(f"MODO = {MODO} | K-Means usa: {nombres}")
ks = list(range(2, 9))
inercia, sil = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_km)
    inercia.append(m.inertia_); sil.append(silhouette_score(X_km, m.labels_))
print("Silueta por k:", {k: round(s, 3) for k, s in zip(ks, sil)})

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(ks, inercia, "bo-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia")
ax[0].set_title("Método del codo")
ax[1].plot(ks, sil, "go-"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silueta promedio")
ax[1].set_title("Coeficiente de silueta por k")
plt.tight_layout(); salida("03_eleccion_k.png")

# ------------------------------------------------------------------
# 4. K-Means final (k = 3) y evaluación interna
# ------------------------------------------------------------------
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_km)
cl = km.labels_
sil_k = silhouette_score(X_km, cl)
print(f"\nK-Means k={K}: silueta={sil_k:.3f} | tamaños de cluster={np.bincount(cl)}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for c in range(K):
    ax[0].scatter(Z[cl == c, 0], Z[cl == c, 1], label=f"Cluster {c}", edgecolor="k", alpha=.8)
cent = km.cluster_centers_[:, :2] if MODO == "2_pca" else pca.transform(km.cluster_centers_)
ax[0].scatter(cent[:, 0], cent[:, 1], c="red", marker="X", s=250, edgecolor="k", label="Centroides")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
ax[0].set_title("Clusters de K-Means en el plano PCA")
s_m = silhouette_samples(X_km, cl); y_low = 10
for c in range(K):
    s_c = np.sort(s_m[cl == c]); h = len(s_c)
    ax[1].fill_betweenx(np.arange(y_low, y_low + h), 0, s_c, alpha=.8, label=f"Cluster {c}")
    y_low += h + 10
ax[1].axvline(sil_k, color="red", ls="--")
ax[1].set_xlabel("Coeficiente de silueta"); ax[1].set_title("Silueta por muestra"); ax[1].legend()
plt.tight_layout(); salida("04_kmeans.png")

# ------------------------------------------------------------------
# 5. Interpretación: perfil de cada cluster (qué variables lo caracterizan)
# -----"""
PROGRAMA 2 - APRENDIZAJE NO SUPERVISADO
Dataset: Wine (178 muestras, 13 variables químicas)  -> las etiquetas NO se usan para aprender
Algoritmos: PCA + K-Means (+ clustering jerárquico como apoyo visual)
Las etiquetas reales solo se usan AL FINAL para interpretar/validar.
Salida : gráficas en ./graficas_no_supervisado
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

CARPETA = "graficas_no_supervisado"
sns.set_theme(style="whitegrid")

# ---------------- CONFIGURACIÓN ----------------
# MODO de variables:
#   "completo"    -> las 13 variables originales
#   "2_variables" -> solo 2 variables originales (las de VARIABLES_2)
#   "2_pca"       -> las 13 variables reducidas a 2 componentes principales
MODO = "completo"
VARIABLES_2 = ["flavanoids", "color_intensity"]
MOSTRAR = True     # True: abre una ventana por gráfica (ciérrala para ver la siguiente)
GUARDAR = False    # True: además guarda cada gráfica como PNG en CARPETA
# -----------------------------------------------

if GUARDAR:
    os.makedirs(CARPETA, exist_ok=True)


def salida(nombre):
    """Guarda y/o muestra la figura actual."""
    if GUARDAR:
        plt.savefig(f"{CARPETA}/{nombre}", dpi=150)
    if MOSTRAR:
        plt.show()
    plt.close()

# ------------------------------------------------------------------
# 1. Datos (MISMO dataset que el programa supervisado)
# ------------------------------------------------------------------
wine = load_wine(as_frame=True)
X, y_real = wine.data, wine.target          # y_real SOLO para interpretar al final
clases = list(wine.target_names)
if MODO == "2_variables":
    X = X[VARIABLES_2]
X_esc = StandardScaler().fit_transform(X)   # PCA y K-Means dependen de escalas -> estandarizar

# ------------------------------------------------------------------
# 2. PCA
# ------------------------------------------------------------------
pca = PCA().fit(X_esc)
Z = pca.transform(X_esc)
var = pca.explained_variance_ratio_
n = X.shape[1]
n95 = int(np.argmax(np.cumsum(var) >= 0.95) + 1)
print(f"PCA: PC1+PC2 explican {var[:2].sum():.1%} | componentes para 95%: {n95}")

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].bar(range(1, n + 1), var, label="Individual")
ax[0].plot(range(1, n + 1), np.cumsum(var), "ro-", label="Acumulada")
ax[0].axhline(0.95, ls="--", color="gray")
ax[0].set_xlabel("Componente"); ax[0].set_ylabel("Varianza explicada")
ax[0].set_title("Varianza explicada por PCA"); ax[0].legend()
sc = ax[1].scatter(Z[:, 0], Z[:, 1], c="gray", edgecolor="k", alpha=.8)
ax[1].set_xlabel(f"PC1 ({var[0]:.1%})"); ax[1].set_ylabel(f"PC2 ({var[1]:.1%})")
ax[1].set_title("Proyección PCA 2D (sin colores: PCA no conoce las clases)")
cargas = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
sns.heatmap(cargas, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax[2])
ax[2].set_title("Cargas: peso de cada variable en PC1 y PC2")
plt.tight_layout(); salida("01_pca.png")

# Biplot: muestras + vectores de las variables
plt.figure(figsize=(9, 8))
plt.scatter(Z[:, 0], Z[:, 1], c="lightgray", edgecolor="k", alpha=.7)
for nombre, (a, b) in zip(X.columns, pca.components_[:2].T):
    plt.arrow(0, 0, a * 4, b * 4, color="crimson", head_width=.08)
    plt.text(a * 4.3, b * 4.3, nombre, color="crimson", fontsize=9)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Biplot PCA: variables sobre las muestras")
plt.tight_layout(); salida("02_biplot.png")

# ------------------------------------------------------------------
# 3. K-Means: elegir k (codo + silueta)
# ------------------------------------------------------------------
# Entrada de K-Means: variables estandarizadas, o las 2 primeras componentes de PCA
X_km = Z[:, :2] if MODO == "2_pca" else X_esc
nombres = ["PC1", "PC2"] if MODO == "2_pca" else list(X.columns)
print(f"MODO = {MODO} | K-Means usa: {nombres}")
ks = list(range(2, 9))
inercia, sil = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_km)
    inercia.append(m.inertia_); sil.append(silhouette_score(X_km, m.labels_))
print("Silueta por k:", {k: round(s, 3) for k, s in zip(ks, sil)})

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(ks, inercia, "bo-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia")
ax[0].set_title("Método del codo")
ax[1].plot(ks, sil, "go-"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silueta promedio")
ax[1].set_title("Coeficiente de silueta por k")
plt.tight_layout(); salida("03_eleccion_k.png")

# ------------------------------------------------------------------
# 4. K-Means final (k = 3) y evaluación interna
# ------------------------------------------------------------------
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_km)
cl = km.labels_
sil_k = silhouette_score(X_km, cl)
print(f"\nK-Means k={K}: silueta={sil_k:.3f} | tamaños de cluster={np.bincount(cl)}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for c in range(K):
    ax[0].scatter(Z[cl == c, 0], Z[cl == c, 1], label=f"Cluster {c}", edgecolor="k", alpha=.8)
cent = km.cluster_centers_[:, :2] if MODO == "2_pca" else pca.transform(km.cluster_centers_)
ax[0].scatter(cent[:, 0], cent[:, 1], c="red", marker="X", s=250, edgecolor="k", label="Centroides")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
ax[0].set_title("Clusters de K-Means en el plano PCA")
s_m = silhouette_samples(X_km, cl); y_low = 10
for c in range(K):
    s_c = np.sort(s_m[cl == c]); h = len(s_c)
    ax[1].fill_betweenx(np.arange(y_low, y_low + h), 0, s_c, alpha=.8, label=f"Cluster {c}")
    y_low += h + 10
ax[1].axvline(sil_k, color="red", ls="--")
ax[1].set_xlabel("Coeficiente de silueta"); ax[1].set_title("Silueta por muestra"); ax[1].legend()
plt.tight_layout(); salida("04_kmeans.png")

# ------------------------------------------------------------------
# 5. Interpretación: perfil de cada cluster (qué variables lo caracterizan)
# ------------------------------------------------------------------
perfil = pd.DataFrame(X_km, columns=nombres).groupby(cl).mean().T
plt.figure(figsize=(7, 8))
sns.heatmap(perfil, cmap="vlag", center=0, annot=True, fmt=".1f")
plt.xlabel("Cluster"); plt.title("Perfil de cada cluster (media estandarizada)")
plt.tight_layout(); salida("05_perfil_clusters.png")

# Dendrograma (apoyo visual a la estructura jerárquica)
plt.figure(figsize=(14, 6))
dendrogram(linkage(X_km, "ward"), truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Dendrograma (Ward, 30 últimos nodos)"); plt.ylabel("Distancia")
plt.tight_layout(); salida("06_dendrograma.png")

# ------------------------------------------------------------------
# 6. Validación EXTERNA (opcional): ahora sí se comparan con las clases reales
# ------------------------------------------------------------------
ari = adjusted_rand_score(y_real, cl)
tabla = pd.crosstab(pd.Series(y_real.map(dict(enumerate(clases))), name="Clase real"),
                    pd.Series(cl, name="Cluster"))
print(f"ARI contra clases reales: {ari:.3f}\n{tabla}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for k, nom in enumerate(clases):
    ax[0].scatter(Z[y_real == k, 0], Z[y_real == k, 1], label=nom, edgecolor="k", alpha=.8)
ax[0].set_title("Clases reales sobre PCA (solo para comparar)"); ax[0].legend()
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_title(f"Clase real vs cluster (ARI = {ari:.2f})")
plt.tight_layout(); salida("07_validacion_externa.png")

if GUARDAR:
    print(f"\nGráficas guardadas en ./{CARPETA}")
-------------------------------------------------------------
perfil = pd.DataFrame(X_km, columns=nombres).groupby(cl).mean().T
plt.figure(figsize=(7, 8))
sns.heatmap(perfil, cmap="vlag", center=0, annot=True, fmt=".1f")
plt.xlabel("Cluster"); plt.title("Perfil de cada cluster (media estandarizada)")
plt.tight_layout(); salida("05_perfil_clusters.png")

# Dendrograma (apoyo visual a la estructura jerárquica)
plt.figure(figsize=(14, 6))
dendrogram(linkage(X_km, "ward"), truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Dendrograma (Ward, 30 últimos nodos)"); plt.ylabel("Distancia")
plt.tight_layout(); salida("06_dendrograma.png")

# ------------------------------------------------------------------
# 6. Validación EXTERNA (opcional): ahora sí se comparan con las clases reales
# ------------------------------------------------------------------
ari = adjusted_rand_score(y_real, cl)
tabla = pd.crosstab(pd.Series(y_real.map(dict(enumerate(clases))), name="Clase real"),
                    pd.Series(cl, name="Cluster"))
print(f"ARI contra clases reales: {ari:.3f}\n{tabla}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for k, nom in enumerate(clases):
    ax[0].scatter(Z[y_real == k, 0], Z[y_real == k, 1], label=nom, edgecolor="k", alpha=.8)
ax[0].set_title("Clases reales sobre PCA (solo para comparar)"); ax[0].legend()
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_title(f"Clase real vs cluster (ARI = {ari:.2f})")
plt.tight_layout(); salida("07_validacion_externa.png")

if GUARDAR:
    print(f"\nGráficas guardadas en ./{CARPETA}")
