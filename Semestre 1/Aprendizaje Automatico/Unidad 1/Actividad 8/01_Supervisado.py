# 01_Supervisado.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Aprendizaje Automático
# Germán Godínez Cardoza
# Código Actividad 1: 
# Entrenar un modelo supervisado simple (clasificación).
# Código desarrollado usando Claude de Anthropic con modelo Sonnet 5

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

CARPETA = "graficas_supervisado"
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
# 1. Datos (MISMO dataset y misma semilla que el programa no supervisado)
# ------------------------------------------------------------------
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
clases = list(wine.target_names)
if MODO == "2_variables":
    X = X[VARIABLES_2]
nombres = ["PC1", "PC2"] if MODO == "2_pca" else list(X.columns)
print(f"MODO = {MODO} | variables de entrada al modelo: {nombres}\n")


def preproc():
    """Pasos de preprocesamiento (instancias nuevas cada vez)."""
    return [StandardScaler()] + ([PCA(n_components=2)] if MODO == "2_pca" else [])


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

prep = make_pipeline(*preproc()).fit(X_tr)   # se ajusta SOLO con entrenamiento
X_tr_e, X_te_e = prep.transform(X_tr), prep.transform(X_te)

# ------------------------------------------------------------------
# 2. Entrenamiento
# ------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr_e, y_tr)
lr = LogisticRegression(max_iter=1000).fit(X_tr_e, y_tr)

y_pred = rf.predict(X_te_e)
print(f"Random Forest - exactitud en prueba: {accuracy_score(y_te, y_pred):.3f}")
print(f"Reg. Logística - exactitud en prueba: {accuracy_score(y_te, lr.predict(X_te_e)):.3f}\n")
print(classification_report(y_te, y_pred, target_names=clases))

# Validación cruzada (más confiable que una sola partición)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_rf = cross_val_score(make_pipeline(*preproc(), RandomForestClassifier(200, random_state=42)), X, y, cv=cv)
cv_lr = cross_val_score(make_pipeline(*preproc(), LogisticRegression(max_iter=1000)), X, y, cv=cv)
print(f"Validación cruzada 5-fold  RF: {cv_rf.mean():.3f} ± {cv_rf.std():.3f} | "
      f"LR: {cv_lr.mean():.3f} ± {cv_lr.std():.3f}")

# ------------------------------------------------------------------
# 3. Gráfica 1: matriz de confusión + importancia de variables
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=clases).plot(
    ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Matriz de confusión (conjunto de prueba)")
imp = pd.Series(rf.feature_importances_, index=nombres).sort_values()
imp.plot.barh(ax=ax[1], color="steelblue")
ax[1].set_title("Importancia de variables (Random Forest)")
plt.tight_layout(); salida("01_confusion_importancia.png")

# ------------------------------------------------------------------
# 4. Gráfica 2: probabilidades predichas + comparación de modelos por CV
# ------------------------------------------------------------------
proba = rf.predict_proba(X_te_e)
confianza = proba.max(axis=1)
correcto = y_pred == y_te.values
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].scatter(range(len(confianza)), confianza, c=np.where(correcto, "seagreen", "red"),
              edgecolor="k")
ax[0].axhline(1 / 3, ls="--", color="gray", label="Azar (1/3)")
ax[0].set_xlabel("Muestra de prueba"); ax[0].set_ylabel("Probabilidad de la clase predicha")
ax[0].set_title("Confianza del modelo (verde = acierto, rojo = error)"); ax[0].legend()
ax[1].boxplot([cv_rf, cv_lr], tick_labels=["Random Forest", "Reg. Logística"])
ax[1].set_ylabel("Exactitud"); ax[1].set_title("Validación cruzada 5-fold")
plt.tight_layout(); salida("02_confianza_validacion.png")

# ------------------------------------------------------------------
# 5. Gráfica 3: curva de aprendizaje
# ------------------------------------------------------------------
tam, tr_sc, va_sc = learning_curve(
    make_pipeline(*preproc(), RandomForestClassifier(200, random_state=42)), X, y, cv=cv,
    train_sizes=np.linspace(0.2, 1.0, 6), random_state=42)
plt.figure(figsize=(8, 6))
plt.plot(tam, tr_sc.mean(1), "o-", label="Entrenamiento")
plt.plot(tam, va_sc.mean(1), "o-", label="Validación")
plt.fill_between(tam, va_sc.mean(1) - va_sc.std(1), va_sc.mean(1) + va_sc.std(1), alpha=.2)
plt.xlabel("Tamaño del conjunto de entrenamiento"); plt.ylabel("Exactitud")
plt.title("Curva de aprendizaje"); plt.legend()
plt.tight_layout(); salida("03_curva_aprendizaje.png")

# ------------------------------------------------------------------
# 6. Gráfica 4: frontera de decisión en 2D (PCA solo para visualizar)
# ------------------------------------------------------------------
if X_tr_e.shape[1] == 2:                      # ya hay solo 2 dimensiones: se grafican directo
    Z_tr, Z_te, ejes = X_tr_e, X_te_e, [n + " (estandarizada)" if MODO != "2_pca" else n for n in nombres]
else:                                         # 13 variables: PCA solo para poder visualizar
    pca = PCA(n_components=2).fit(X_tr_e)
    Z_tr, Z_te, ejes = pca.transform(X_tr_e), pca.transform(X_te_e), ["PC1", "PC2"]
rf2 = RandomForestClassifier(200, random_state=42).fit(Z_tr, y_tr)
xx, yy = np.meshgrid(np.linspace(Z_tr[:, 0].min() - 1, Z_tr[:, 0].max() + 1, 300),
                     np.linspace(Z_tr[:, 1].min() - 1, Z_tr[:, 1].max() + 1, 300))
zz = rf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 7))
plt.contourf(xx, yy, zz, alpha=.25, cmap="viridis")
for k, nom in enumerate(clases):
    plt.scatter(Z_te[y_te.values == k, 0], Z_te[y_te.values == k, 1], label=nom,
                edgecolor="k", s=70)
plt.title(f"Frontera de decisión (RF sobre 2 dimensiones, "
          f"exactitud 2D = {rf2.score(Z_te, y_te):.2%})")
plt.xlabel(ejes[0]); plt.ylabel(ejes[1]); plt.legend()
plt.tight_layout(); salida("04_frontera_decision.png")

if GUARDAR:
    print(f"\nGráficas guardadas en ./{CARPETA}")
