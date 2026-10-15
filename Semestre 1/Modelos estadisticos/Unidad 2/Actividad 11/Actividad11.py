# Actividad11.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Modelos Estadísticos para el Aprendizaje Automático y Ciencia de Datos
# Germán Godínez Cardoza
# Código Actividad : Explorando distribuciones continuas

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

rng = np.random.default_rng(42)
n = 10_000

# --- Normal: varía la media (sigma fija = 1) ---
normal_mu = {mu: rng.normal(loc=mu, scale=1, size=n) for mu in [0, 2, 5]}

# --- Normal: varía la desviación estándar (media fija = 0) ---
normal_sigma = {s: rng.normal(loc=0, scale=s, size=n) for s in [0.5, 1, 2]}

# --- Exponencial: varía la escala (scale = 1/lambda) ---
expo = {b: rng.exponential(scale=b, size=n) for b in [0.5, 1, 2]}

# --- Gamma: varía la forma k (escala fija = 1) ---
gamma_k = {k: rng.gamma(shape=k, scale=1, size=n) for k in [1, 2, 5]}

# --- Verificación: media y desviación de cada muestra ---
def resumen(nombre, diccionario):
    print(f"\n{nombre}")
    for param, datos in diccionario.items():
        print(f"  parámetro={param}: media={datos.mean():.3f}, desv={datos.std():.3f}")

resumen("Normal (varía media)", normal_mu)
resumen("Normal (varía desviación)", normal_sigma)
resumen("Exponencial (varía escala)", expo)
resumen("Gamma (varía forma k)", gamma_k)


fig, ax = plt.subplots(2, 2, figsize=(13, 9))

def graficar(axis, diccionario, pdf_func, x, etiqueta, titulo, xlabel):
    for param, datos in diccionario.items():
        p = axis.hist(datos, bins=60, density=True, alpha=0.35)
        color = p[2][0].get_facecolor()
        axis.plot(x, pdf_func(x, param), color=color, lw=2,
                  label=f"{etiqueta} = {param}")
    axis.set_title(titulo)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Densidad")
    axis.legend()

# 1) Normal: varía la media
x = np.linspace(-5, 10, 500)
graficar(ax[0, 0], normal_mu,
         lambda x, mu: stats.norm.pdf(x, loc=mu, scale=1),
         x, "μ", "Normal: varía la media (σ = 1)", "x")

# 2) Normal: varía la desviación estándar
x = np.linspace(-8, 8, 500)
graficar(ax[0, 1], normal_sigma,
         lambda x, s: stats.norm.pdf(x, loc=0, scale=s),
         x, "σ", "Normal: varía σ (μ = 0)", "x")

# 3) Exponencial: varía la escala
x = np.linspace(0, 8, 500)
graficar(ax[1, 0], expo,
         lambda x, b: stats.expon.pdf(x, scale=b),
         x, "escala", "Exponencial: varía la escala", "x")

# 4) Gamma: varía la forma k
x = np.linspace(0, 15, 500)
graficar(ax[1, 1], gamma_k,
         lambda x, k: stats.gamma.pdf(x, a=k, scale=1),
         x, "k", "Gamma: varía la forma k (θ = 1)", "x")

plt.tight_layout()
plt.show()

filas = []

def agregar(familia, etiqueta, diccionario, teo_media, teo_desv):
    for param, datos in diccionario.items():
        filas.append({
            "Distribución": familia,
            "Parámetro": f"{etiqueta}={param}",
            "Media": datos.mean(),
            "Desv. est.": datos.std(),
            "Sesgo": stats.skew(datos),
            "CV": datos.std() / abs(datos.mean()) if datos.mean() != 0 else np.nan,
            "Media teórica": teo_media(param),
            "Desv. teórica": teo_desv(param),
        })

agregar("Normal (varía μ)", "μ", normal_mu, lambda m: m, lambda m: 1)
agregar("Normal (varía σ)", "σ", normal_sigma, lambda s: 0, lambda s: s)
agregar("Exponencial", "escala", expo, lambda b: b, lambda b: b)
agregar("Gamma", "k", gamma_k, lambda k: k, lambda k: np.sqrt(k))

tabla = pd.DataFrame(filas).round(3)
print(tabla.to_string(index=False))
