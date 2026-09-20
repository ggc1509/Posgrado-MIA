# Act3Tarea1.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Aprendizaje Automático
# Germán Godínez Cardoza
# Código Actividad 1: 
# Implementar un perceptrón para resolver OR.
# Código desarrollado usando Claude de Anthropic con modelo Sonnet 5

import numpy as np

class Perceptron:
    def __init__(self, n_entradas, tasa_aprendizaje=0.1, epocas=20):
        # Inicializamos los pesos (incluyendo el bias) en cero
        self.pesos = np.zeros(n_entradas + 1)  # +1 para el bias
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas

    def funcion_activacion(self, x):
        # Función escalón (step function)
        return 1 if x >= 0 else 0

    def predecir(self, entradas):
        # Producto punto entre pesos y entradas, más el bias
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]
        return self.funcion_activacion(suma)

    def entrenar(self, X, y):
        for epoca in range(self.epocas):
            print(f"\nÉpoca {epoca + 1}")
            errores_totales = 0
            for entradas, etiqueta in zip(X, y):
                prediccion = self.predecir(entradas)
                error = etiqueta - prediccion
                errores_totales += abs(error)

                # Actualización de pesos y bias (regla del perceptrón)
                self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                self.pesos[0] += self.tasa_aprendizaje * error

                print(f"  Entrada: {entradas}, Esperado: {etiqueta}, "
                      f"Predicho: {prediccion}, Pesos: {self.pesos}")

            if errores_totales == 0:
                print(f"\n¡Convergencia alcanzada en la época {epoca + 1}!")
                break


# Datos de entrenamiento para la compuerta OR 
# Se agrega la tabla de verdad correspondiente a la compuerta OR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])  # Salida esperada para OR

# --- Crear y entrenar el perceptrón ---
perceptron = Perceptron(n_entradas=2, tasa_aprendizaje=0.1, epocas=10)
perceptron.entrenar(X, y)

# --- Probar el perceptrón entrenado ---
print("\n--- Resultados finales ---")
for entradas in X:
    resultado = perceptron.predecir(entradas)
    print(f"OR({entradas[0]}, {entradas[1]}) = {resultado}")
