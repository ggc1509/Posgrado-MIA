# Act4Tarea3.py
# Maestria en inteligencia artificial
# TecNM campus Tijuana
# Aprendizaje Automático
# Germán Godínez Cardoza
# Código Actividad 3:
# Implementar una red neuronal con PyTorch para XOR.
# Código desarrollado usando Claude de Anthropic con modelo Sonnet 5

import torch
import torch.nn as nn
import torch.optim as optim

# --- Datos de entrenamiento para XOR ---
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# --- Definición del modelo (MLP: 2 -> 2 -> 1) ---
class RedXOR(nn.Module):
    def __init__(self):
        super(RedXOR, self).__init__()
        self.capa_oculta = nn.Linear(2, 2)   # 2 entradas -> 2 neuronas ocultas
        self.activacion_oculta = nn.Sigmoid()
        self.capa_salida = nn.Linear(2, 1)   # 2 ocultas -> 1 salida
        self.activacion_salida = nn.Sigmoid()

    def forward(self, x):
        x = self.activacion_oculta(self.capa_oculta(x))
        x = self.activacion_salida(self.capa_salida(x))
        return x


# --- Instanciar modelo, función de pérdida y optimizador ---
torch.manual_seed(42)  # reproducibilidad
modelo = RedXOR()
criterio = nn.BCELoss()                       # entropía cruzada binaria
optimizador = optim.SGD(modelo.parameters(), lr=0.5)

# --- Entrenamiento ---
epocas = 10000
for epoca in range(epocas):
    # Forward pass
    salida = modelo(X)
    perdida = criterio(salida, y)

    # Backward pass y actualización de pesos
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    if epoca % 1000 == 0:
        print(f"Época {epoca:5d} | Pérdida: {perdida.item():.6f}")

print(f"\nÉpoca final | Pérdida: {perdida.item():.6f}")

# --- Resultados finales ---
print("\n--- Resultados finales ---")
with torch.no_grad():
    predicciones = modelo(X)
    for entrada, esperado, pred in zip(X, y, predicciones):
        clase = 1 if pred.item() >= 0.5 else 0
        print(f"XOR({int(entrada[0])}, {int(entrada[1])}) = {clase} "
              f"(esperado: {int(esperado.item())}, raw: {pred.item():.4f})")
