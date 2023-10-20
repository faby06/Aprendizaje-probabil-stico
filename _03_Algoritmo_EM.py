#Algoritmo EM
import numpy as np
from scipy.stats import norm

# Datos de ejemplo
data = np.array([0.9, 1.2, 1.5, 1.8, 2.1])

# Inicializaci�n de los par�metros del modelo
mu = 1.0  # Media inicial
sigma = 1.0  # Desviaci�n est�ndar inicial

# N�mero de iteraciones EM
num_iteraciones = 10

for iteracion in range(num_iteraciones):
    # Expectation Step: Calcular las probabilidades posteriores
    posteriores = norm.pdf(data, loc=mu, scale=sigma)  # Calcular la densidad de probabilidad
    posteriores /= posteriores.sum()  # Normalizar para obtener probabilidades posteriores

    # Maximization Step: Actualizar los par�metros del modelo
    mu = np.sum(posteriores * data) / np.sum(posteriores)  # Actualizar la media
    sigma = np.sqrt(np.sum(posteriores * (data - mu)**2) / np.sum(posteriores))  # Actualizar la desviaci�n est�ndar

    # Imprimir resultados de la iteraci�n actual
    print("Iteracion", iteracion + 1, ": mu =", round(mu, 3), ", sigma =", round(sigma, 3))

# Imprimir los par�metros finales estimados despu�s de todas las iteraciones
print("Resultado final: mu =", round(mu, 3), ", sigma =", round(sigma, 3))