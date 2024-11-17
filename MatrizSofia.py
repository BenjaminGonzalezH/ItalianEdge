import pandas as pd
import numpy as np

df = pd.read_csv("matriz_numerada.csv")
df_sin_primera_columna = df.iloc[:, 1:]
matriz = df_sin_primera_columna.values
#matriz_redondeada = np.round(matriz, 2)

for i in range(matriz.shape[0]):
    for j in range(i, matriz.shape[1]):
        if matriz[i, j] != matriz[j, i]:
            matriz[j, i] -= 1

print(matriz)

diferencias = np.where(matriz != matriz.T)

# Emparejar las coordenadas en una lista de tuplas
coordenadas_no_simetricas = list(zip(diferencias[0], diferencias[1]))

# Mostrar las coordenadas no simétricas y sus valores
for (i, j) in coordenadas_no_simetricas:
    print(f"Coordenada ({i}, {j}): X[i, j] = {matriz[i, j]}, X[j, i] = {matriz[j, i]}")
