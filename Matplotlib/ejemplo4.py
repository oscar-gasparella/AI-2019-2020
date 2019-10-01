import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import ssl

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"

r = requests.get(url)
data = json.loads(r.text)

# Registramos la información de la magnitud de los terremotos
X = []
n = len(data["features"])
for a in range(n):
    X.append(data["features"][a-1]["properties"]["mag"])

# HISTOGRAMA
fig, ax = plt.subplots(figsize=(12, 6))

# Fijar el tamaño de cada intervalo de magnitudes (cada 0.5)
bins = np.arange(0, 10, 0.5)

# Calcular el histograma
n, bins, patches = ax.hist(X, bins, density=0)

# Características de los ejes
ax.set_xlim([0, max(X)])
ax.set_xticks(bins)
ax.set_xlabel('Magnitud del terremoto')
ax.set_ylabel('Frecuencias')
ax.set_title('Histograma de frecuencias')

# Dibujar el gráfico
fig.tight_layout()
plt.show()
