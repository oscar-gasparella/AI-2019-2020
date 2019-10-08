import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import ssl

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"

r = requests.get(url)
data = json.loads(r.text)

# Registramos la información de la magnitud de los terremotos
X = list(map(lambda f : f["properties"].get("mag", 0), data["features"]))

# HISTOGRAMA
fig, ax = plt.subplots(figsize=(12, 6))

# Calcular el histograma
n, bins, patches = ax.hist(X, np.arange(0, 10, 0.5), density=0)

# Características de los ejes
ax.set_xlim([0, max(X)])
ax.set_xticks(bins)
ax.set_xlabel('Magnitud del terremoto')
ax.set_ylabel('Frecuencias')
ax.set_title('Histograma de frecuencias')

# Dibujar el gráfico
fig.tight_layout()
plt.show()
