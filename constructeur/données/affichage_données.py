import pandas as pd
import matplotlib.pyplot as plt
import json

# Chargement du fichier
df = pd.read_csv('données.csv', sep=';')

# Fonction pour extraire les points du Geo Shape
def extract_linestring_points(geo_shape_str):
    try:
        geojson = json.loads(geo_shape_str)
        coords = geojson['coordinates']
        return [(lat, lon) for lon, lat in coords]  # on inverse (lon, lat) -> (lat, lon)
    except:
        return []

# Extraction des lignes
df['lines'] = df['Geo Shape'].apply(extract_linestring_points)

# Tracé
plt.figure(figsize=(10, 8))
for line in df['lines']:
    if line:
        lats, lons = zip(*line)
        plt.plot(lons, lats, linewidth=1)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("LA FRANCE !")
plt.grid(True)
plt.axis('equal')  # Pour respecter les proportions
plt.show()
