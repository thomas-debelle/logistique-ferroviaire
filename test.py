import folium
from folium.plugins import TimestampedGeoJson
import networkx as nx
import json
from datetime import datetime, timedelta

# Exemple : graph networkx avec positions géographiques
G = nx.Graph()
G.add_node(738, pos=(48.8566, 2.3522))  # Paris
G.add_node(711, pos=(50.8503, 4.3517))  # Bruxelles
G.add_node(714, pos=(51.5074, -0.1278)) # Londres
G.add_node(692, pos=(45.7640, 4.8357))  # Lyon
# Ajoutez tous vos nœuds avec pos=(lat, lon)

# Exemple de solution : chaque liste = trajectoire d’un train
solution = [
    [738, 738, 738, 711, 711, 692],
    [714, 714, 738, 738, 692, 692]
]

# Paramètres temporels
start_time = datetime(2025, 1, 1, 8, 0, 0)  # début à 08h00
time_step = timedelta(minutes=1)

# Construction du GeoJSON
features = []
for train_idx, path in enumerate(solution):
    coords = []
    times = []
    for step, node in enumerate(path):
        lat, lon = G.nodes[node]['pos']
        coords.append([lon, lat])  # Folium utilise (lon, lat)
        t = start_time + step*time_step
        times.append(t.isoformat())
    
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords
        },
        "properties": {
            "times": times,
            "style": {"color": "red" if train_idx == 0 else "blue"},
            "popup": f"Train {train_idx}"
        }
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Carte folium
m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)
TimestampedGeoJson(
    data=geojson,
    period="PT1M",
    add_last_point=True,
    auto_play=False
).add_to(m)

m.save("trains_animation.html")
