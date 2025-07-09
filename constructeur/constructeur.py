import pandas as pd
import networkx as nx
import json
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Paramètres
vitesseParDefaut = 100
chemin = 'constructeur/données/vitesse-maximale-nominale-sur-ligne.csv'

# Chargement des données
df = pd.read_csv(chemin, sep=';')
df['V_MAX'] = pd.to_numeric(df['V_MAX'], errors='coerce')       # Conversion de V_MAX en nombre

# Extraction des points depuis Geo Shape
def extract_linestring_points(geo_shape_str):
    try:
        geojson = json.loads(geo_shape_str)
        coords = geojson['coordinates']
        return [(lat, lon) for lon, lat in coords]
    except:
        return []

df['lines'] = df['Geo Shape'].apply(extract_linestring_points)

# Arrondi des points pour simplifier le graphe
def round_point(p, precision=5):
    return (round(p[0], precision), round(p[1], precision))

# Construction du graphe
G = nx.Graph()
for _, row in df.iterrows():
    line = row['lines']
    vMax = row['V_MAX']
    if pd.isna(vMax):
        vMax = vitesseParDefaut
    for i in range(len(line) - 1):
        p1 = round_point(line[i])
        p2 = round_point(line[i+1])
        if p1 != p2:
            dist = geodesic(p1, p2).meters
            weight = dist / vMax
            G.add_edge(p1, p2, weight=weight)


def simplify_graph(G):
    G_simplified = G.copy()
    to_remove = []

    for node in list(G.nodes):
        neighbors = list(G_simplified.neighbors(node))
        if len(neighbors) == 2:
            n1, n2 = neighbors
            if G_simplified.has_edge(n1, n2):
                continue  # Éviter les doublons

            # Récupération des poids et calcul du poids total 
            data1 = G_simplified.get_edge_data(node, n1)
            data2 = G_simplified.get_edge_data(node, n2)
            poidsTotal = data1['weight'] + data2['weight']

            # Fusion des arêtes
            G_simplified.add_edge(n1, n2, weight=poidsTotal)
            to_remove.append(node)

    G_simplified.remove_nodes_from(to_remove)
    return G_simplified

# Application de la simplification
G_simplified = simplify_graph(G)

# Visualisation du graphe simplifié
pos = {node: (node[1], node[0]) for node in G_simplified.nodes}
plt.figure(figsize=(10, 8))
nx.draw(G_simplified, pos, node_size=2, edge_color='blue', with_labels=False)
plt.title("Graphe simplifié (topologie réduite)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.show()
