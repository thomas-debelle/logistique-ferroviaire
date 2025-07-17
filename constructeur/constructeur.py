import pandas as pd
import networkx as nx
import json
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np

def extraire_points(chaineGeoJson):
    try:
        geojson = json.loads(chaineGeoJson)
        coords = geojson['coordinates']
        return [(lat, lon) for lon, lat in coords]
    except:
        return []

def arrondir_point(point, precision=10):
    return (round(point[0], precision), round(point[1], precision))

def convertir_latlon_xy(lat, lon):
    rayonTerre = 6371000  # mètres
    x = rayonTerre * np.radians(lon) * np.cos(np.radians(lat))
    y = rayonTerre * np.radians(lat)
    return (x, y)

def trouver_noeud_proche(graphe, arbreKd, noeud, rayon, eviterNoeudsConnectes=False):
    x, y = convertir_latlon_xy(noeud[0], noeud[1])
    indices = arbreKd.query_ball_point([x, y], r=rayon)

    noeudPlusProche = None
    distanceMinimale = float('inf')
    for idx in indices:
        voisinXY = noeudsXY[idx]
        voisin = correspondanceXYNoeud[voisinXY]
        if voisin == noeud:
            continue
        if eviterNoeudsConnectes and (graphe.has_edge(noeud, voisin) or voisin == noeud):
            continue
        dist = geodesic(noeud, voisin).meters
        if dist < distanceMinimale:
            noeudPlusProche = voisin
            distanceMinimale = dist
    return noeudPlusProche, distanceMinimale

def fusionner_clusters(graphe, rayonFusion):

    def calculer_barycentre(points):
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    noeuds = list(graphe.nodes)
    coordXY = [convertir_latlon_xy(lat, lon) for lat, lon in noeuds]
    arbre = KDTree(coordXY)

    visités = set()
    remplacements = []

    for i, noeud in enumerate(noeuds):
        if noeud in visités:
            continue

        voisins = arbre.query_ball_point(coordXY[i], r=rayonFusion)
        cluster = [noeuds[j] for j in voisins if noeuds[j] not in visités]

        if len(cluster) <= 1:
            continue

        visités.update(cluster)
        pointFusion = calculer_barycentre(cluster)
        remplacements.append((cluster, pointFusion))

    for anciensNoeuds, nouveauNoeud in remplacements:
        voisinsExternes = set()
        for n in anciensNoeuds:
            voisins = list(graphe.neighbors(n))
            for v in voisins:
                if v not in anciensNoeuds:
                    poids = graphe[n][v]['weight']
                    voisinsExternes.add((v, poids))
        graphe.remove_nodes_from(anciensNoeuds)
        for v, poids in voisinsExternes:
            graphe.add_edge(nouveauNoeud, v, weight=poids)

    return graphe




# Paramètres
vitesseParDefaut = 100
rayonRaccordements = 50
cheminFichier = 'constructeur/données/vitesse-maximale-nominale-sur-ligne.csv'

# Chargement des données
df = pd.read_csv(cheminFichier, sep=';')
df['V_MAX'] = pd.to_numeric(df['V_MAX'], errors='coerce')
df['POINTS'] = df['Geo Shape'].apply(extraire_points)

# Construction du graphe brut
graphe = nx.Graph()
for _, ligne in df.iterrows():
    points = ligne['POINTS']
    vMax = ligne['V_MAX']
    if pd.isna(vMax):
        vMax = vitesseParDefaut

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        if p1 != p2:
            dist = geodesic(p1, p2).meters
            poids = dist / vMax
            graphe.add_edge(p1, p2, weight=poids)

# Construction du KDTree
correspondanceXYNoeud = {}
noeudsXY = []

for noeud in graphe.nodes:
    x, y = convertir_latlon_xy(noeud[0], noeud[1])
    noeudsXY.append((x, y))
    correspondanceXYNoeud[(x, y)] = noeud

arbreKd = KDTree(noeudsXY)

# Raccordement des extrémités
for _, ligne in df.iterrows():
    points = ligne['POINTS']
    if len(points) < 2:
        continue
    for extremite in [points[0], points[-1]]:
        if len(list(graphe.neighbors(extremite))) > 1:
            continue            # Si un raccordement a déjà été effectué, on ne traite pas cette extrémité
        noeudProche, dist = trouver_noeud_proche(graphe, arbreKd, extremite, rayonRaccordements, eviterNoeudsConnectes=True)
        if noeudProche:
            graphe.add_edge(extremite, noeudProche, weight=0)



def simplifier_graphe(G, iter=20):
    grapheSimplifie = G.copy()
    noeudsASupprimer = []

    for i in range(iter):
        for node in list(grapheSimplifie.nodes):
            voisins = list(grapheSimplifie.neighbors(node))
            if len(voisins) == 2:
                v1, v2 = voisins
                if grapheSimplifie.has_edge(v1, v2):
                    continue  # Éviter les doublons

                # Récupération des poids et calcul du poids total 
                data1 = grapheSimplifie.get_edge_data(node, v1)
                data2 = grapheSimplifie.get_edge_data(node, v2)
                poidsTotal = data1['weight'] + data2['weight']

                # Fusion des arêtes
                grapheSimplifie.add_edge(v1, v2, weight=poidsTotal)
                noeudsASupprimer.append(node)

        grapheSimplifie.remove_nodes_from(noeudsASupprimer)
    return grapheSimplifie

# Application des simplifications
#grapheSimplifie = fusionner_clusters(graphe, 1000)
#grapheSimplifie = simplifier_graphe(graphe)
grapheSimplifie = graphe

# Visualisation du graphe simplifié
pos = {node: (node[1], node[0]) for node in grapheSimplifie.nodes}
plt.figure(figsize=(10, 8))
nx.draw(grapheSimplifie, pos, node_size=2, edge_color='blue', with_labels=False)
plt.title("Graphe simplifié (topologie réduite)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.show()
