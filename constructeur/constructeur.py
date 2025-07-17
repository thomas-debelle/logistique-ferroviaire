import pandas as pd
import networkx as nx
import json
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import sys
import folium

def extraire_lignes(chaineGeoJson):
    try:
        geojson = json.loads(chaineGeoJson)
        if geojson['type'] == 'LineString':
            return [geojson['coordinates']]
        elif geojson['type'] == 'MultiLineString':
            return geojson['coordinates']
        else:
            return []
    except:
        return []



def arrondir_point(point, precision=10):
    return (round(point[0], precision), round(point[1], precision))

def convertir_latlon_xy(lat, lon):
    rayonTerre = 6371000  # mètres
    x = rayonTerre * np.radians(lon) * np.cos(np.radians(lat))
    y = rayonTerre * np.radians(lat)
    return (x, y)

def longueur_chemin_graphe_safe(graphe, source, target):
    try:
        longueur = nx.shortest_path_length(graphe, source=source, target=target)
        return longueur
    except nx.NetworkXNoPath as e:
        return sys.maxsize          # Retourne une très grande valeur
    
def trouver_noeud_proche(graphe, arbreKd, noeud, rayon, eviterNoeudsAccessible=False, limiteAccessibilite=20):
    x, y = convertir_latlon_xy(noeud[0], noeud[1])
    indices = arbreKd.query_ball_point([x, y], r=rayon)

    noeudPlusProche = None
    distanceMinimale = float('inf')
    for idx in indices:
        voisinXY = noeudsXY[idx]
        voisin = correspondanceXYNoeud[voisinXY]
        if voisin == noeud:
            continue
        if eviterNoeudsAccessible and (graphe.has_edge(noeud, voisin) or voisin == noeud or longueur_chemin_graphe_safe(graphe, noeud, voisin) < limiteAccessibilite):
            continue            # Si un arête est déjà présente, le voisin est le noeud, ou le voisin est déjà accessible en un nombre d'arêtes limité, alors le voisin n'est pas éligible au raccordement. 
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

def est_dans_zone(point, latMin, latMax, lonMin, lonMax):
    lat, lon = point
    return latMin <= lat <= latMax and lonMin <= lon <= lonMax



# Paramètres
vitesseParDefaut = 100
rayonRaccordements = 400
cheminFichier = 'constructeur/données/données.csv'
#lonMin = -4.70
#lonMax = -1.169
#latMin = 46.68
#latMax = 49.00
lonMin = -90
lonMax = 90
latMin = -90
latMax = 90

# Chargement des données
df = pd.read_csv(cheminFichier, sep=';')
df['V_MAX'] = pd.to_numeric(df['V_MAX'], errors='coerce')

# Construction du graphe brut
graphe = nx.Graph()
segments = []
libellesSegments = []
for _, ligne in df.iterrows():
    # Extraction de la vitesse
    vMax = ligne['V_MAX']
    if pd.isna(vMax):
        vMax = vitesseParDefaut

    # Extraction et ajout des segments
    lignes = extraire_lignes(ligne['Geo Shape'])
    for lignesCoords in lignes:
        segment = [(lat, lon) for lon, lat in lignesCoords]
        segments.append(segment)
        libellesSegments.append(ligne['LIB_LIGNE'] + f' ({int(vMax)} km/h)')

        for i in range(len(segment) - 1):
            p1 = segment[i]
            p2 = segment[i + 1]
            if not (est_dans_zone(p1, latMin, latMax, lonMin, lonMax) or est_dans_zone(p2, latMin, latMax, lonMin, lonMax)):
                continue
            if p1 != p2:
                dist = geodesic(p1, p2).kilometers
                poids = (dist / vMax) * 60      # Temps de parcours estimé en minutes
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
for segment in segments:
    if len(segment) < 2:
        continue
    for extremite in [segment[0], segment[-1]]:
        if not est_dans_zone(extremite, latMin, latMax, lonMin, lonMax) or len(list(graphe.neighbors(extremite))) > 1:
            continue            # Si un raccordement a déjà été effectué, on ne traite pas cette extrémité
        noeudProche, dist = trouver_noeud_proche(graphe, arbreKd, extremite, rayonRaccordements, eviterNoeudsAccessible=True)
        if noeudProche:
            graphe.add_edge(extremite, noeudProche, weight=0)



def simplifier_graphe(G, iter=100):
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
grapheSimplifie = simplifier_graphe(graphe)
grapheSimplifie = fusionner_clusters(grapheSimplifie, 1000)
#grapheSimplifie = graphe


# ------------------------------------------
# Visualisation (utile si le graphe est trop gros)
# ------------------------------------------
pos = {node: (node[1], node[0]) for node in grapheSimplifie.nodes}
plt.figure(figsize=(10, 8))
nx.draw(grapheSimplifie, pos, node_size=2, edge_color='black', with_labels=False)
plt.title("Graphe simplifié (topologie réduite)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.show()



# ------------------------------------------
# Création de la carte Folium
# ------------------------------------------
map = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="OpenStreetMap")

# Ajout des segments
for s in range(len(segments)):
    segment = segments[s]
    folium.PolyLine(
        segment,
        color="blue",
        weight=2,
        opacity=0.2,
        tooltip=libellesSegments[s]
    ).add_to(map)


# Ajout du graphe simplifié
for u, v, data in grapheSimplifie.edges(data=True):
    coords = [(u[0], u[1]), (v[0], v[1])]  # (lat, lon)
    folium.PolyLine(
        coords,
        color="red",
        weight=2,
        opacity=0.6,
        tooltip=f"Temps de parcours: {round(data['weight'], 2)} min"
    ).add_to(map)

# Sauvegarder la carte
map.save("graphe_ferroviaire.html")
