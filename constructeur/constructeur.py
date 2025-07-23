import pandas as pd
import networkx as nx
import json
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import sys
import folium

# -------------------------------------
# Fonctions
# -------------------------------------
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
    
def trouver_noeud_proche(graphe, arbreKd, noeud, rayon, noeudsXY, correspondanceXYNoeud, eviterNoeudsAccessible=False, typeNoeud=None, limiteAccessibilite=20):
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
        if (not typeNoeud is None) and graphe.nodes[voisin].get("typeNoeud") != typeNoeud:
            continue            # Si le noeud n'est pas du type passé en argument, alors il ne peut pas être retourné
        dist = geodesic(noeud, voisin).meters
        if dist < distanceMinimale:
            noeudPlusProche = voisin
            distanceMinimale = dist
    return noeudPlusProche, distanceMinimale

def est_dans_zone(point, latMin, latMax, lonMin, lonMax):
    lat, lon = point
    return latMin <= lat <= latMax and lonMin <= lon <= lonMax

def simplifier_graphe(G, iter=100):
    grapheSimplifie = G.copy()

    for _ in range(iter):
        noeudsASupprimer = []
        for node in list(grapheSimplifie.nodes):
            type_node = grapheSimplifie.nodes[node].get("typeNoeud")
            if type_node in ['Triage', 'ITE']:
                continue

            voisins = list(grapheSimplifie.neighbors(node))
            if len(voisins) == 2:
                v1, v2 = voisins
                if grapheSimplifie.has_edge(v1, v2):
                    continue  # Éviter les doublons

                # Récupération des poids et calcul du poids total 
                data1 = grapheSimplifie.get_edge_data(node, v1)
                data2 = grapheSimplifie.get_edge_data(node, v2)
                poidsTotal = data1['weight'] + data2['weight']
                typeExploit = 'Double' if (data1['typeExploit'] == 'Double' and data2['typeExploit'] == 'Double') else 'Simple'

                # Fusion des arêtes
                grapheSimplifie.add_edge(v1, v2, weight=poidsTotal, typeExploit=typeExploit)
                noeudsASupprimer.append(node)

        grapheSimplifie.remove_nodes_from(noeudsASupprimer)

    return grapheSimplifie

def extraire_composante_principale(graphe):
    composantes = list(nx.connected_components(graphe))     
    principale = max(composantes, key=len)                  # Cherche la plus grande composante connexe
    return graphe.subgraph(principale).copy()

def coef_vitesse_fret(vMaxNominale):
    """
    Retourne un coefficient appliqué sur la vitesse max nominale de ligne pour connaître la vitesse max du train de fret.
    """
    if 10 <= vMaxNominale <= 80:
        return 1
    elif 90 <= vMaxNominale <= 100:
        return 80 / 90
    elif 110 <= vMaxNominale <= 120:
        return 90 / 110
    elif 130 <= vMaxNominale <= 160:
        return 100 / 130
    else:
        return 120 / 170





# -------------------------------------
# Paramètres
# -------------------------------------
# Paramètres de données
cheminFichierLignes = 'constructeur/données/liste-des-lignes.csv'           # Remarque: le fichier des lignes a été consolidé avec des informations provenant de nombreuses sources.
cheminFichierGares = 'constructeur/données/liste-des-gares.csv'
cheminFichierTriages = 'constructeur/données/liste-des-triages.csv'
cheminFichierITE = 'constructeur/données/liste-des-ite.csv'
cheminFichierChantiers = 'constructeur/données/chantiers-de-transport-combines.csv'

# Paramètres de génération du graphe
vitesseParDefaut = 160          # Par défaut si la vitesse n'est pas renseignée dans les données.
vitesseNominaleMax = 220        # Les lignes dont la vitesse nominale max est supérieure à cette valeur sont exclues du graphe.
rayonRaccordements = 400        # Rayon de recherche pour raccorder les extrémités des lignes.
statutsLigneAutorises = ['Exploitée', 'Transférée en voie de service']      # Statut des lignes pouvant être ajoutées au graphe
exploitationsDoubles = ['Double voie', 'Voie banalisée']                    # Type d'exploitation en voies doubles

#lonMin = -90
#lonMax = 90
#latMin = -90
#latMax = 90
# Coordonnées de la Bretagne (pour les tests)
#lonMin = -4.70         
#lonMax = -1.169
#latMin = 46.68
#latMax = 49.00
# Coordonnées de l'Auvergne-Rhône-Alpes
lonMin = 2.06  
lonMax = 7.68 
latMin = 44.39 
latMax = 46.78 



# -------------------------------------
# Processus principal
# -------------------------------------
# Chargement des données
dfLignes = pd.read_csv(cheminFichierLignes, sep=';')
dfGares = pd.read_csv(cheminFichierGares, sep=';')
dfTriages = pd.read_csv(cheminFichierTriages, sep=';')
dfITE = pd.read_csv(cheminFichierITE, sep=';')
dfChantiers = pd.read_csv(cheminFichierChantiers, sep=';')

# Retraitement des données
dfLignes['V_MAX'] = pd.to_numeric(dfLignes['V_MAX'], errors='coerce')
dfITE['GARE'] = dfITE['GARE'].fillna('')

# Construction des lignes
graphe = nx.Graph()
segments = []
libellesSegments = []
for _, ligne in dfLignes.iterrows():
    # Extraction de la vitesse
    vMax = ligne['V_MAX']
    if pd.isna(vMax):
        vMax = vitesseParDefaut
    vMax = float(vMax)

    # Calcul de la vitesse de circulation effective du train de fret
    vEff = coef_vitesse_fret(vMax) * vMax

    # Extraction et ajout des lignes
    lignes = extraire_lignes(ligne['Geo Shape'])
    for lignesCoords in lignes:
        segment = [(lat, lon) for lon, lat in lignesCoords]
        segments.append(segment)
        libellesSegments.append(f'<b>{ligne['LIB_LIGNE']}</b><br><b>Vitesse max nominale:</b> {int(vMax)} km/h<br><b>Vitesse effective fret:</b> {round(vEff, 2)} km/h')

        for i in range(len(segment) - 1):
            p1 = segment[i]
            p2 = segment[i + 1]         # Toutes les lignes sont ajoutées au tracé des segments
            if (
                not (est_dans_zone(p1, latMin, latMax, lonMin, lonMax) or est_dans_zone(p2, latMin, latMax, lonMin, lonMax))
                or not ligne['STATUT'] in statutsLigneAutorises
                or ligne['LGV'] == 'Oui'
                or ligne['V_MAX'] > vitesseNominaleMax):      # Si la ligne ne correspond pas aux paramètres, alors elle n'est pas ajoutée au graphe
                continue

            if p1 != p2:
                dist = geodesic(p1, p2).kilometers
                poids = (dist / vMax) * 60      # Temps de parcours estimé en minutes
                exploit = 'Double' if ligne['EXPLOIT'] in exploitationsDoubles else 'Simple'

                graphe.add_node(p1, typeNoeud='Ligne', nomNoeud="")
                graphe.add_node(p2, typeNoeud='Ligne')
                graphe.add_edge(p1, p2, weight=poids, typeExploit=exploit)

# Ajout des autres noeuds
for _, gare in dfGares.iterrows():
    coords = gare['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax) or gare['FRET'] != 'O':
        continue
    graphe.add_node(coords, typeNoeud='Gare', libelleNoeud=gare['LIBELLE'])

for _, triage in dfTriages.iterrows():
    coords = triage['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, typeNoeud='Triage', libelleNoeud=triage['LIBELLE'])     # Si une gare et un triage sont à la même position, le triage prend le dessus.

for _, ite in dfITE.iterrows():
    coords = ite['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, typeNoeud='ITE', libelleNoeud=ite['GARE'])

for _, chantier in dfChantiers.iterrows():
    coords = (float(chantier['Latitude']), float(chantier['Longitude']))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, typeNoeud='Chantier', libelleNoeud=chantier['VILLE'])

# Construction du KDTree associé aux noeuds du graphe
correspondanceXYNoeud = {}
noeudsXY = []

for noeud in graphe.nodes:
    x, y = convertir_latlon_xy(noeud[0], noeud[1])
    noeudsXY.append((x, y))
    correspondanceXYNoeud[(x, y)] = noeud

arbreKd = KDTree(noeudsXY)

# Raccordement des extrémités de lignes
for segment in segments:
    if len(segment) < 2:
        continue
    for extremite in [segment[0], segment[-1]]:
        if not graphe.has_node(extremite) or len(list(graphe.neighbors(extremite))) > 1:
            continue            # L'extrémité n'est pas traitée si un raccordement a déjà été effectué ou si elle n'est pas dans la zone configurée
        noeudProche, dist = trouver_noeud_proche(
            graphe, 
            arbreKd, 
            extremite, 
            rayonRaccordements, 
            noeudsXY, 
            correspondanceXYNoeud, 
            eviterNoeudsAccessible=True, 
            typeNoeud='Ligne')          # On ne relie pas les extrémités à des ITE ou des triages, car la ligne pourrait alors ne pas être raccordée correctement
        if noeudProche:
            graphe.add_edge(extremite, noeudProche, weight=0, typeExploit='Double')       # Le raccordement étant théorique, on considère qu'il est de poids nul et à double sens.

# Raccordement des autres noeuds
for noeud in graphe.nodes:
    typeNoeud = graphe.nodes[noeud].get("typeNoeud")
    if typeNoeud != 'Ligne':
        if not est_dans_zone(noeud, latMin, latMax, lonMin, lonMax):
            continue

        noeudProche = 1     # Valeur de base pour rentrer dans la boucle
        while noeudProche:      # Raccordement à tous les noeuds proches qui ne sont pas accessibles
            noeudProche, dist = trouver_noeud_proche(
                graphe,
                arbreKd,
                noeud,
                rayonRaccordements,
                noeudsXY,
                correspondanceXYNoeud,
                eviterNoeudsAccessible=True
            )
            if noeudProche:
                graphe.add_edge(noeud, noeudProche, weight=0, typeExploit='Double')           # Poids nul car on suppose un raccordement direct


# Application des simplifications
graphe = simplifier_graphe(graphe)
graphe = extraire_composante_principale(graphe)



# ------------------------------------------
# Création de la carte Folium
# ------------------------------------------
map = folium.Map(location=[(latMin + latMax) / 2, (lonMin + lonMax) / 2], zoom_start=8, tiles="OpenStreetMap")

# Ajout des segments d'origine (lignes bleues transparentes)
for s in range(len(segments)):
    segment = segments[s]
    folium.PolyLine(
        segment,
        color="blue",
        weight=2,
        opacity=0.2,
        tooltip=libellesSegments[s]
    ).add_to(map)

# Ajout du graphe simplifié (arêtes rouges)
for u, v, data in graphe.edges(data=True):
    coords = [(u[0], u[1]), (v[0], v[1])]
    folium.PolyLine(
        coords,
        color="red",
        weight=2,
        opacity=0.6,
        tooltip=f"<b>Régiment d'exploitation</b> : {data.get("typeExploit", 'Double')}<br><b>Temps de parcours</b> : {round(data['weight'], 2)} min"
    ).add_to(map)

# Ajout des nœuds selon leur type
for noeud, data in graphe.nodes(data=True):
    lat, lon = noeud
    typeNoeud = data.get("typeNoeud", 'Ligne')

    radius = 3
    if typeNoeud == 'Ligne':
        color = "black"
        radius = 1
    elif typeNoeud == 'Gare':
        color = "green"
    elif typeNoeud == 'Triage':
        color = "brown"
        radius = 5
    elif typeNoeud == 'ITE':
        color = "purple"
    elif typeNoeud == 'Chantier':
        color = "blue"
        radius = 5
    else:
        color = "gray"

    folium.CircleMarker(
        location=(lat, lon),
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=str(typeNoeud + ' - ' + data.get("libelleNoeud", ""))
    ).add_to(map)

# Exportation de la carte et du graphe
map.save("graphe_ferroviaire.html")
nx.write_graphml(graphe, "graphe_ferroviaire.graphml")
print("Graphe sauvegardé.")
