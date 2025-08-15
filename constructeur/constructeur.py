import pandas as pd
import networkx as nx
import json
from math import atan2, radians, degrees, sin, cos
from geopy.distance import geodesic
from scipy.spatial import KDTree
import numpy as np
import sys
import folium
from folium.plugins import MousePosition
from itertools import combinations
from collections import deque


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
rayonRaccordementsStockages = 100       # Rayon de recherche pour raccorder les noeuds de stockage (ITE, Chantiers, etc).
rayonRaccordementsLignes = 500          # Rayon de recherche pour raccorder les extrémités des lignes. Les angles sont également pris en compte pour le raccordement.
angleVirageMax = 35             # Angle de virage maximum autorisé. Utilisé comme contrainte pour les raccordements et pour générer les transitions de rebroussement.

statutsLigneAutorises = ['Exploitée', 'Transférée en voie de service']      # Statut des lignes pouvant être ajoutées au graphe
exploitationsDoubles = ['Double voie', 'Voie banalisée']                    # Type d'exploitation en voies doubles
nbCarIndex = 6                  # Nombre de caractères pour formatter l'affichage de l'index des noeuds
capacitesParType = {            # Capacité par défaut, en nombre d'éléments (wagons + motrices) pour chaque type de noeud
    'Gare Fret': 50,
    'Chantier': 300,
    'ITE': 50,
    'Triage': 1000
}           # A l'avenir, permettre à l'utilisateur de configurer noeud par noeud les capacités
arrondirDurees = True

# -------------------------------------
# Exemples de cartes
# -------------------------------------
# Coordonnées globales
#lonMin = -90
#lonMax = 90
#latMin = -90
#latMax = 90
# Coordonnées de la Bretagne
#lonMin = -4.70
#lonMax = -1.169
#latMin = 46.68
#latMax = 49.00
#Coordonnées de l'Auvergne-Rhône-Alpes
lonMin = 2.06  
lonMax = 7.68 
latMin = 44.39 
latMax = 46.78
# Coordonnées du Berry
#lonMin = 1.41
#lonMax = 3.30
#latMin = 46.56
#latMax = 47.33
# Coordonnées de Provence-Alpes-Côte-d'Azur
#lonMin = 4.25
#lonMax = 7.68
#latMin = 43.00
#latMax = 44.80
# Coordonnées de Nevers
#lonMin = 3.08
#lonMax = 3.25
#latMin = 46.90
#latMax = 47.05




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

def calculer_azimut(lat1, lon1, lat2, lon2):
    """
    Calcule l'azimut: angle dans le plan horizontal enre la direction d'un objet et un direction de référence.
    """
    # Conversion degrés - radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    d_lon = lon2_rad - lon1_rad
    x = sin(d_lon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(d_lon)
    azimut = degrees(atan2(x, y))
    return (azimut + 360) % 360  # Normalisation 0–360 degrés

def calculer_angle(latSommet, lonSommet, lat1, lon1, lat2, lon2):
    """
    Calcule l'angle entre deux points en utilisant les azimuts.
    latSommet, lonSommet: coordonnées du sommet de l'angle.
    """
    azimut1 = calculer_azimut(latSommet, lonSommet, lat1, lon1)
    azimut2 = calculer_azimut(latSommet, lonSommet, lat2, lon2)

    angle = abs(azimut1 - azimut2) % 360
    angle = min(angle, 360 - angle)         # Angle le plus petit (<= 180°)
    return angle

def longueur_chemin_graphe_safe(graphe, source, target):
    """
    Variante safe pour calculer la longueur d'un chemin. Si aucun chemin n'est trouvé, retourne une très grande valeur.
    """
    try:
        longueur = nx.shortest_path_length(graphe, source=source, target=target)
        return longueur
    except nx.NetworkXNoPath as e:
        return sys.maxsize          # Retourne une très grande valeur
    
def trouver_noeud_proche(graphe, arbreKd, noeud, rayon, noeudsXY, correspondanceXYNoeud, eviterNoeudsAccessible=False, typeNoeud=None, limiteAccessibilite=20, angleMax=None, noeudAdjacent=None):
    x, y = convertir_latlon_xy(noeud[0], noeud[1])
    indices = arbreKd.query_ball_point([x, y], r=rayon)     # Indices des candidats

    noeudPlusProche = None
    distanceMinimale = float('inf')
    for idx in indices:
        candidatXY = noeudsXY[idx]
        candidat = correspondanceXYNoeud[candidatXY]
        if candidat == noeud:
            continue
        if eviterNoeudsAccessible and (graphe.has_edge(noeud, candidat) or candidat == noeud or longueur_chemin_graphe_safe(graphe, noeud, candidat) < limiteAccessibilite):
            continue            # Si un arête est déjà présente, le voisin est le noeud, ou le voisin est déjà accessible en un nombre d'arêtes limité, alors le voisin n'est pas éligible au raccordement. 
        if (not typeNoeud is None) and graphe.nodes[candidat].get("typeNoeud") != typeNoeud:
            continue            # Si le noeud n'est pas du type passé en argument, alors il ne peut pas être retourné
        if angleMax is not None and noeudAdjacent is not None:
            angle = 180 - calculer_angle(
                noeud[0], noeud[1],
                noeudAdjacent[0], noeudAdjacent[1],
                candidat[0], candidat[1]
            )
            if angle > angleMax:
                continue

        dist = geodesic(noeud, candidat).meters
        if dist < distanceMinimale:
            noeudPlusProche = candidat
            distanceMinimale = dist
    return noeudPlusProche, distanceMinimale


def est_dans_zone(point, latMin, latMax, lonMin, lonMax):
    lat, lon = point
    return latMin <= lat <= latMax and lonMin <= lon <= lonMax

def simplifier_graphe(graphe, iter=100):
    # Fonction pour mise à jour des transitions de rebroussement
    def maj_transitions(graphe, noeudCible, ancienVoisin, nouveauVoisin):
        transBloquees = graphe.nodes[noeudCible]['transRebroussement']        # Parcoure les transitions de rebroussement du premier voisin
        for i, (n1, n2) in enumerate(transBloquees):
            if n1 == ancienVoisin:
                transBloquees[i] = tuple(sorted([nouveauVoisin, n2]))
            elif n2 == ancienVoisin:
                transBloquees[i] = tuple(sorted([n1, nouveauVoisin]))       # Nouveau noeud source de la transition


    # Processus principal
    grapheSimplifie = graphe.copy()
    for _ in range(iter):       # Simplification en plusieurs itérations
        noeudsASupprimer = []
        for noeud in list(grapheSimplifie.nodes):
            typeNoeud = grapheSimplifie.nodes[noeud].get("typeNoeud")
            if typeNoeud != 'Croisement':
                continue        # Pas de simplification possible si le noeud n'est pas un croisement

            voisins = list(grapheSimplifie.neighbors(noeud))
            if len(voisins) == 1:
                noeudsASupprimer.append(noeud)      # Suppression des noeuds cul-de-sac qui ne sont pas des noeuds de stockage
            elif len(voisins) == 2:
                v1, v2 = voisins        # Sélection des deux voisins à relier
                if grapheSimplifie.has_edge(v1, v2):
                    continue  # Si une arête existe déjà, pas d'opération à réaliser

                # Récupération des poids et calcul du poids total 
                data1 = grapheSimplifie.get_edge_data(noeud, v1)
                data2 = grapheSimplifie.get_edge_data(noeud, v2)
                poidsTotal = data1['weight'] + data2['weight']
                typeExploit = 'Double' if (data1['exploit'] == 'Double' and data2['exploit'] == 'Double') else 'Simple'

                # Modification des transitions bloquées pour les deux voisins à relier
                maj_transitions(grapheSimplifie, v1, noeud, v2)
                maj_transitions(grapheSimplifie, v2, noeud, v1)      # Suppression de 'noeud' comme intermédiaire entre v1 et v2 dans les transitions

                # Fusion des arêtes
                grapheSimplifie.add_edge(v1, v2, weight=poidsTotal, exploit=typeExploit)
                noeudsASupprimer.append(noeud)

        grapheSimplifie.remove_nodes_from(noeudsASupprimer)

    # Epurage des transitions après suppression de noeuds
    for noeud in grapheSimplifie.nodes:
        transBloquees = grapheSimplifie.nodes[noeud]['transRebroussement']
        transASuppr = []
        for t in transBloquees:
            if (not t[0] in grapheSimplifie.nodes) or (not t[1] in grapheSimplifie.nodes):
                transASuppr.append(t)
        for t in transASuppr:
            transBloquees.remove(t)

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
    
def reindexer_graphe(graphe):
    """
    Parcoure tous les noeuds du graphe indexé par les coordonnées, et leur attribut un index numérique.
    Les coordonnées sont stockées dans un attribut.
    """
    # Réinitialisation de l'index
    idn = 1
    for noeud in graphe.nodes:
        graphe.nodes[noeud]['index'] = idn
        graphe.nodes[noeud]['lat'] = noeud[0]
        graphe.nodes[noeud]['lon'] = noeud[1]
        idn += 1

    # Création du mapping : ancien id -> nouvel id
    mapping = {n: graphe.nodes[n]['index'] for n in graphe.nodes}

    # Application du mapping aux transitions
    for noeud in graphe.nodes:
        transBloquees = graphe.nodes[noeud]['transRebroussement']
        for i, t in enumerate(transBloquees):
            transBloquees[i] = tuple(sorted([mapping[t[0]], mapping[t[1]]]))        # Application du mapping avec tri, pour que les indices soient dans l'ordre croissant dans les transitions bloquées

    # Finalisation de la réindexation
    return nx.relabel_nodes(graphe, mapping)

def rendre_graphe_immuable(graphe):
    for noeud in graphe.nodes:
        transitionsImmuables = json.dumps(graphe.nodes[noeud]['transRebroussement'])
        graphe.nodes[noeud]['transRebroussement'] = transitionsImmuables
    return graphe



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
indexNoeuds = set()             # Collection des index utilisés pour les noeuds identifiés (tous les types autres que noeuds de ligne)

# Création de l'index noeuds
idn = 1

# Ajout des lignes
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

                graphe.add_node(p1, typeNoeud='Croisement', capacite=0, index=idn)
                graphe.add_node(p2, typeNoeud='Croisement', capacite=0, index=idn+1)
                graphe.add_edge(p1, p2, weight=poids, exploit=exploit)
                idn+=2

# Ajout des noeuds de gare
for _, gare in dfGares.iterrows():
    coords = gare['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax) or gare['FRET'] != 'O':
        continue        # Seules les gares de fret sont ajoutées au réseau
    capacite = capacitesParType['Gare Fret']
    graphe.add_node(coords, index=idn, typeNoeud='Gare', libelle=gare['LIBELLE'], capacite=capacite)
    idn += 1

# Ajout des noeuds de triage
for _, triage in dfTriages.iterrows():
    coords = triage['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, index=idn, typeNoeud='Triage', libelle=triage['LIBELLE'], capacite=capacitesParType['Triage'])     # Si une gare et un triage sont à la même position, le triage prend le dessus.
    idn += 1

# Ajout des noeuds d'ITE
for _, ite in dfITE.iterrows():
    coords = ite['C_GEO'].split(',')
    coords = (float(coords[0]), float(coords[1]))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, index=idn, typeNoeud='ITE', libelle=ite['GARE'], capacite=capacitesParType['ITE'])
    idn += 1

# Ajout des noeuds de chantier
for _, chantier in dfChantiers.iterrows():
    coords = (float(chantier['Latitude']), float(chantier['Longitude']))
    if not est_dans_zone(coords, latMin, latMax, lonMin, lonMax):
        continue
    graphe.add_node(coords, index=idn, typeNoeud='Chantier', libelle=chantier['VILLE'].capitalize(), capacite=capacitesParType['Chantier'])
    idn += 1


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
            continue  # L'extrémité n'est pas traitée si un raccordement a déjà été effectué ou si elle n'est pas dans la zone configurée

        # Récupération du point adjacent à l'extrémité pour calculer la direction de la ligne
        if extremite == segment[0]:
            noeudAdjacent = segment[1]
        else:
            noeudAdjacent = segment[-2]

        noeudProche, dist = trouver_noeud_proche(
            graphe, 
            arbreKd, 
            extremite, 
            rayonRaccordementsLignes, 
            noeudsXY, 
            correspondanceXYNoeud, 
            eviterNoeudsAccessible=True, 
            typeNoeud='Croisement',      # On ne relie pas les extrémités à des ITE ou des triages, car la ligne pourrait alors ne pas être raccordée correctement
            angleMax=angleVirageMax,
            noeudAdjacent=noeudAdjacent
        )

        if noeudProche:
            graphe.add_edge(extremite, noeudProche, weight=0, exploit='Double')       # Le raccordement étant théorique, on considère qu'il est de poids nul et à double sens.


# Raccordement des noeuds de stockage à tous les noeuds dans le rayon
for noeud in list(graphe.nodes):
    typeNoeud = graphe.nodes[noeud].get("typeNoeud")
    if typeNoeud != 'Croisement':
        if not est_dans_zone(noeud, latMin, latMax, lonMin, lonMax):
            continue

        x, y = convertir_latlon_xy(noeud[0], noeud[1])
        indices = arbreKd.query_ball_point([x, y], r=rayonRaccordementsStockages)     # Indices des candidats
        for idx in indices:
            candidatXY = noeudsXY[idx]
            candidat = correspondanceXYNoeud[candidatXY]
            graphe.add_edge(noeud, candidat, weight=0, exploit='Double')           # Poids nul car on suppose un raccordement direct


# Construction des transitions bloquées (en fonction des angles d'arrivée et de départ des trains)
for noeud in graphe.nodes:
    graphe.nodes[noeud]['transRebroussement'] = []
    voisins = sorted(list(graphe.neighbors(noeud)))      # Tri lexicographique des coordonnées (pour s'assurer que les combinaisons soient toujours dans le bon ordre)
    if len(voisins) <= 2 or graphe.nodes[noeud]['typeNoeud'] != 'Croisement':
        continue          # Toutes les transitions sont autorisées s'il y a moins de 3 noeuds
    # Sinon, calcul des angles entre toutes les paires de noeuds voisins
    for paireVoisins in list(combinations(voisins, 2)):
        voisin1 = paireVoisins[0]
        voisin2 = paireVoisins[1]
        poidsVoisin1 = graphe.edges[(paireVoisins[0], noeud)]['weight']
        poidsVoisin2 = graphe.edges[(paireVoisins[1], noeud)]['weight']
        angleVirage = 180 - calculer_angle(noeud[0], noeud[1], voisin1[0], voisin1[1], voisin2[0], voisin2[1])      # Calcul l'angle de virage entre les deux noeuds
        if (angleVirage > angleVirageMax
            and (poidsVoisin1 > 0 or (poidsVoisin1 == 0 and graphe.nodes[voisin1]['typeNoeud'] == 'Croisement'))
            and (poidsVoisin2 > 0 or (poidsVoisin2 == 0 and graphe.nodes[voisin2]['typeNoeud'] == 'Croisement'))):    # Les noeuds de stockages raccordés ne peuvent pas être sanctionnés, car les angles ne sont pas pris en compte lors du raccordement.
            graphe.nodes[noeud]['transRebroussement'].append(paireVoisins)               # Blocage des virages avec des angles trop élevés


# ------------------------------------------
# Application des opérations sur le graphe
# ------------------------------------------
graphe = simplifier_graphe(graphe)
graphe = extraire_composante_principale(graphe)
graphe = reindexer_graphe(graphe)
graphe = rendre_graphe_immuable(graphe)     # Rend le graphe immuable (en convertissant les listes et tuples en chaînes de caractères avec json.dumps)


# ------------------------------------------
# Création de la carte Folium
# ------------------------------------------
map = folium.Map(location=[(latMin + latMax) / 2, (lonMin + lonMax) / 2], zoom_start=8, tiles="OpenStreetMap")

# Ajout des segments de lignes sans simplifications (lignes bleues transparentes)
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
    coordsU = (graphe.nodes[u]['lat'], graphe.nodes[u]['lon'])
    coordsV = (graphe.nodes[v]['lat'], graphe.nodes[v]['lon'])
    coords = [coordsU, coordsV]
    exploit = data.get("exploit", 'Double')
    folium.PolyLine(
        coords,
        color="red",
        weight=(4 if exploit == 'Double' else 2),
        opacity=0.6,
        tooltip=f"<b>[{u}, {v}]</b><br><b>Régime d'exploitation</b> : {exploit}<br><b>Temps de parcours</b> : {round(data['weight'], 2)} min"
    ).add_to(map)

# Ajout des nœuds selon leur type
groupeCroisements = folium.FeatureGroup(name='Croisements')
groupeStockages = folium.FeatureGroup(name='Stockages')
for noeud, data in graphe.nodes(data=True):
    # Extraction des données
    lat, lon = data['lat'], data['lon']
    idnStr = str(data.get("index", "")).zfill(nbCarIndex)
    transRebroussement = graphe.nodes[noeud]['transRebroussement']
    capacite = graphe.nodes[noeud]['capacite']
    typeNoeud = data.get("typeNoeud", 'Croisement')

    radius = 3
    if typeNoeud == 'Croisement':
        color = "black"
        radius = 1
    elif typeNoeud == 'Gare':
        color = "green"
        radius = 3
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

    # Création du marqueur
    marqueur = folium.CircleMarker(
        location=(lat, lon),
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=(str(f'<b>[{idnStr}]</b> ' 
                     + typeNoeud + ' - ' + (data.get("libelle", "")) if typeNoeud != 'Croisement' else f'[{idnStr}] Noeud') 
                     + '<br><b>Transitions de rebroussement</b> : ' + (str(transRebroussement) if len(transRebroussement) > 0 else '')
                     + '<br><b>Capacité</b> : ' + str(capacite)
                     + '<br><b>Nb. de voisins</b> : ' + str(len(list(graphe.neighbors(noeud))))
                     + '<br><b>Lat.</b> : ' + str(graphe.nodes[noeud]['lat'])
                     + '<br><b>Lon.</b> : ' + str(graphe.nodes[noeud]['lon']))
    )

    # Ajout du marqueur au bon groupe
    if typeNoeud == 'Croisement':
        marqueur.add_to(groupeCroisements)
    else:
        marqueur.add_to(groupeStockages)


# Ajout des groupes à la map       
groupeCroisements.add_to(map)
groupeStockages.add_to(map)
folium.LayerControl().add_to(map)       # Création d'un contrôle pour les différentes couches de marqueurs
MousePosition().add_to(map)

# Exportation de la carte et du graphe
map.save("graphe_ferroviaire.html")
nx.write_graphml(graphe, "graphe_ferroviaire.graphml")
print(f"Graphe sauvegardé ! -- Nb. de noeuds: {len(graphe.nodes)}")
