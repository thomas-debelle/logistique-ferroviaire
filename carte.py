import folium
from folium.plugins import MousePosition


def generer_carte(graphe, latMin, latMax, lonMin, lonMax, segments=None, libellesSegments=None, nbCarIndex=6, aretesVisibles=True, couleurAretes='red', supprimerCouleursNoeuds=False):
    # ------------------------------------------
    # Création de la carte Folium
    # ------------------------------------------
    map = folium.Map(location=[(latMin + latMax) / 2, (lonMin + lonMax) / 2], zoom_start=8, tiles="OpenStreetMap")

    # Ajout des segments de lignes brutes (lignes bleues transparentes)
    if segments and libellesSegments:
        for s in range(len(segments)):
            segment = segments[s]
            folium.PolyLine(
                segment,
                color="blue",
                weight=2,
                opacity=0.2,
                tooltip=libellesSegments[s]
            ).add_to(map)

    # Ajout des arêtes
    for u, v, data in graphe.edges(data=True):
        coordsU = (graphe.nodes[u]['lat'], graphe.nodes[u]['lon'])
        coordsV = (graphe.nodes[v]['lat'], graphe.nodes[v]['lon'])
        coords = [coordsU, coordsV]
        exploit = data.get("exploit", 'Double')
        folium.PolyLine(
            coords,
            color=couleurAretes,
            weight=(4 if exploit == 'Double' else 2),
            opacity=0.6 if aretesVisibles else 0.2,
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
            color = "green" if not supprimerCouleursNoeuds else 'black'
            radius = 3
        elif typeNoeud == 'Triage':
            color = "brown" if not supprimerCouleursNoeuds else 'black'
            radius = 5
        elif typeNoeud == 'ITE':
            color = "purple" if not supprimerCouleursNoeuds else 'black'
        elif typeNoeud == 'Chantier':
            color = "blue" if not supprimerCouleursNoeuds else 'black'
            radius = 5
        else:
            color = "gray" if not supprimerCouleursNoeuds else 'black'

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

    return map