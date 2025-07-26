import networkx as nx
from enum import Enum
from typing import Dict, List

# ---------------------------------------
# Définition des classes et des types
# ---------------------------------------
class Config:
    def __init__(self):
        self.cheminGraphe = 'graphe_ferroviaire.graphml'

class Motrice:
    def __init__(self, noeudActuel: int, capacite=50):
        self.noeudActuel = noeudActuel
        self.capacite = capacite

class Lot:
    """
    Ensemble de wagons regroupés pour une même commande ou un même client.
    """
    def __init__(self, noeudOrigine: int, noeudDestination: int, taille=1):
        """
        taille: nombre de wagons du lot.
        """
        self.noeudOrigine = noeudOrigine
        self.noeudDestination = noeudDestination
        self.taille = taille
        
class Probleme:
    def __init__(self):
        self.graphe: nx.Graph = None
        self.motrices = []
        self.lots = []
        self.requetes = {}
        self.wagons = {}

class TypeMouvement(Enum):
    Recuperer = "Réc."
    Deposer = "Dép."
    Attendre = "Att."
    Parcourir = "Par."

class Mouvement(Enum):
    def __init__(self, typeMouvement: TypeMouvement, param: int):
        self.typeMouvement = typeMouvement
        self.param = param

    def __str__(self):
        return f'({self.typeMouvement.value} {self.param})'
    
    def __repr__(self):
        return self.__str__()
    
Solution = Dict[int, List[Mouvement]]


# ---------------------------------------
# Fonctions 
# ---------------------------------------
def importer_graphe(cheminGraphe):
    return nx.read_graphml(cheminGraphe)

def extraire_noeud(graphe: nx.Graph, index: int):
    """
    Recherche le noeud correspondant à l'index passé en argument dans le graphe.
    """
    return next((n for n, d in graphe.nodes(data=True) if d.get('index', 0) == index), None)

# ---------------------------------------
# Processus principal
# ---------------------------------------
def main():
    config = Config()
    graphe = importer_graphe(config.cheminGraphe)
    print("Arêtes du graphe importé:", graphe.edges())

main()