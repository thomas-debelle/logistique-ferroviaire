import networkx as nx

# ---------------------------------------
# Paramètres
# ---------------------------------------
class Params:
    cheminGraphe = 'graphe_ferroviaire.graphml'

# ---------------------------------------
# Fonctions 
# ---------------------------------------
def importer_graphe(cheminGraphe):
    return nx.read_graphml(cheminGraphe)

# ---------------------------------------
# Processus principal
# ---------------------------------------
def main():
    graphe = importer_graphe(Params.cheminGraphe)
    print("Arêtes du graphe importé:", graphe.edges())

main()