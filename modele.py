import networkx as nx
from enum import Enum
from typing import Dict, List
from deap import base, creator, tools, algorithms
import heapq
import random
import numpy as np
import logging
from logging import info, warning, error
import os
import datetime
from glob import glob
from math import ceil
import sys

# ---------------------------------------
# Définition des classes et des types
# ---------------------------------------
class Config:
    def __init__(self):
        self.dossierLogs = 'logs'
        self.fichierLogs = None
        self.nbLogsMax = 10

        self.nbGenerations = 1000
        self.taillePopulation = 150
        self.cxpb = 0.85
        self.mutpb = 0.15

        self.cheminGraphe = 'graphe_ferroviaire.graphml'
        self.nbMvtParLot = 5                            # Nom de mouvements max par lot dans une solution
        self.lambdaTempsAttente = 3.0                   # Paramètre lambda de la loi exponentielle utilisée pour générer les temps d'attente dans les mutations

        self.horizonTemp = 500                          # Horizon temporel de la simulation



class Motrice:
    def __init__(self, index: int, noeudOrigine: int, capacite=50):
        self.index = index
        self.noeudOrigine = noeudOrigine
        self.capacite = capacite

    def __str__(self):
        return f'(Mot. {self.index})'
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Motrice) and self.index == other.index

    def __hash__(self):
        return hash(self.index)

class Lot:
    """
    Ensemble de wagons regroupés pour une même commande ou un même client.
    """
    def __init__(self, index: int, noeudOrigine: int, noeudDestination: int, debutCommande: int, finCommande: int, taille=1):
        """
        index: identifiant unique du lot.
        debutCommande: instant à partir duquel le lot est disponible pour être transporté.
        finCommande: instant à partir duquel la commande est considérée comme en retard.
        taille: nombre de wagons du lot.
        """
        self.index = index
        self.noeudOrigine = noeudOrigine
        self.noeudDestination = noeudDestination
        self.debutCommande = debutCommande
        self.finCommande = finCommande
        self.taille = taille

    def __str__(self):
        return f'(Lot {self.index})'
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Lot) and self.index == other.index

    def __hash__(self):
        return hash(self.index)
    
class Sillon:
    """
    Représentation simplifiée d'un sillon ferroviaire.
    """
    def __init__(self, noeudDebut, noeudFin, tempsDebut, tempsFin, motrice: Motrice=None):
        """
        Le temps début est inclus, le temps fin est exclus.
        motrice: motrice détentrice du sillon. Laisser à None pour ne pas spécifier.
        """
        self.noeudDebut = noeudDebut
        self.noeudFin = noeudFin
        self.tempsDebut = tempsDebut
        self.tempsFin = tempsFin
        self.motrice = motrice

    def est_dans_sillon(self, u, v, t):
        return (u == self.noeudDebut and v == self.noeudFin and t >= self.tempsDebut and t < self.tempsFin)
    
    def __str__(self):
        return f'({self.noeudDebut}, {self.noeudFin}, {self.tempsDebut}, {self.tempsFin})'
    
    def __repr__(self):
        return self.__str__()
        
class Probleme:
    def __init__(self, graphe: nx.Graph):
        self.graphe = graphe
        self.motrices = []
        self.lots = []
        self.blocages = []              # Liste des sillons bloqués (par exemple, par d'autres compagnies ferroviaires)

class TypeMouvement(Enum):
    Recuperer = 0
    Deposer = 1
    Attendre = 2

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Mouvement:
    def __init__(self, type: TypeMouvement, noeud: int, motrice: Motrice, param=None):
        self.type = type
        self.noeud = noeud
        self.motrice = motrice
        self.param = param

    def __str__(self):
        return f'({self.type.name}, {self.motrice}, {self.noeud}, {self.param})' if self.param else f'({self.type.name}, {self.motrice}, {self.noeud})'
    
    def __repr__(self):
        return self.__str__()
    
Solution = Dict[Motrice, List[Mouvement]]

class TypeMutation(Enum):
    Suppression = -1
    Ajout = +1

# ---------------------------------------
# Fonctions 
# ---------------------------------------
def pause_and_exit():
    info("Appuyez sur une touche pour continuer...")
    input()
    exit()

def initialiser_logs(config: Config):
    """
    Initialise les logs pour permettre leur écriture à la fois dans un console et dans un fichier.
    """
    # Vérification du dossier
    dossierLogs = config.dossierLogs
    if not os.path.isdir(dossierLogs):
        os.mkdir(dossierLogs)

    # Création du fichier et affectation au flux de sortie
    date = datetime.datetime.now()
    nomFichier = date.strftime("log_%Y%m%d_%H%M%S.txt")
    config.fichierLogs = open(dossierLogs + '/' + nomFichier, 'w')

    # Affectation des flux de sortie
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handlerFichier = logging.FileHandler(dossierLogs + '/' + nomFichier)
    handlerFichier.setLevel(logging.INFO)
    handlerConsole = logging.StreamHandler()
    handlerConsole.setLevel(logging.INFO)

    logger.addHandler(handlerFichier)
    logger.addHandler(handlerConsole)

    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s   %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handlerFichier.setFormatter(formatter)
    handlerConsole.setFormatter(formatter)

    info("Logs initialisés !")

    # Suppression des plus anciens logs
    pattern = os.path.join(dossierLogs, 'log_*.txt')
    fichiersLogs = sorted(glob(pattern), key=os.path.getmtime)
    if len(fichiersLogs) > config.nbLogsMax:
        fichiersASupprimer = fichiersLogs[:len(fichiersLogs) - config.nbLogsMax]
        for fichier in fichiersASupprimer:
            os.remove(fichier)
            info(f"Suppression de {fichier}.")

def importer_graphe(cheminGraphe):
    grapheOriginal = nx.read_graphml(cheminGraphe)
    grapheConverti = nx.DiGraph() if grapheOriginal.is_directed() else nx.Graph()       # Recréation du graphe avec identifiants de nœuds convertis en int
    
    # Copie des nœuds
    for noeud in grapheOriginal.nodes:
        grapheConverti.add_node(int(noeud), **grapheOriginal.nodes[noeud])

    # Copie des arêtes avec conversion des extrémités
    for u, v, data in grapheOriginal.edges(data=True):
        grapheConverti.add_edge(int(u), int(v), **data)

    return grapheConverti


def extraire_noeud(graphe: nx.Graph, index: int):
    """
    Recherche le noeud correspondant à l'index passé en argument dans le graphe.
    """
    return next((n for n, d in graphe.nodes(data=True) if d.get('index', 0) == index), None)

def dijkstra_temporel(graphe, origine, dest, instant=0, filtre=None):
    """
    Implémentation personnalité de Djikstra permettant de filtrer les arcs non-disponibles à chaque instant.
    Le filtre est une fonction prenant en paramètre (u, v, t) et retournant True ou False (arc disponible ou non).
    Chaque étape de l'itinéraire retourné est un tuple sous la forme (noeud, instant).
    """
    queue = [(instant, origine, [])]      # File de priorité : (tempsCumul, noeudActuel, chemin)
    visites = {}

    while queue:
        instantActuel, noeudActuel, chemin = heapq.heappop(queue)

        if (noeudActuel in visites) and (instantActuel >= visites[noeudActuel]):
            continue
        visites[noeudActuel] = instantActuel

        chemin = chemin + [(noeudActuel, instantActuel)]

        if noeudActuel == dest:
            return chemin

        for successeur in graphe[noeudActuel]:
            poids = graphe[noeudActuel][successeur]['weight']
            if filtre and filtre(noeudActuel, successeur, instantActuel):
                heapq.heappush(queue, (instantActuel + poids, successeur, chemin))

    return None                     # Aucun chemin trouvé


def resoudre_probleme(config: Config, probleme: Probleme):
    """
    Cherche une solution pour résoudre l'exploitation.
    """
    noeudsCibles = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds pouvant être utilisés comme cible pour un mouvement de déplacement

    def compter_mvt_motrice(ind, lotNum: int):
        """
        Compte les mouvements du lot numéro 'lotNum'.
        """
        cmptMvt = 0
        while ind[lotNum * config.nbMvtParLot + cmptMvt] and cmptMvt < config.nbMvtParLot:
            cmptMvt += 1
        return cmptMvt
    
    def calculer_attelage(ind, lotNum: int, pos: int):
        """
        Détermination de l'attelage du lot lorsqu'on atteint le début du mouvement n° 'pos'. 
        Retourne None si le lot n'est pas attelé à ce mouvement.
        """
        statutAttelage = None
        for numMvt in range(pos):
            mvt = ind[lotNum * config.nbMvtParLot + numMvt]
            if mvt.type == TypeMouvement.Recuperer:
                statutAttelage = mvt.motrice
            elif mvt.type == TypeMouvement.Deposer:
                statutAttelage = None
        return statutAttelage

    def inserer_dans_ind(ind, lotNum, valeur, index):
        """
        Insère une valeur dans un individu en décalant les autres entrées.
        """
        debutMvts = lotNum * config.nbMvtParLot
        for i in range(config.nbMvtParLot - 1, index, -1):
            ind[debutMvts+i] = ind[debutMvts+i-1]
        ind[index] = valeur

    def generer_individu():
        """
        Génère aléatoirement une solution sous la forme d'un individu. 
        """
        ind = [None] * len(probleme.lots) * config.nbMvtParLot      # Un individu est une liste concaténée de suites de mouvements associées à chaque lots
        return creator.Individual(ind)

    def reparer_individu(ind):
        """
        Réorganise les mouvements, regroupe les attentes et supprime les mouvements inutiles.
        """
        for lotNum in range(len(probleme.lots)):
            debutMvts = lotNum * config.nbMvtParLot     # Position dans l'individu du début des mvts rattachés à la motrice

            # Regroupement des attentes
            for pos in range(config.nbMvtParLot-1):
                attente = ind[debutMvts + pos]
                attenteSuivante = ind[debutMvts + pos + 1]
                if (attente and attenteSuivante 
                    and attente.type == TypeMouvement.Attendre 
                    and attenteSuivante.type == TypeMouvement.Attendre
                    and attente.motrice == attenteSuivante.motrice
                    and attente.noeud == attenteSuivante.noeud):
                    ind[debutMvts + pos].param += ind[debutMvts + pos + 1].param
                    ind[debutMvts + pos + 1] = None

            # Suppression des mouvements inutiles
            statutAttelage = None
            for numMvt in range(config.nbMvtParLot):
                mvt = ind[debutMvts + numMvt]
                if not mvt:
                    continue

                if mvt.type == TypeMouvement.Recuperer:
                    if statutAttelage:
                        ind[debutMvts + numMvt] = None      # Mouvement inutile
                    else:
                        statutAttelage = mvt.motrice
                elif mvt.type == TypeMouvement.Deposer:
                    if statutAttelage:
                        statutAttelage = None
                    else:
                        ind[debutMvts + numMvt] = None      # Mouvement inutile

            # Réorganisation des mouvements
            nonNones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParLot] if mvt]
            nones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParLot] if not mvt]
            mvtsLot = nonNones + nones                  # Reconstruction du tableau pour la motrice avec tous les None à la fin
            ind = ind[:debutMvts] + mvtsLot + ind[debutMvts+len(mvtsLot):]       # Insertion du nouvel ordre dans l'individu
        return creator.Individual(ind)
    
    def muter_individu(ind):
        # Sélection d'un lot et comptage des mouvements
        lotNum = random.choice(range(len(probleme.lots)))
        lot = probleme.lots[lotNum]
        debutMvts = lotNum * config.nbMvtParLot         # Position dans l'individu du début des mvts rattachés au lot
        cmptMvt = compter_mvt_motrice(ind, lotNum)

        # Sélection d'une mutation
        if cmptMvt > 0 and cmptMvt < config.nbMvtParLot:
            typeMut = random.choice([TypeMutation.Ajout, TypeMutation.Suppression])
        elif cmptMvt >= config.nbMvtParLot:
            typeMut = TypeMutation.Suppression
        else: # cmptMvt == 0
            typeMut = TypeMutation.Ajout

        # Sélection d'un position pour la mutation
        posMut = random.choice(range(cmptMvt+1)) if typeMut == TypeMutation.Ajout else random.choice(range(cmptMvt))

        # Application de la mutation
        if typeMut == TypeMutation.Ajout:
            # Extraction du mouvement précédent
            mvtPrecedent = ind[debutMvts + posMut - 1] if posMut > 0 else None

            # Sélection du mouvement à ajouter
            attelage = calculer_attelage(ind, lotNum, posMut)
            if attelage:
                typeMvt = random.choice([TypeMouvement.Attendre, TypeMouvement.Deposer])
            else:
                typeMvt = random.choice([TypeMouvement.Attendre, TypeMouvement.Recuperer])

            # Ajout d'un mouvement Attendre
            if typeMvt == TypeMouvement.Attendre:
                noeudCible = random.choice(noeudsCibles)
                motrice = random.choice(probleme.motrices)
                duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))
                nouveauMvt = Mouvement(typeMvt, noeudCible, motrice, duree)
                inserer_dans_ind(ind, lotNum, nouveauMvt, debutMvts + posMut)
            # Ajout d'un mouvement Récupérer
            elif typeMvt == TypeMouvement.Recuperer:
                noeudCible = mvtPrecedent.noeud if mvtPrecedent else lot.noeudOrigine
                motrice = random.choice(probleme.motrices)
                nouveauMvt = Mouvement(typeMvt, noeudCible, motrice)
                inserer_dans_ind(ind, lotNum, nouveauMvt, debutMvts + posMut)
            # Ajout d'un mouvement Déposer
            elif typeMvt == TypeMouvement.Deposer:
                noeudCible = random.choice(noeudsCibles)
                motrice = mvtPrecedent.motrice
                nouveauMvt = Mouvement(typeMvt, noeudCible, motrice)
                inserer_dans_ind(ind, lotNum, nouveauMvt, debutMvts + posMut)
        elif typeMut == TypeMutation.Suppression:
            ind[debutMvts + posMut] = None

        return reparer_individu(ind), 

    def croiser_individus(ind1, ind2):
        enfant1 = [None] * len(probleme.lots) * config.nbMvtParLot
        enfant2 = [None] * len(probleme.lots) * config.nbMvtParLot
        
        # Sélection d'un segment aléatoire dans chaque parent et échange
        for motNum in range(len(probleme.motrices)):
            cmpt1 = compter_mvt_motrice(ind1, motNum)
            cmpt2 = compter_mvt_motrice(ind2, motNum)
            debutMvtsMot = motNum * config.nbMvtParLot

            # Dans le cas où l'un des individus n'inclut qu'un mouvement, le croisement n'est pas possible
            if cmpt1 < 2 or cmpt2 < 2:
                for pos in range(config.nbMvtParLot):
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                continue

            # Sélection du segment
            a1, b1 = sorted(random.sample(range(max(cmpt1, cmpt2)), 2))

            # Echange des deux segments
            for pos in range(config.nbMvtParLot):
                if pos < a1:
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                elif pos >= a1 and pos < b1:
                    enfant1[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                else:   # pos >= b1
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]

        return reparer_individu(enfant1), reparer_individu(enfant2)
    
    def evaluer_individu(ind):
        # Variables pour la simulation
        numMvtActuels = [0] * len(probleme.lots)                            # Indice du mouvement actuel pour chaque lot
        derniersNoeudsLots = [0] * len(probleme.lots)                       # Index du dernier noeud visité pour chaque lot. Lots indexés par leur position dans ind.
        derniersNoeudsMots = dict()                                         # Index du dernier noeud visité pour chaque motrice. Motrices indexées avec leurs références.

        # Vérifie que l'individu n'est pas vide
        if not any(pos is not None for pos in ind):
            return (sys.maxsize, sys.maxsize, sys.maxsize)
        
        # Initialisation des derniers noeuds
        numLot = 0
        for mot in probleme.lots:
            derniersNoeudsLots[numLot] = mot.noeudOrigine
            numLot += 1
        for mot in probleme.motrices:
            derniersNoeudsMots[mot] = mot.noeudOrigine

        # Simulation
        for t in range(config.horizonTemp):
            for numLot in range(len(probleme.lots)):
                # Extraction des informations de lot
                lot = probleme.lots[numLot]

                # Extraction et vérification du mouvement actuel
                numMvtActuel = numMvtActuels[numLot]
                mvtActuel = ind[numLot * config.nbMvtParLot + numMvtActuel]
                if not mvtActuel:
                    continue

                pass            # Sélectionner la motrice
        
        return (0, 0)
            

    # Initialisation de l'algorithme génétique
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("individual", generer_individu)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluer_individu)

    toolbox.register("mate", croiser_individus)
    toolbox.register("mutate", muter_individu)
    toolbox.register("select", tools.selNSGA2)

    # Définition des statistiques à collecter à chaque génération
    stats = tools.Statistics(lambda aff: aff.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nbevals", "avg", "std", "min", "max"]

    # Exécution de l'algorithme
    pop = toolbox.population(n=config.taillePopulation)
    nbGenerations = config.nbGenerations
    cxpb, mutpb = config.cxpb, config.mutpb    # Probabilités de croisement et de mutation

    # Evaluation initiale
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Boucle de l'algorithme NSGA-II
    for gen in range(nbGenerations):
        progeniture = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        for ind in progeniture:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + progeniture, k=len(pop))

        # Affichage des statistiques de la population
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(progeniture), **record)
        
        info(logbook.stream)

    # Extraction du front de pareto
    frontPareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    for ind in frontPareto[:5]:
        info(f"Scores : ({int(ind.fitness.values[0])}, {int(ind.fitness.values[1])})")
    return frontPareto[0]           # TODO: convertir l'individu en solution à la fin de la résolution

# ---------------------------------------
# Processus principal
# ---------------------------------------
def main():
    # Chargement de la config
    config = Config()
    initialiser_logs(config)

    # Importation du problème
    graphe = importer_graphe(config.cheminGraphe)
    probleme = Probleme(graphe)
    probleme.motrices = [Motrice(0, 183)]
    probleme.lots = [Lot(0, 222, 277, 0, 300), Lot(1, 270, 172, 0, 300)]

    # Résolution du problème
    resoudre_probleme(config, probleme)
    pass

main()