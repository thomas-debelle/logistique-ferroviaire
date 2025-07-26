import networkx as nx
from enum import Enum
from typing import Dict, List
from deap import base, creator, tools, algorithms
import random
import numpy as np
import logging
from logging import info, warning, error
import os
import datetime
from glob import glob
from math import ceil

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
        self.mutpb = 0.1

        self.cheminGraphe = 'graphe_ferroviaire.graphml'
        self.nbMvtMax = 20                          # Nom de mouvements max par motrice dans une solution
        self.lambdaTempsAttente = 3.0               # Paramètre lambda de la loi exponentielle utilisée pour générer les temps d'attente dans les mutations

class Motrice:
    def __init__(self, noeudActuel: int, capacite=50, libelle='X'):
        self.noeudActuel = noeudActuel
        self.capacite = capacite
        self.libelle = libelle

    def __str__(self):
        return f'(Mot. {self.libelle})'
    
    def __repr__(self):
        return self.__str__()

class Lot:
    """
    Ensemble de wagons regroupés pour une même commande ou un même client.
    """
    def __init__(self, noeudOrigine: int, noeudDestination: int, debutCommande: int, finCommande: int, taille=1):
        """
        taille: nombre de wagons du lot.
        """
        self.noeudOrigine = noeudOrigine
        self.noeudDestination = noeudDestination
        self.debutCommande = debutCommande
        self.finCommande = finCommande
        self.taille = taille
        self.noeudActuel = noeudOrigine

    def __str__(self):
        return f'(Lot {self.noeudOrigine}->{self.noeudDestination})'
    
    def __repr__(self):
        return self.__str__()
        
class Probleme:
    def __init__(self, graphe: nx.Graph):
        self.graphe = graphe
        self.motrices = []
        self.lots = []

class TypeMouvement(Enum):
    Recuperer = 0
    Deposer = 1
    Attendre = 2
    Aller = 3

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Mouvement:
    def __init__(self, typeMouvement: TypeMouvement, param):
        self.typeMouvement = typeMouvement
        self.param = param

    def __str__(self):
        return f'({self.typeMouvement.name}, {self.param})'
    
    def __repr__(self):
        return self.__str__()
    
Solution = Dict[Motrice, List[Mouvement]]

class TypeMutation(Enum):
    Suppression = -1
    Ajout = +1
    Echange = 0

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
    return nx.read_graphml(cheminGraphe)

def extraire_noeud(graphe: nx.Graph, index: int):
    """
    Recherche le noeud correspondant à l'index passé en argument dans le graphe.
    """
    return next((n for n, d in graphe.nodes(data=True) if d.get('index', 0) == index), None)

def resoudre_probleme(config: Config, probleme: Probleme):
    """
    Cherche une solution pour résoudre l'exploitation.
    """
    typesMouvementsListe = list(TypeMouvement)
    typesMutationsListe = list(TypeMutation)
    noeudsCibles = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds pouvant être utilisés comme cible pour un mouvement de déplacement

    def generer_individu():
        """
        Génère aléatoirement une solution sous la forme d'un individu. 
        """
        ind = [None] * len(probleme.motrices) * config.nbMvtMax         # Un individu est une liste concaténée des mouvements successifs de toutes les motrices
        return creator.Individual(ind)
    
    def muter_individu(ind):
        # Sélection d'une motrice
        motNum = random.choice(range(len(probleme.motrices)))
        mot = probleme.motrices[motNum]

        # Comptage des mouvements de la motrice
        posMvtsMot = motNum * config.nbMvtMax                           # Position dans l'individu du premier mouvement rattaché à la motrice
        cmptMvt = 0
        while ind[posMvtsMot + cmptMvt] and cmptMvt < config.nbMvtMax:
            cmptMvt += 1

        # Sélection d'une mutation
        if cmptMvt > 0 and cmptMvt < config.nbMvtMax:
            typeMut = random.choice(typesMutationsListe)
        elif cmptMvt == 0:
            typeMut = TypeMutation.Ajout
        else: # cmptMvt >= config.nbMvtMax:
            typeMut = random.choice([TypeMutation.Echange, TypeMutation.Suppression])

        # Application de la mutation
        typeMut = random.choice(typesMutationsListe) if cmptMvt > 0 else TypeMutation.Ajout
        if typeMut == TypeMutation.Ajout:
            # Construction d'un mouvement
            typeMvt = random.choice(typesMouvementsListe)
            if typeMvt == TypeMouvement.Recuperer:
                lot = random.choice(probleme.lots)
                ind[posMvtsMot + cmptMvt] = Mouvement(typeMvt, lot)         # Ignoré si le lot n'est pas sur le même noeud que la motrice lors de l'évaluation du mouvement
            elif typeMvt == TypeMouvement.Deposer:
                lot = random.choice(probleme.lots)
                ind[posMvtsMot + cmptMvt] = Mouvement(typeMvt, lot)         # Ignoré si le lot n'est pas attaché à la motrice lors de l'évaluation du mouvement
            elif typeMvt == TypeMouvement.Attendre:
                duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))
                ind[posMvtsMot + cmptMvt] = Mouvement(typeMvt, duree)
            else: # typeMvt == TypeMouvement.Aller
                cible = random.choice(noeudsCibles)
                ind[posMvtsMot + cmptMvt] = Mouvement(typeMvt, cible)
        elif typeMut == TypeMutation.Suppression:
            # Suppression d'un mouvement
            pos = random.choice(range(cmptMvt)) + posMvtsMot
            ind[pos] = None
        elif typeMut == TypeMutation.Echange:
            pos1, pos2 = random.choice(range(cmptMvt)) + posMvtsMot, random.choice(range(cmptMvt)) + posMvtsMot     # Sélection de deux positions à échanger
            ind[pos1], ind[pos2] = ind[pos2], ind[pos1]

        return ind,     # TODO: réparer l'individu. Rassembler les temps d'attente. Décaler les mouvements pour éviter les None
    
    def croiser_individus(ind1, ind2):
        return ind1, ind2

    def evaluer_individu(ind):
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
    probleme.motrices = [Motrice(183)]
    probleme.lots = [Lot(222, 277, 0, 1000)]

    # Résolution du problème
    resoudre_probleme(config, probleme)
    pass

main()