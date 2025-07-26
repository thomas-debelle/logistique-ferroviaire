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
        self.nbMvtParMot = 20                          # Nom de mouvements max par motrice dans une solution
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
    def __init__(self, type: TypeMouvement, param):
        self.type = type
        self.param = param

    def __str__(self):
        return f'({self.type.name}, {self.param})'
    
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
        ind = [None] * len(probleme.motrices) * config.nbMvtParMot         # Un individu est une liste concaténée des mouvements successifs de toutes les motrices
        return creator.Individual(ind)

    def compter_mvt_motrice(ind, motNum: int):
        """
        Compte les mouvements de la motrice numéro 'motNum'.
        """
        cmptMvt = 0
        while ind[motNum * config.nbMvtParMot  + cmptMvt] and cmptMvt < config.nbMvtParMot:
            cmptMvt += 1
        return cmptMvt
    
    def reparer_individu(ind):
        """
        Réorganise les mouvements et regroupe les attentes.
        """
        for motNum in range(len(probleme.motrices)):
            debutMvts = motNum * config.nbMvtParMot     # Position dans l'individu du début des mvts rattachés à la motrice

            # Regroupement des attentes
            for pos in range(config.nbMvtParMot-1):
                if (ind[debutMvts + pos] and ind[debutMvts + pos + 1] 
                    and ind[debutMvts + pos].type == TypeMouvement.Attendre 
                    and ind[debutMvts + pos + 1].type == TypeMouvement.Attendre):
                    ind[debutMvts + pos].param += ind[debutMvts + pos + 1].param
                    ind[debutMvts + pos + 1] = None

            # Réorganisation des mouvements
            nonNones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParMot] if mvt]
            nones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParMot] if not mvt]
            mvtsMot = nonNones + nones                  # Reconstruction du tableau pour la motrice avec tous les None à la fin
            ind = ind[:debutMvts] + mvtsMot + ind[debutMvts+len(mvtsMot):]       # Insertion du nouvel ordre dans l'individu
        return creator.Individual(ind)

    def muter_individu(ind):
        # Sélection d'une motrice et comptage des mouvements
        motNum = random.choice(range(len(probleme.motrices)))
        debutMvts = motNum * config.nbMvtParMot         # Position dans l'individu du début des mvts rattachés à la motrice
        cmptMvt = compter_mvt_motrice(ind, motNum)

        # Sélection d'une mutation
        if cmptMvt > 0 and cmptMvt < config.nbMvtParMot:
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
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, lot)         # Ignoré si le lot n'est pas sur le même noeud que la motrice lors de l'évaluation du mouvement
            elif typeMvt == TypeMouvement.Deposer:
                lot = random.choice(probleme.lots)
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, lot)         # Ignoré si le lot n'est pas attaché à la motrice lors de l'évaluation du mouvement
            elif typeMvt == TypeMouvement.Attendre:
                duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, duree)
            else: # typeMvt == TypeMouvement.Aller
                cible = random.choice(noeudsCibles)
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, cible)
        elif typeMut == TypeMutation.Suppression:
            # Suppression d'un mouvement
            pos = random.choice(range(cmptMvt)) + debutMvts
            ind[pos] = None
        elif typeMut == TypeMutation.Echange:
            pos1, pos2 = random.choice(range(cmptMvt)) + debutMvts, random.choice(range(cmptMvt)) + debutMvts     # Sélection de deux positions à échanger
            ind[pos1], ind[pos2] = ind[pos2], ind[pos1]

        return reparer_individu(ind), 
    
    def croiser_individus(ind1, ind2):
        enfant1 = [None] * len(probleme.motrices) * config.nbMvtParMot
        enfant2 = [None] * len(probleme.motrices) * config.nbMvtParMot
        
        # Sélection d'un segment aléatoire dans chaque parent et échange
        for motNum in range(len(probleme.motrices)):
            cmpt1 = compter_mvt_motrice(ind1, motNum)
            cmpt2 = compter_mvt_motrice(ind2, motNum)
            debutMvtsMot = motNum * config.nbMvtParMot

            # Dans le cas où l'un des individus n'inclut qu'un mouvement, le croisement n'est pas possible
            if cmpt1 < 2 or cmpt2 < 2:
                for pos in range(config.nbMvtParMot):
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                continue

            # Sélection du segment
            a1, b1 = sorted(random.sample(range(max(cmpt1, cmpt2)), 2))

            # Echange des deux segments
            for pos in range(config.nbMvtParMot):
                if pos < a1:
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                elif pos >= a1 and pos < b1:
                    enfant1[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                else:   # pos >= b1
                    enfant1[debutMvtsMot + pos] = ind1[debutMvtsMot + pos]
                    enfant2[debutMvtsMot + pos] = ind2[debutMvtsMot + pos]

        return reparer_individu(ind1), reparer_individu(ind2)       # TODO: réparer l'individu

    def evaluer_individu(ind):
        return (100, 100)

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