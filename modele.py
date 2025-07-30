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
from time import time
import sys
from functools import lru_cache
from collections import namedtuple

# ---------------------------------------
# Définition des classes et des types
# ---------------------------------------
class Config:
    def __init__(self):
        self.dossierLogs = 'logs'
        self.fichierLogs = None
        self.nbLogsMax = 10

        self.nbGenerations = 200
        self.taillePopulation = 150
        self.cxpb = 0.85
        self.mutpb = 0.20
        self.pbOpeEtapes = 0.5                          # Probabilité que les étapes soient sujettes à des mutations ou des croisements
        # self.pbOpeMvts = 1 - self.pbOpeEtapes           # Probabilité que les mouvements soient sujettes à des mutations ou des croisements

        self.cheminGraphe = 'graphe_ferroviaire.graphml'
        self.nbMvtParMot = 5                            # Nom de mouvements max par lot dans une solution

        self.lambdaTempsAttente = 3.0                   # Paramètre lambda de la loi exponentielle utilisée pour générer les temps d'attente dans les mutations
        self.ecartementMinimal = 8                      # Ecartement minimal (en min) entre deux trains qui se suivent
        self.tempsAttelage = 10                         # Temps de manoeuvre pour l'attelage
        self.tempsDesattelage = 5                       # Temps de manoeuvre pour le désattelage
        self.horizonTemp = 500                          # Horizon temporel de la simulation

        self.coutMax = 10000                            # Coût maximum appliqué par défaut aux objectifs
        self.coeffDeplacements = 1.0                    # Coefficient appliqué à l'objectif de déplacement
        self.coeffRetard = 1.0                          # Coefficient appliqué aux coûts de retard à l'arrivée
        self.coutLotNonLivre = 1000                     # Coût de base d'un lot non livré
        self.coeffDepassementCapaMotrice = 1.0          # Coefficient appliqué aux coûts pour les dépassements de capacité des motrices.



class Motrice:
    def __init__(self, index: int, noeudOrigine: int, capacite=50):
        self.index = index
        self.noeudOrigine = noeudOrigine
        self.capacite = capacite

    def __str__(self):
        return f'M{self.index}'
    
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
    def __init__(self, index: int, noeudOrigine: int, noeudDest: int, debutCommande: int, finCommande: int, taille=1):
        """
        index: identifiant unique du lot.
        debutCommande: instant à partir duquel le lot est disponible pour être transporté.
        finCommande: instant à partir duquel la commande est considérée comme en retard.
        taille: nombre de wagons du lot.
        """
        self.index = index
        self.noeudOrigine = noeudOrigine
        self.noeudDest = noeudDest
        self.debutCommande = debutCommande
        self.finCommande = finCommande
        self.taille = taille

    def __str__(self):
        return f'L{self.index}'
    
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
    def __init__(self, graphe: nx.Graph, motrices: list, lots: list):
        self.graphe = graphe
        self.motrices = motrices
        self.lots = lots
        self.blocages = []              # Liste des sillons bloqués par défaut (par exemple, par d'autres compagnies ferroviaires)

        indexMotrices = set()
        indexLots = set()
        for mot in self.motrices:
            if not mot.index in indexMotrices:
                indexMotrices.add(mot.index)
            else:
                error(f"Index en double dans les motrices: {mot.index}")
        for lot in self.lots:
            if not lot.index in indexLots:
                indexLots.add(lot.index)
            else:
                error(f"Index en double dans les lots: {lot.index}")

class TypeMouvement(Enum):
    Recuperer = 'Réc.'
    Deposer = 'Dép.'
    Attendre = 'Att.'

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Solution:
    def __init__(self, objs, tracageMots: Dict[Motrice, List[int]], tracageLots: Dict[Lot, List[int]], tracageAttelages: Dict[Lot, Motrice]):
        self.objs = objs
        self.tracageMots = tracageMots
        self.tracageLots = tracageLots
        self.tracageAttelages = tracageAttelages

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

def clear_dict(d: dict, val=0):
    d = {k: val for k in d}

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

import heapq

def dijkstra_temporel(graphe, origine, dest, instant=0, filtre=None):
    queue = [(instant, origine)]
    visites = {}
    parents = {origine: (None, instant)}

    while queue:
        temps, noeud = heapq.heappop(queue)

        if noeud in visites and temps >= visites[noeud]:
            continue
        visites[noeud] = temps

        if noeud == dest:
            chemin = []
            while noeud is not None:
                parent, t = parents[noeud]
                chemin.append((noeud, t))
                noeud = parent
            return list(reversed(chemin))

        for succ, data in graphe[noeud].items():
            poids = data['weight']
            if not filtre or filtre(noeud, succ, temps):
                nouveau_temps = temps + poids
                if succ not in visites or nouveau_temps < visites.get(succ, float('inf')):
                    parents[succ] = (noeud, nouveau_temps)
                    heapq.heappush(queue, (nouveau_temps, succ))


    return None                     # Aucun chemin trouvé


def resoudre_probleme(config: Config, probleme: Probleme):
    """
    Cherche une solution pour résoudre l'exploitation.
    """
    noeudsCapa = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds avec une capacité de stockage, et pouvant être utilisés comme cible pour une étape de lot ou un mouvement de motrice

    class Individu:
        def __init__(self):
            self.mvtsMotrices = {}
            for mot in probleme.motrices:
                self.mvtsMotrices[mot] = []      # Pas d'utilisation de dict.fromkeys, car provoque des erreurs de hash.
            self.etapesLots = {}
            for lot in probleme.lots:
                self.etapesLots[lot] = []

    def generer_individu():
        """
        Génère un individu initial. Chaque individu est un tableau de tableau, qui continent
        """
        ind = creator.Individual()
        for lot in probleme.lots:
            ind.etapesLots[lot] = [lot.noeudOrigine, lot.noeudDest]
            mot = random.choice(probleme.motrices)           # Sélection de motrices aléatoires pour amener directement les lots à destination
            ind.mvtsMotrices[mot].append({'type': TypeMouvement.Recuperer, 'lot': lot, 'etape': 0})      # Les mouvements sont exécutés dans l'ordre, lorsque le lot atteint son étape 0 (c'est à dire, dès le départ).
            ind.mvtsMotrices[mot].append({'type': TypeMouvement.Deposer, 'lot': lot, 'etape': 1})
        return ind

    def reparer_individu(ind):
        return ind
    
    def muter_individu(ind: Individu):
        tirageMutEtapes = random.uniform(0, 1)
        # Mutation des étapes
        if tirageMutEtapes <= config.pbOpeEtapes:
            lot = random.choice(probleme.lots)
            mutSupEtape = random.uniform(0, 1) <= 0.5 and len(ind.etapesLots[lot]) > 2
            # Suppression d'un étape
            if mutSupEtape:  # Suppression uniquement si d'autres étapes que origine et dest
                etape = random.randint(0, len(ind.etapesLots[lot]) - 2) + 1       # Sélection d'un étape entre les étapes origine et dest
                ind.etapesLots[lot].pop(etape)
            # Ajout d'une étape
            else:
                etape = random.randint(0, len(ind.etapesLots[lot]) - 2) + 1
                noeudCible = random.choice(noeudsCapa)      # Sélection aléatoire d'un noeud éligible
                ind.etapesLots[lot].insert(etape, noeudCible)
            # TODO: attribuer l'étape à une motrice en créant un mouvement (sinon, la simulation sera bloquée)
            # TODO: pour chaque motrice, s'assurer que les mouvements portant sur un même lot suivent la chronologie des étapes
            for mot in probleme.motrices:
                for i, mvt in enumerate(ind.mvtsMotrices[mot]):
                    lotMvt = mvt.get('lot', None) 
                    # Si suppression d'une étape
                    if mutSupEtape and lotMvt == lot and mvt['etape'] == etape:
                        ind.mvtsMotrices[mot].pop(i)    # Suppression du mouvement
                    # Si ajout d'une étape
                    elif not mutSupEtape and lotMvt == lot and mvt['etape'] >= etape:
                        mvt['etape'] += 1               # Incrémentation des étapes

        # Mutation des mouvements
        else:
            pass


        return reparer_individu(ind),

    def croiser_individus(ind1, ind2):
        return ind1, ind2

    def evaluer_individu(ind):
        sol = simuler_individu(ind)
        return sol.objs
    
    def simuler_individu(ind, tracage=False):
        """
        tracage: si à False, les positions successives des lots et motrices ne sont pas tracées dans la solution.
        """
        # -------------------------------------------------
        # Collections et variables pour la simulation
        # -------------------------------------------------
        derniersNoeudsMots = dict()                                 # Dernier noeud visité par chaque motrice
        derniersNoeudsLots = dict()                                 # Dernier noeud visité par chaque lot
        attelages = dict()                                          # Attelage associant une motrice à chaque lot

        indicesMvtsActuels = dict.fromkeys(probleme.motrices, 0)    # Indice du mouvement actuel pour chaque motrice
        indicesEtapesActuelles = dict.fromkeys(probleme.lots, 0)    # Indice de l'étape actuelle pour chaque lot
        itinerairesActuels = dict.fromkeys(probleme.motrices, None) # Itinéraires suivis actuellement pour chaque motrice
        blocagesItineraires = []                                    # Blocages générés par les itinéraires

        tempsAttenteMots = dict.fromkeys(probleme.motrices, 0)      # Temps d'attente restant pour chaque motrice
        nbWagonsAtteles = dict.fromkeys(probleme.motrices, 0)       # Nombre de wagons attelés pour chaque motrice

        objCoutsLogistiques = 0         # Objectif de coûts d'exploitation de la solution
        objDeplacement = 0              # Objectif de durée de parcours pour l'ensemble des motrices
        tracageMots = {}
        tracageLots = {}
        tracageAttelages = {}
        lotsValides = dict.fromkeys(probleme.lots, False)

        # Initialisation des derniers noeuds
        for mot in probleme.motrices:
            derniersNoeudsMots[mot] = mot.noeudOrigine
        for lot in probleme.lots:
            derniersNoeudsLots[lot] = lot.noeudOrigine


        # -------------------------------------------------
        # Définition des actions, exécutées à la fin de l'itinéraire des mouvements
        # -------------------------------------------------
        def action_recuperer(mot, lot):
            attelages[lot] = mot
            tempsAttenteMots[mot] += config.tempsAttelage

            # Gestion des dépassements de capacité pour les motrices
            nbWagonsAtteles[mot] += lot.taille
            if nbWagonsAtteles[mot] > mot.capacite:
                objCoutsLogistiques += (nbWagonsAtteles - mot.capacite) * config.coeffDepassementCapaMotrice

        def action_deposer(mot, lot):
            attelages[lot] = None
            derniersNoeudsLots[lot] = derniersNoeudsMots[mot]                    # Mise à jour de la position du lot avant son désattelage
            tempsAttenteMots[mot] += config.tempsDesattelage                     # TODO: bloquer aussi le lot pendant la durée du désattelage
            nbWagonsAtteles[mot] -= lot.taille
            indicesEtapesActuelles[lot] += 1

        def action_attendre(mot, duree):
            tempsAttenteMots[mot] += duree


        # -------------------------------------------------
        # Boucle temporelle
        # -------------------------------------------------
        for t in range(config.horizonTemp):
            # Mise à jour du traçage (si activé)
            if tracage:
                for mot in probleme.motrices:
                    tracageMots[mot][t] = derniersNoeudsMots[mot]
                for lot in probleme.lots:
                    tracageLots[lot][t] = derniersNoeudsLots[lot]
                for lot in probleme.lots:
                    tracageAttelages[lot][t] = attelages.get(lot, None)

            # Simulation des motrices
            for mot in probleme.motrices:
                # Si tous les mouvement ont été traitées, la motrice a terminé son travail
                if indicesMvtsActuels[mot] >= len(ind.mvtsMotrices[mot]):
                    continue
                mvtActuel = ind.mvtsMotrices[mot][indicesMvtsActuels[mot]]

                # Extraction du noeud actuel
                if mvtActuel['type'] == TypeMouvement.Attendre:
                    noeudCible = mvtActuel['noeud']
                elif mvtActuel['type'] == TypeMouvement.Recuperer or mvtActuel['type'] == TypeMouvement.Deposer:
                    lotCible = mvtActuel['lot']
                    noeudCible = ind.etapesLots[lotCible][mvtActuel['etape']]

                # Génération d'un nouvel itinéraire
                if derniersNoeudsMots[mot] != noeudCible and not itinerairesActuels.get(mot, None):
                    def est_arc_disponible(u, v, t):                # Filtre pour Dijkstra
                        # Vérification des blocages du problème
                        for sillon in probleme.blocages:
                            if sillon.est_dans_sillon(u, v, t):
                                return False
                        # Vérification des blocages ajoutés par les itinéraires
                        for sillon in blocagesItineraires:
                            if sillon.est_dans_sillon(u, v, t) and sillon.motrice != mot:
                                return False
                        return True

                    itineraire = dijkstra_temporel(probleme.graphe, derniersNoeudsMots[mot], noeudCible, t, est_arc_disponible)
                    itinerairesActuels[mot] = itineraire

                    # Si aucun itinéraire n'est disponible, la motrice est en attente à cet instant
                    if not itineraire:
                        continue

                    # Blocage des arcs de l'itinéraire
                    for i in range(len(itineraire)-1):
                        u = itineraire[i]               # Extraction des information
                        v = itineraire[i+1]
                        debutParcours = round(u[1])
                        finParcours = round(v[1])

                        blocagesItineraires.append(Sillon(u[0], v[0], debutParcours, debutParcours+config.ecartementMinimal, mot))      # Ajout des sillons bloqués
                        if probleme.graphe.edges[u[0], v[0]]['exploit'] == 'Simple':
                            blocagesItineraires.append(Sillon(v[0], u[0], debutParcours, finParcours+config.ecartementMinimal, mot))
                        pass            # TODO: Traiter les cas où la durée est à 0 (passage immédiatement au mouvement suivant)

                # Calcul de l'avancée sur l'itinéraire actuel
                if itinerairesActuels[mot]:
                    itineraireTermine = True
                    itineraire = itinerairesActuels[mot]
                    for i in range(len(itineraire)):
                        if t < itineraire[i][1]:                # Parcoure tous les noeuds de l'itinéraire jusqu'à trouver un noeud qui n'a pas encore été atteint 
                            itineraireTermine = False
                            derniersNoeudsMots[mot] = itineraire[i-1][0]
                            break

                    # Finalisation de l'itinéraire
                    if itineraireTermine:
                        derniersNoeudsMots[mot] = itineraire[-1][0]
                        dureeItineraire = round(itinerairesActuels[mot][-1][1] - itinerairesActuels[mot][0][1])     # Calcule la durée totale du parcours avant de supprimer l'itinéraire
                        itinerairesActuels[mot] = None                                                              # Neutralisation de l'itinéraire
                        objDeplacement += dureeItineraire * config.coeffDeplacements                                # Application de la pénalité de déplacement

                # Application de l'action liée au mouvement
                if derniersNoeudsMots[mot] == noeudCible:
                    actionValide = False            # Pour les mouvements dépendants des lots : vérifie que le lot est bien à l'étape actuelle
                    if mvtActuel['type'] == TypeMouvement.Recuperer and indicesEtapesActuelles[mvtActuel['lot']] == mvtActuel['etape']:
                        action_recuperer(mot, mvtActuel['lot'])
                        actionValide = True
                    elif mvtActuel['type'] == TypeMouvement.Deposer and indicesEtapesActuelles[mvtActuel['lot']] == mvtActuel['etape']:
                        action_deposer(mot, mvtActuel['lot'])
                        actionValide = True
                    elif mvtActuel['type'] == TypeMouvement.Attendre:
                        action_attendre(mot, mvtActuel['duree'])
                        actionValide = True

                    # Passage au mouvement suivant
                    if actionValide:
                        indicesMvtsActuels[mot] += 1

            # Simulation des lots
            for lot in probleme.lots:
                # Mise à jour de la position de la motrice
                if attelages.get(lot, None):
                    derniersNoeudsLots[lot] = derniersNoeudsMots[attelages[lot]]

                # Si le lot n'a pas encore atteint sa destination, pas de traitements supplémentaires
                if attelages.get(lot, None) or derniersNoeudsLots[lot] != lot.noeudDest:
                    continue

                # Application des pénalités de retard
                if t >= lot.finCommande and not lotsValides[lot]:
                    objCoutsLogistiques += (t - lot.finCommande) * config.coeffRetard           # La pénalité correspond au nombre d'instants écoulé depuis la fin de la commande
                lotsValides[lot] = True


        # Finalisation de la simulation
        sol = Solution((objDeplacement, objCoutsLogistiques), tracageMots, tracageLots, tracageAttelages)
        return sol

    # Initialisation de l'algorithme génétique
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", Individu, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("individual", generer_individu)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluer_individu)

    toolbox.register("mate", croiser_individus)
    toolbox.register("mutate", muter_individu)
    toolbox.register("select", tools.selNSGA2)

    # Définition des statistiques à collecter à chaque génération
    stats = tools.Statistics(lambda aff: aff.fitness.values)
    stats.register("avg", lambda x: np.round(np.mean(x, axis=0)))
    stats.register("std", lambda x: np.round(np.std(x, axis=0)))
    stats.register("min", lambda x: np.round(np.min(x, axis=0)))
    stats.register("max", lambda x: np.round(np.max(x, axis=0)))

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

    sol = calculer_solution(frontPareto[0])
    return sol

# ---------------------------------------
# Processus principal
# ---------------------------------------
def main():
    # Chargement de la config
    config = Config()
    initialiser_logs(config)

    # Importation du problème
    graphe = importer_graphe(config.cheminGraphe)
    motrices = [Motrice(0, 183), Motrice(1, 261), Motrice(2, 270)]
    lots = [Lot(0, 280, 277, 0, 300), Lot(1, 270, 172, 0, 300), Lot(2, 270, 244, 0, 300)]
    probleme = Probleme(graphe, motrices, lots)

    # Résolution du problème
    resoudre_probleme(config, probleme)
    pass

main()