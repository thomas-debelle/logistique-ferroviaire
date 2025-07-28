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

        self.cheminGraphe = 'graphe_ferroviaire.graphml'
        self.nbMvtParLot = 5                            # Nom de mouvements max par lot dans une solution

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

class Mouvement:
    def __init__(self, type: TypeMouvement, noeud: int, motrice: Motrice, lot: Lot, priorite: int=3, param=None):
        """
        lot: lot commanditant le mouvement.
        priorite: plus la valeur est petite, plus le mouvement est prioritaire.
        """
        self.type = type
        self.noeud = noeud
        self.motrice = motrice
        self.lot = lot
        self.priorite = priorite
        self.param = param

    def __str__(self):
        return f'({self.type.value}, {self.motrice}, {self.lot}, N{self.noeud}, P{self.priorite}, {self.param})' if self.param else f'({self.type.value}, {self.motrice}, {self.lot}, N{self.noeud}, P{self.priorite})'
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return (
            isinstance(other, Mouvement) 
            and self.type == other.type 
            and self.noeud == other.noeud 
            and self.motrice == other.motrice 
            and self.lot == other.lot 
            and self.priorite == other.priorite 
            and self.param == other.param)
    
    def __hash__(self):
        return hash((self.type, self.noeud, self.motrice, self.lot, self.priorite, self.param))

class Solution:
    def __init__(self, objs, tracageMots: Dict[Motrice, List[int]], tracageLots: Dict[Lot, List[int]], tracageAttelages: Dict[Lot, Motrice]):
        self.objs = objs
        self.tracageMots = tracageMots
        self.tracageLots = tracageLots
        self.tracageAttelages = tracageAttelages

class TypeMutation(Enum):
    Ajout = 0
    Suppression = 1
    Variation = 2       # Remarque: pas de déplacement, car brise la causalité (de plus, les suppressions et les attentes peuvent déjà être réalisées n'importe où dans l'individu)

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
    noeudsCibles = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds pouvant être utilisés comme cible pour un mouvement de déplacement

    def compter_mvt_motrice(ind, lotNum: int):
        """
        Compte les mouvements du lot numéro 'lotNum'.
        """
        cmptMvt = 0
        while cmptMvt < config.nbMvtParLot and ind[lotNum * config.nbMvtParLot + cmptMvt]:
            cmptMvt += 1
        return cmptMvt

    def inserer_dans_ind(ind, numLot, mvt, pos):
        """
        Insère une valeur dans un individu en décalant les autres entrées.
        La position est relative au lot 'numLot'.
        """
        debutMvts = numLot * config.nbMvtParLot
        for i in range(config.nbMvtParLot - 1, pos, -1):
            ind[debutMvts+i] = ind[debutMvts+i-1]
        ind[debutMvts+pos] = mvt

    def lib_noeud(noeud: int):
        data = probleme.graphe.nodes.get(noeud, None)
        if data:
            return data['libelle']
        else:
            return ''

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
        for numLot in range(len(probleme.lots)):
            lot = probleme.lots[numLot]
            debutMvts = numLot * config.nbMvtParLot     # Position dans l'individu du début des mvts rattachés à la motrice

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
                    if statutAttelage or numMvt == config.nbMvtParLot - 1:      # Pas de Récup en tant que dernier mouvement
                        ind[debutMvts + numMvt] = None      # Mouvement inutile
                    else:
                        statutAttelage = mvt.motrice
                elif mvt.type == TypeMouvement.Deposer:
                    if statutAttelage == mvt.motrice:                           # On vérifie que c'est la bonne motrice qui réalise la dépose
                        statutAttelage = None
                    else:
                        ind[debutMvts + numMvt] = None      # Mouvement inutile
            if statutAttelage:      # Ajout d'une dépose finale s'il y a toujours un attelage
                ind[debutMvts + config.nbMvtParLot - 1] = Mouvement(TypeMouvement.Deposer, lot.noeudDestination, statutAttelage, lot, priorite=1)       # La dépose peut être forcée sur le dernier mouvement, car on s'est assuré qu'il n'y a pas de récup à ce moment là

            # Réorganisation des mouvements
            nonNones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParLot] if mvt]
            nones = [mvt for mvt in ind[debutMvts:debutMvts+config.nbMvtParLot] if not mvt]
            mvtsLot = nonNones + nones                  # Reconstruction du tableau pour la motrice avec tous les None à la fin
            ind = ind[:debutMvts] + mvtsLot + ind[debutMvts+len(mvtsLot):]       # Insertion du nouvel ordre dans l'individu
        return creator.Individual(ind)
    
    def muter_individu(ind):
        # Sélection d'un lot et comptage des mouvements
        numLot = random.choice(range(len(probleme.lots)))
        lot = probleme.lots[numLot]
        debutMvts = numLot * config.nbMvtParLot         # Position dans l'individu du début des mvts rattachés au lot
        cmptMvt = compter_mvt_motrice(ind, numLot)

        # Sélection d'une mutation
        if cmptMvt > 0 and cmptMvt < config.nbMvtParLot:
            typeMut = random.choice([TypeMutation.Ajout, TypeMutation.Suppression, TypeMutation.Variation])
        elif cmptMvt >= config.nbMvtParLot:
            typeMut = [TypeMutation.Suppression, TypeMutation.Variation]
        else: # cmptMvt == 0
            typeMut = TypeMutation.Ajout

        # Mutation Ajout
        if typeMut == TypeMutation.Ajout:
            # Sélection du mouvement à ajouter
            sel = random.randint(1, 100)      # 33% de chance d'ajouter une attente, 66% d'ajouter un couple Réc+Dép

            # Ajout de mouvements Récupérer + Déposer
            if sel > 33 and cmptMvt < config.nbMvtParLot - 1:      # S'il n'y a pas de place, on n'ajoute pas de couple Réc+Dép
                mvtPrecedent = ind[debutMvts + cmptMvt - 1]
                noeudOrigine = mvtPrecedent.noeud if mvtPrecedent else lot.noeudOrigine
                noeudDestination = random.choice(noeudsCibles)
                motrice = random.choice(probleme.motrices)
                priorite = random.randint(1, 3)
                
                nouveauMvtRec = Mouvement(TypeMouvement.Recuperer, noeudOrigine, motrice, lot, priorite)
                nouveauMvtDep = Mouvement(TypeMouvement.Deposer, noeudDestination, motrice, lot, priorite)
                ind[debutMvts + cmptMvt] = nouveauMvtRec
                ind[debutMvts + cmptMvt + 1] = nouveauMvtDep
            # Ajout d'un mouvement Attendre
            else:
                noeudCible = random.choice(noeudsCibles)
                motrice = random.choice(probleme.motrices)
                priorite = random.choice([1, 2, 3])
                duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))
                nouveauMvt = Mouvement(TypeMouvement.Attendre, noeudCible, motrice, lot, priorite, duree)          # Remarque: pour un mouvement d'attente, le lot n'est pas utilisé
                pos = random.randint(0, cmptMvt)            # Les attentes peuvent être insérées n'importe où sans briser la causalité 
                inserer_dans_ind(ind, numLot, nouveauMvt, pos)
        # Mutation Suppression
        elif typeMut == TypeMutation.Suppression:
            # Sélection de l'emplacement du mouvement à supprimer dans la séquence
            pos = random.randint(0, cmptMvt - 1)
            ind[debutMvts + pos] = None         # Suppression directe du mouvement. Les corrections sont apportées dans la réparation.
        # Mutation Variation
        elif typeMut == TypeMutation.Variation:
            pos = random.randint(0, cmptMvt - 1)

            # Sélection d'une nouvelle priorité
            priorite = None
            while priorite is None or priorite == ind[debutMvts + pos].priorite:
                priorite = random.randint(1, 3)                                 # On s'assure de proposer une nouvelle priorité pour que la mutation soit utile

            if ind[debutMvts + pos].type == TypeMouvement.Recuperer:
                ind[debutMvts + pos].priorite = priorite                        # Variation de la priorité
            elif ind[debutMvts + pos].type == TypeMouvement.Deposer:
                ind[debutMvts + pos].priorite = priorite                        # Variation de la priorité
            elif ind[debutMvts + pos].type == TypeMouvement.Attendre:
                if random.randint(0, 1) == 1:
                    ind[debutMvts + pos].priorite = priorite                    # Variation de la priorité
                else:
                    ind[debutMvts + pos].duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))      # Variation de l'attente

        return reparer_individu(ind), 

    def croiser_individus(ind1, ind2):
        # Sélection du segment
        a, b = sorted(random.sample(range(config.nbMvtParLot * len(probleme.lots)), 2))

        # Echange des segments
        enfant1 = ind1[:a] + ind2[a:b] + ind1[b:]
        enfant2 = ind2[:a] + ind1[a:b] + ind2[b:]
        return reparer_individu(enfant1), reparer_individu(enfant2)
    
    def evaluer_individu(ind):
        """
        Fonction relai d'évaluation pour la toolbox de DEAP.
        """
        return simuler_individu(tuple(ind))
    
    def calculer_solution(ind):
        """
        Calcule la solution associée à un individu.
        """
        return simuler_individu(tuple(ind), retournerSolution=True)
    
    @lru_cache(maxsize=128)
    def simuler_individu(ind: tuple, retournerSolution=False):
        """
        Fonction d'évaluation principale, qui utilise un cache sur la variante hashable d'un individu.
        tracage: si à True, retourne le traçage temporel de la solution. Sinon, retourne directement les objectifs.
        """
        # Déclaration des collections utilisées pour l'évaluation
        derniersNoeudsMots = dict()                             # Dernier noeud visité par chaque motrice
        mvtsActuelsMots = dict()                                # Mouvement actuel pour chaque motrice
        motsTerminees = set()                                    # Collection des motrices pour lesquelles le travail est terminé
        tempsAttenteMots = dict.fromkeys(probleme.motrices, 0)  # Temps d'attente restant pour chaque motrice
        nbWagonsAtteles = dict.fromkeys(probleme.motrices, 0)   # Nombre de wagons attelés pour chaque motrice

        derniersNoeudsLots = dict()                             # Dernier noeud visité par chaque lot
        indicesMvtsLots = dict.fromkeys(probleme.lots, 0)       # Indice du mouvement actuel dans l'individu pour chaque lot
        attelages = dict()                                      # Attelage associant une motrice à chaque lot
        lotsValides = set()

        itinerairesActuels = dict()                             # Itinéraires suivis actuellement pour chaque motrice
        blocagesItineraires = []                                # Blocages générés par les itinéraires

        # -----------------------------------
        # Réinitialisation du traçage
        # -----------------------------------
        if retournerSolution:
            tracageMots = dict.fromkeys(probleme.motrices, [None] * config.horizonTemp)                     # Liste toutes les positions successives pour chaque motrice et chaque lot
            tracageLots = dict.fromkeys(probleme.lots, [None] * config.horizonTemp)
            tracageAttelages = dict.fromkeys(probleme.lots, [None] * config.horizonTemp)

            for k, v in tracageMots.items():
                tracageMots[k] = [None] * config.horizonTemp
            for k, v in tracageLots.items():
                tracageLots[k] = [None] * config.horizonTemp
            for k, v in tracageAttelages.items():
                tracageAttelages[k] = [None] * config.horizonTemp

        # -----------------------------------
        # Vérifications initiales
        # -----------------------------------
        # Vérifie que l'individu n'est pas vide
        if not any(pos is not None for pos in ind):
            return (config.coutMax, config.coutMax)
        
        # Initialisation des derniers noeuds
        for mot in probleme.motrices:
            derniersNoeudsMots[mot] = mot.noeudOrigine
        for lot in probleme.lots:
            derniersNoeudsLots[lot] = lot.noeudOrigine

        # -----------------------------------
        # Boucle temporelle
        # -----------------------------------
        objCoutsLogistiques = 0         # Objectif de coûts d'exploitation de la solution
        objDeplacement = 0              # Objectif de durée de parcours pour l'ensemble des motrices
        for t in range(config.horizonTemp):
            # Si toutes les motrices ont terminé leur travail, on sort de la boucle temporelle
            if len(motsTerminees) == len(probleme.motrices):
                break

            # -----------------------------------
            # Mise à jour du traçage
            # -----------------------------------
            if retournerSolution:
                for mot in probleme.motrices:
                    tracageMots[mot][t] = derniersNoeudsMots[mot]
                for lot in probleme.lots:
                    tracageLots[lot][t] = derniersNoeudsLots[lot]
                for lot in probleme.lots:
                    tracageAttelages[lot][t] = attelages.get(lot, None)

            # -----------------------------------
            # Simulation des motrices
            # -----------------------------------
            for mot in probleme.motrices:
                # Si la motrice a terminé son travail, passage à la suivante
                if mot in motsTerminees:
                    continue

                # Extraction et vérification du mouvement actuel de la motrice
                mvtActuel = mvtsActuelsMots.get(mot, None)
                if not mvtActuel:
                    mvtSelectionne = None                   # Si aucun mouvement actuel: sélection du mouvement suivant
                    for numLot in range(len(probleme.lots)):
                        # Vérification du lot
                        lot = probleme.lots[numLot]
                        if indicesMvtsLots[lot] >= config.nbMvtParLot:      # Si on a déjà terminé tous les mouvements du lot, alors il est ignoré
                            continue
                        # Sélection du mouvement potentiel
                        mvtPotentiel = ind[numLot * config.nbMvtParLot + indicesMvtsLots[lot]]
                        if not mvtSelectionne or (mvtPotentiel and mvtPotentiel.motrice == mot and mvtPotentiel.priorite < mvtSelectionne.priorite):
                            mvtSelectionne = mvtPotentiel
                            break
                    # Validation définitive du mouvement
                    mvtsActuelsMots[mot] = mvtSelectionne
                    mvtActuel = mvtsActuelsMots[mot]

                # Si aucun mouvement n'a été sélectionné, alors la motrice a terminé son travail et ne sera plus traitée
                if not mvtActuel:
                    motsTerminees.add(mot)
                    continue

                # -----------------------------------
                # Itinéraire et avancée
                # -----------------------------------
                # Fonction pour l'itinéraire
                def est_arc_disponible(u, v, t):
                    """
                    Fonction pour la vérification des arcs bloqués
                    """
                    # Vérification des blocages du problème
                    for sillon in probleme.blocages:
                        if sillon.est_dans_sillon(u, v, t):
                            return False
                    # Vérification des blocages ajoutés par les itinéraires
                    for sillon in blocagesItineraires:
                        if sillon.est_dans_sillon(u, v, t) and sillon.motrice != mot:
                            return False
                    return True
                
                # Traitement de l'itinéraire
                if not itinerairesActuels.get(mot, None):
                    itineraire = dijkstra_temporel(probleme.graphe, derniersNoeudsMots[mot], mvtActuel.noeud, t, est_arc_disponible)
                    itinerairesActuels[mot] = itineraire

                    # Si aucun itinéraire n'est disponible, la motrice reste immobile à cet instant
                    if not itineraire:
                        continue

                    # Blocage des arcs de l'itinéraire
                    for i in range(len(itineraire)-1):
                        # Extraction des information
                        u = itineraire[i]
                        v = itineraire[i+1]
                        debutParcours = round(u[1])
                        finParcours = round(v[1])

                        # Ajout des sillons bloqués
                        blocagesItineraires.append(Sillon(u[0], v[0], debutParcours, debutParcours+config.ecartementMinimal, mot))
                        if probleme.graphe.edges[u[0], v[0]]['exploit'] == 'Simple':
                            blocagesItineraires.append(Sillon(v[0], u[0], debutParcours, finParcours+config.ecartementMinimal, mot))
                pass            # TODO: Traiter les cas où la durée est à 0 (passage immédiatement au mouvement suivant)

                # Calcul de l'avancée
                itineraireTermine = True
                itineraire = itinerairesActuels[mot]
                for i in range(len(itineraire)):
                    if t < itineraire[i][1]:                # Parcoure toutes les étapes de l'itinéraire jusqu'à trouver une étape qui n'a pas encore été atteinte 
                        itineraireTermine = False
                        derniersNoeudsMots[mot] = itineraire[i-1][0]
                        break

                # Finalisation de l'itinéraire
                if not itineraireTermine:
                    continue                # Fin du traitement si l'itinéraire n'est pas terminé
                derniersNoeudsMots[mot] = itineraire[-1][0]
                dureeItineraire = round(itinerairesActuels[mot][-1][1] - itinerairesActuels[mot][0][1])      # Calcule la durée totale du parcours avant de supprimer l'itinéraire
                itinerairesActuels[mot] = None

                # Application de la pénalité de déplacement
                objDeplacement += dureeItineraire * config.coeffDeplacements

                # -----------------------------------
                # Traitement des actions
                # -----------------------------------
                # Action d'attelage
                if mvtActuel.type == TypeMouvement.Recuperer:
                    lot = mvtActuel.lot
                    attelages[lot] = mot
                    tempsAttenteMots[mot] += config.tempsAttelage

                    # Gestion des dépassements de capacité pour les motrices
                    nbWagonsAtteles[mot] += lot.taille
                    if nbWagonsAtteles[mot] > mot.capacite:
                        objCoutsLogistiques += (nbWagonsAtteles - mot.capacite) * config.coeffDepassementCapaMotrice
                
                # Action de désattelage
                elif mvtActuel.type == TypeMouvement.Deposer:
                    lot = mvtActuel.lot
                    attelages[lot] = None
                    derniersNoeudsLots[lot] = derniersNoeudsMots[mot]                    # Mise à jour de la position du lot avant son désattelage
                    tempsAttenteMots[mot] += config.tempsDesattelage                     # TODO: bloquer aussi le lot pendant la durée du désattelage
                    nbWagonsAtteles[mot] -= lot.taille

                # Action d'attente
                elif mvtActuel.type == TypeMouvement.Attendre:
                    tempsAttenteMots[mot] += mvtActuel.param

                # Passage au mouvement suivant
                indicesMvtsLots[lot] += 1
                mvtsActuelsMots[mot] = None


            # -----------------------------------
            # Simulation des lots
            # -----------------------------------
            for lot in probleme.lots:
                # Mise à jour de la position de la motrice
                if attelages.get(lot, None):
                    derniersNoeudsLots[lot] = derniersNoeudsMots[attelages[lot]]

                # Si le lot n'a pas encore atteint sa destination, pas de traitements supplémentaires
                if attelages.get(lot, None) or derniersNoeudsLots[lot] != lot.noeudDestination:
                    continue

                # Application des pénalités de retard
                if t >= lot.finCommande and not lot in lotsValides:
                    objCoutsLogistiques += (t - lot.finCommande) * config.coeffRetard           # La pénalité correspond au nombre d'instants écoulé depuis la fin de la commande
                lotsValides.add(lot)


        # -----------------------------------
        # Finalisation de la simulation
        # -----------------------------------
        # Ajout des coûts non livrés
        solutionValide = True                   # La solution est dite valide si tous les lots sont livrés
        for lot in probleme.lots:
            if not lot in lotsValides:
                objCoutsLogistiques += config.coutLotNonLivre
                solutionValide = False

        # L'objectif de distance est neutralisé tant que toutes les commandes ne sont pas réalisées
        if not solutionValide:
            objDeplacement = config.coutMax                         

        # Fin de l'évaluation. Retourne soit l'objectif (si évaluation classique) soit la solution.
        if retournerSolution:
            return Solution((objDeplacement, objCoutsLogistiques), tracageMots, tracageLots, tracageAttelages)
        else:
            return (int(objDeplacement), int(objCoutsLogistiques))
            

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