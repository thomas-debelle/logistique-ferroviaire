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
        self.nbMvtParMot = 20                           # Nom de mouvements max par motrice dans une solution
        self.lambdaTempsAttente = 3.0                   # Paramètre lambda de la loi exponentielle utilisée pour générer les temps d'attente dans les mutations
        self.ecartementMinimal = 8                      # Ecartement minimal (en min) entre deux trains qui se suivent
        self.tempsAttelage = 10                         # Temps de manoeuvre pour l'attelage
        self.tempsDesattelage = 5                       # Temps de manoeuvre pour le désattelage
        self.horizonTemp = 500                          # Horizon temporel de la simulation

        self.coeffDeplacements = 1.0                    # Coefficient appliqué à l'objectif de déplacement
        self.coeffRetard = 1.0                          # Coefficient appliqué aux coûts de retard à l'arrivée
        self.coutLotNonLivre = 1000                     # Coût d'un lot non livré
        self.coeffDepassementCapaNoeud = 1.0            # Coefficient appliqué aux coûts pour les dépassements de capacité sur les noeuds.  # TODO: à utiliser
        self.coeffDepassementCapaMotrice = 1.0          # Coefficient appliqué aux coûts pour les dépassements de capacité des motrices.

        self.penaliteMvtInutiles = 100                  # Pénalité pour chaque mouvement inutile
        self.penaliteLotNonLivre = 100                  # Pénalité pour chaque lot non livré


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
    def __init__(self, type: TypeMouvement, noeud: int, param):
        self.type = type
        self.noeud = noeud
        self.param = param

    def __str__(self):
        return f'({self.type.name}, {self.noeud}, {self.param})'
    
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
    typesMouvementsListe = list(TypeMouvement)
    typesMutationsListe = list(TypeMutation)
    noeudsCibles = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds pouvant être utilisés comme cible pour un mouvement de déplacement

    def compter_mvt_motrice(ind, motNum: int):
        """
        Compte les mouvements de la motrice numéro 'motNum'.
        """
        cmptMvt = 0
        while not ind[motNum * config.nbMvtParMot + cmptMvt] is None and cmptMvt < config.nbMvtParMot:
            cmptMvt += 1
        return cmptMvt
    
    def generer_individu():
        """
        Génère aléatoirement une solution sous la forme d'un individu. 
        """
        ind = [None] * len(probleme.motrices) * config.nbMvtParMot         # Un individu est une liste concaténée des mouvements successifs de toutes les motrices
        return creator.Individual(ind)
    
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
        if typeMut == TypeMutation.Ajout:
            # Construction d'un mouvement
            typeMvt = random.choice(typesMouvementsListe)
            cible = random.choice(noeudsCibles)
            if typeMvt == TypeMouvement.Recuperer:
                lot = random.choice(probleme.lots)
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, cible, lot)         # Ignoré si le lot n'est pas sur le même noeud que la motrice lors de l'évaluation du mouvement
            elif typeMvt == TypeMouvement.Deposer:
                lot = random.choice(probleme.lots)
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, cible, lot)         # Ignoré si le lot n'est pas attaché à la motrice lors de l'évaluation du mouvement
            else: #typeMvt == TypeMouvement.Attendre
                duree = ceil(np.random.exponential(scale=config.lambdaTempsAttente))
                ind[debutMvts + cmptMvt] = Mouvement(typeMvt, cible, duree)
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

        return reparer_individu(enfant1), reparer_individu(enfant2)

    def evaluer_individu(ind):
        # Variables pour la simulation
        numMvtActuels = [0] * len(probleme.motrices)                            # Indice du mouvement actuel pour chaque motrice
        derniersNoeudsMots = [0] * len(probleme.motrices)                       # Index du dernier noeud visité pour chaque motrice. Motrices indexées par leur position dans ind.
        tempsAttenteMots = [0] * len(probleme.motrices)                         # Temps d'attente restant pour chaque motrice
        nbWagonsAtteles = [0] * len(probleme.motrices)                          # Nombre de wagons attelés pour chaque motrice

        derniersNoeudsLots = dict()                                             # Index du dernier noeud visité pour chaque lot. Lots indexés avec leurs références.
        attelages = dict()                                                      # Motrice d'attelage de chaque lot. Motrice indexée par sa position dans ind. 
        lotsValides = set()

        itinerairesActuels = [None] * len(probleme.motrices)                    # Itinéraires suivis actuellement pour chaque motrice
        blocagesItineraires = []                                                # Blocages générés par les itinéraires

        # Vérifie que l'individu n'est pas vide
        if not any(pos is not None for pos in ind):
            return (sys.maxsize, sys.maxsize, sys.maxsize)
        
        # Initialisation des derniers noeuds
        numMot = 0
        for mot in probleme.motrices:
            derniersNoeudsMots[numMot] = mot.noeudOrigine
            numMot += 1
        for lot in probleme.lots:
            derniersNoeudsLots[lot] = lot.noeudOrigine

        # Simulation
        objDeplacement = 0              # Objectif de durée de parcours pour l'ensemble des motrices
        objCoutsLogistiques = 0         # Objectif de coûts d'exécution pour la solution proposée
        objPenalisation = 0                # Objectif pour la pénalisation des mouvements inutiles
        for t in range(config.horizonTemp):
            # TODO: supprimer progressivement tous les blocages expirés (une fois tous les n instants)
            # Traitement des motrices
            for numMot in range(len(probleme.motrices)):
                # Extraction des informations de la motrice
                mot = probleme.motrices[numMot]

                # Fonction pour la vérification des arcs bloqués
                def est_arc_disponible(u, v, t):
                    # Vérification des blocages du problème
                    for sillon in probleme.blocages:
                        if sillon.est_dans_sillon(u, v, t):
                            return False
                    # Vérification des blocages ajoutés par les itinéraires
                    for sillon in blocagesItineraires:
                        if sillon.est_dans_sillon(u, v, t) and sillon.motrice != mot:
                            return False
                    return True

                # Traitement des temps d'attente
                if tempsAttenteMots[numMot] > 0:
                    tempsAttenteMots[numMot] -= 1
                    continue

                # Extraction et vérification du mouvement actuel
                numMvtActuel = numMvtActuels[numMot]
                mvtActuel = ind[numMvtActuel]
                if not mvtActuel:
                    continue

                # Génération d'un itinéraire
                if not itinerairesActuels[numMot]:
                    itineraire = dijkstra_temporel(probleme.graphe, derniersNoeudsMots[numMot], mvtActuel.noeud, t, est_arc_disponible)
                    itinerairesActuels[numMot] = itineraire
                    
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
                pass            # TODO: Traiter les cas où la durée est à 0 (passage immédiatement au mouvement suivant) OU où aucun itinéraire n'est disponible

                # Calcul de l'avancée
                itineraireTermine = True
                itineraire = itinerairesActuels[numMot]
                for i in range(len(itineraire)):
                    if t < itineraire[i][1]:                # Parcoure toutes les étapes de l'itinéraire jusqu'à trouver une étape qui n'a pas encore été atteinte 
                        itineraireTermine = False
                        derniersNoeudsMots[numMot] = itineraire[i-1][0]
                        break

                # Fin du traitement si l'itinéraire n'est pas terminé
                if not itineraireTermine:
                    continue
                derniersNoeudsMots[numMot] = itineraire[-1][0]
                dureeItineraire = round(itinerairesActuels[numMot][-1][1] - itinerairesActuels[numMot][0][1])      # Calcule la durée totale du parcours avant de supprimer l'itinéraire
                itinerairesActuels[numMot] = None

                # Application de la pénalité de déplacement
                objDeplacement += dureeItineraire * config.coeffDeplacements

                # Action d'attelage
                if mvtActuel.type == TypeMouvement.Recuperer:
                    lot = mvtActuel.param
                    if not attelages.get(lot, None) and derniersNoeudsMots[numMot] == derniersNoeudsLots[lot] and t > lot.debutCommande:
                        attelages[lot] = numMot
                        tempsAttenteMots[numMot] += config.tempsAttelage

                        # Gestion des dépassements de capacité pour les motrices
                        nbWagonsAtteles[numMot] += lot.taille
                        capaMot = probleme.motrices[numMot].capacite
                        if nbWagonsAtteles[numMot] > capaMot:
                            objCoutsLogistiques += (nbWagonsAtteles - capaMot) * config.coeffDepassementCapaMotrice
                    else:
                        objPenalisation += config.penaliteMvtInutiles        # Sanction des mouvements inutiles.
                # Action de désattelage
                elif mvtActuel.type == TypeMouvement.Deposer:
                    lot = mvtActuel.param
                    if attelages.get(lot, None) == numMot:
                        attelages[lot] = None
                        derniersNoeudsLots[lot] = derniersNoeudsMots[numMot]                    # Mise à jour de la position du lot avant son désattelage
                        tempsAttenteMots[numMot] += config.tempsDesattelage                     # TODO: bloquer aussi le lot pendant la durée du désattelage
                        nbWagonsAtteles[numMot] -= lot.taille
                    else:
                        objPenalisation += config.penaliteMvtInutiles        # Sanction des mouvements inutiles.
                # Action d'attente
                elif mvtActuel.type == TypeMouvement.Attendre:
                    tempsAttenteMots[numMot] += mvtActuel.param

                # Passage au mouvement suivant
                numMvtActuels[numMot] += 1

            # Traitement des lots
            for lot in probleme.lots:
                # Mise à jour de la position de la motrice
                if attelages.get(lot, None):
                    derniersNoeudsLots[lot] = derniersNoeudsMots[attelages[lot]]

                # Si le lot n'a pas encore atteint sa destination, pas de traitements supplémentaires
                if attelages.get(lot, None) or derniersNoeudsLots[lot] != lot.noeudDestination:
                    continue

                # Application des pénalités de retard
                if t >= lot.finCommande:
                    objCoutsLogistiques += (lot.finCommande - t) * config.coeffRetard           # La pénalité correspond au nombre d'instants écoulé depuis la fin de la commande
                lotsValides.add(lot)
        

        # Pénalisation des lots non livrés
        for lot in probleme.lots:
            if not lot in lotsValides:
                objCoutsLogistiques += config.coutLotNonLivre
                objPenalisation += config.penaliteLotNonLivre

        return (0, int(objCoutsLogistiques), int(objPenalisation))
            

    # Initialisation de l'algorithme génétique
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
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
    probleme.lots = [Lot(0, 222, 277, 0, 300)]

    # Résolution du problème
    resoudre_probleme(config, probleme)
    pass

main()