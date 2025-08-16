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
from glob import glob
from copy import deepcopy
from frozendict import frozendict
from functools import lru_cache
import json
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
from carte import generer_carte
from collections import defaultdict
from PIL import Image, ImageDraw
import io, base64
import signal
from sys import maxsize

# Pour l'interruption du programme
def handler(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handler)

# ---------------------------------------
# Définition des classes et des types
# ---------------------------------------
class Config:
    def __init__(self):
        # Paramètres du programme
        self.dossierLogs = 'logs'
        self.fichierLogs = None
        self.nbLogsMax = 10

        # Paramètres de l'algorithme
        self.nbGenerations = 300                        # Nombre de génération
        self.taillePopulation = 100                     # Taille de la population
        self.cxpb = 0.85                                # Probabilité de croisement
        self.mutpb = 0.25                               # Probabilité de mutation
        
        self.ecartementMinimal = 8                      # Ecartement temporel minimal (en min) entre deux trains qui se suivent
        self.dureeAttelage = 10                         # Temps de manoeuvre pour l'attelage (en min)
        self.dureeDesattelage = 5                       # Temps de manoeuvre pour le désattelage (en min)
        self.dureeRebroussement = 30                    # Temps supplémentaire ajouté en cas de rebroussement sur un arc (en min)
        self.horizonTemp = 1440                         # Horizon temporel de la simulation (en min)
        self.dureeConservationBlocages = 50             # Nombre d'instants entre deux nettoyages des blocages temporaires
        self.lambdaTempsAttente = 20.0                  # Facteur d'échelle utilisé pour la distribution exponentielle des temps d'attente
        self.orienterAjoutEtape = False                 # Si activé, une heuristique est utilisée pour orienter l'ajout d'étapes (recherche sur le plus court chemins entre les étapes précédentes et suivantes)

        self.coutMax = 10000                            # Coût maximum, appliqué à l'objectif de distance si toutes les livraisons ne sont pas réalisées dans l'horizon temporel
        self.coutFixeParMotrice = 0                     # Coût logistique fixe pour chaque motrice utilisée dans le problème
        self.coeffDeplacements = 1.0                    # Coefficient appliqué à l'objectif de déplacement
        self.coeffRetard = 1.0                          # Coefficient appliqué aux coûts de retard à l'arrivée
        self.coeffDepassementCapaMotrice = 1.0          # Coefficient appliqué aux coûts pour les dépassements de la capacité de traction des motrices.
        self.coeffDepassementCapaNoeud = 1.0            # Coefficient appliqué aux coûts pour les dépassements de la capacité de stockage des noeuds.

        self.lonMin = 2.06                              # Coordonnées pour générer l'animation  
        self.lonMax = 7.68 
        self.latMin = 44.39  
        self.latMax = 46.78


class Motrice:
    def __init__(self, index: int, noeudOrigine: int, capacite=50, retourBase=False, taille=2):
        """
        taille: taux d'occupation du noeud lorsque la motrice est en attente.
        """
        self.index = index
        self.noeudOrigine = noeudOrigine
        self.capacite = capacite
        self.retourBase = retourBase
        self.taille = taille

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
        self.blocages = defaultdict(list)              # Liste des sillons bloqués par défaut (par exemple, par d'autres compagnies ferroviaires)

        # Vérification des index
        indexMotrices = set()
        indexLots = set()
        for mot in self.motrices:
            if not mot.index in indexMotrices:
                indexMotrices.add(mot.index)
            else:
                raise Exception(f"Index en double dans les motrices: {mot.index}")
        for lot in self.lots:
            if not lot.index in indexLots:
                indexLots.add(lot.index)
            else:
                raise Exception(f"Index en double dans les lots: {lot.index}")
        for b in self.blocages:
            if not (b.noeudDebut, b.noeudFin) in self.graphe.edges:
                warning(f"L'arc ({b.noeudDebut}, {b.noeudFin}) n'existe pas et ne sera pas pris en compte.")

    def ajouter_blocage(self, sillon: Sillon):
        cle = (sillon.noeudDebut, sillon.noeudFin)
        if not cle in self.graphe.edges:
            warning(f"L'arc ({cle}) n'existe pas et ne sera pas pris en compte.")
        self.blocages[cle].append(sillon)

class TypeMouvement(Enum):
    Recuperer = 'Réc.'
    Deposer = 'Dép.'

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()
    
class Mouvement:
    def __init__(self, type: TypeMouvement, lot: Lot, etape: int, attente: int=0):
        self.type = type
        self.lot = lot
        self.etape = etape
        self.attente = attente      # Temps d'attente avant le début du mouvement

    def __str__(self):
        return f'({self.type}, {self.lot}, {self.etape}, {self.attente})'
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        return isinstance(value, Mouvement) and self.type == value.type and self.lot == value.lot and self.etape == value.etape
    
    def __hash__(self):
        return hash((self.type, self.lot, self.etape))

class Solution:
    def __init__(self, objs, tracageMots: Dict[Motrice, List[int]], tracageLots: Dict[Lot, List[int]], tracageAttelages: Dict[Lot, Motrice], etapesLots: Dict[Lot, List[int]]):
        self.objs = objs
        self.tracageMots = tracageMots
        self.tracageLots = tracageLots
        self.tracageAttelages = tracageAttelages
        self.etapesLots = etapesLots

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
    date = datetime.now()
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
        noeudConverti = int(noeud)
        grapheConverti.add_node(noeudConverti, **grapheOriginal.nodes[noeud])
        grapheConverti.nodes[noeudConverti]['transRebroussement'] = json.loads(grapheConverti.nodes[noeudConverti]['transRebroussement'])       # Extraction des transitions bloquées

    # Copie des arêtes avec conversion des extrémités
    for u, v, data in grapheOriginal.edges(data=True):
        grapheConverti.add_edge(int(u), int(v), **data)

    return grapheConverti

def lib_noeud(graphe, noeud: int):
    """
    Extrait le libellé du noeud. Utile pour le déboguage.
    """
    data = graphe.nodes.get(noeud, None)
    if data:
        return data.get('libelle', '')
    else:
        return ''
    
def coords_noeud(graphe, noeud: int):
    data = graphe.nodes.get(noeud, None)
    if data:
        return (float(data['lat']), float(data['lon']))
    else:
        return (None, None)
    
def traduire_itineraire(graphe, itineraire: list):
    """
    itineraire: Liste de couples, où le premier élément est un noeud, et le second élément est une durée de parcours depuis le départ.
    """
    if itineraire is None:
        return None
    
    traduction = []
    for couple in itineraire:
        traduction.append(((str(couple[0]).ljust(6) + ' | ' + lib_noeud(graphe, couple[0])).ljust(40), couple[1]))
    return traduction
        

def extraire_noeud(graphe: nx.Graph, index: int):
    """
    Recherche le noeud correspondant à l'index passé en argument dans le graphe.
    """
    return next((n for n, d in graphe.nodes(data=True) if d.get('index', 0) == index), None)

def dijkstra_special(graphe, origine, dest, instantInitial, fonctionCout, dureeRebroussement=0):
    """
    Variante de Dijkstra permettant de filtrer les arcs en fonction de l'instant.
    Détecte et pénalise également les rebroussements.
    Retourne un chemin et la liste des rebroussements.
    """

    queue = [(instantInitial, origine, [], [], None)]  # (tempsCumul, noeudActuel, chemin, noeudPrecedent)
    visites = {}

    while queue:
        instantActuel, noeudActuel, chemin, rebroussements, noeudPrecedent = heapq.heappop(queue)

        if (noeudActuel in visites) and (instantActuel >= visites[noeudActuel]):
            continue
            
        visites[noeudActuel] = instantActuel
        chemin = chemin + [(noeudActuel, instantActuel)]

        if noeudActuel == dest:
            return chemin, rebroussements

        for successeur in graphe[noeudActuel]:      # Parcours des successeurs
            poids = fonctionCout(noeudActuel, successeur, instantActuel)
            nouveauxRebroussements = rebroussements     # Par défaut, pas de copie

            # Vérification des transitions de rebroussement (en fonction des angles de virages)
            if noeudPrecedent and sorted([noeudPrecedent, successeur]) in graphe.nodes[noeudActuel]['transRebroussement']:
                dureeParcours = poids + dureeRebroussement
                nouveauxRebroussements = rebroussements + [(noeudActuel, instantActuel)]        # Copie et création d'une nouvelle liste
            else:
                dureeParcours = poids
            instantSuivant = maxsize if dureeParcours == maxsize else instantActuel + dureeParcours
            heapq.heappush(queue, (instantSuivant, successeur, chemin, nouveauxRebroussements, noeudActuel))     # Le noeud peut être emprunté, et est ajouté à la pile

    return None, None

    


def resoudre_probleme(config: Config, probleme: Probleme):
    """
    Cherche une solution pour résoudre l'exploitation.
    """
    noeudsCapa = [n for n, d in probleme.graphe.nodes(data=True) if d.get('capacite') > 0]        # Liste des noeuds avec une capacité de stockage, et pouvant être utilisés comme cible pour une étape de lot ou un mouvement de motrice

    class Individu:
        def __init__(self, autreInd=None):
            if autreInd is not None:
                # Constructeur par copie
                self.mvtsMotrices = {mot: deepcopy(mvts) for mot, mvts in autreInd.mvtsMotrices.items()}
                self.etapesLots = {lot: deepcopy(etapes) for lot, etapes in autreInd.etapesLots.items()}
            else:
                # Constructeur par défaut
                self.mvtsMotrices = {}
                for mot in probleme.motrices:
                    self.mvtsMotrices[mot] = []     # Pas d'utilisation de dict.fromkeys, car provoque des erreurs de hash
                self.etapesLots = {}
                for lot in probleme.lots:
                    self.etapesLots[lot] = []
    
        def __eq__(self, other):
            return (self.mvtsMotrices == other.mvtsMotrices and
                    self.etapesLots == other.etapesLots)

        def __hash__(self):
            mvtsMotricesFrozen = frozendict({mot.index: tuple(mvts) for mot, mvts in self.mvtsMotrices.items()})
            etapesLotsFrozen = frozendict({lot.index: tuple(etapes) for lot, etapes in self.etapesLots.items()})
            return hash((mvtsMotricesFrozen, etapesLotsFrozen))
        
        def inserer_mvts_lies(self, mot: Motrice, mvtRecup: Mouvement, mvtDepose: Mouvement):
            """
            Insère les mouvements en respectant la causalité.
            Attention, les mouvements doivent être liés.
            """
            # Recherche d'un emplacement pour les mouvements (avec respect de la causalité) 
            mvtsInseres = False
            for i, mvt in enumerate(self.mvtsMotrices[mot]):
                if mvt.type == TypeMouvement.Deposer and mvt.lot == mvtRecup.lot and mvt.etape == mvtRecup.etape:
                    self.mvtsMotrices[mot].insert(i+1, mvtRecup)
                    self.mvtsMotrices[mot].insert(i+2, mvtDepose)        # Insertion des mouvements juste avant i
                    mvtsInseres = True
                    break 
            if not mvtsInseres:
                self.mvtsMotrices[mot].append(mvtRecup)
                self.mvtsMotrices[mot].append(mvtDepose)

        def extraire_mvt_lie(self, mot: Motrice, mvt: Mouvement):
            """
            Recherche le mouvement lié au mouvement passé en argument.
            Retourne un tuple contenant son index et sa valeur.
            """
            # Extraction de l'index
            mvt1 = mvt
            try:
                indexMvt1 = self.mvtsMotrices[mot].index(mvt)
            except Exception:
                error(f'Le mouvement {mvt} n\'est pas rattaché à la motrice {mot}.')
                pause_and_exit()

            # Recherche de mvt2 (mouvement lié) en fonction du type de mvt1
            if mvt1.type == TypeMouvement.Recuperer:
                for i in range(indexMvt1+1, len(self.mvtsMotrices[mot])):
                    mvt2Potentiel = self.mvtsMotrices[mot][i]
                    if mvt2Potentiel.type == TypeMouvement.Deposer and mvt2Potentiel.lot == mvt1.lot and mvt2Potentiel.etape == mvt1.etape+1:
                        return (i, mvt2Potentiel)
            elif mvt1.type == TypeMouvement.Deposer:
                for i in range(indexMvt1-1, -1, -1):
                    mvt2Potentiel = self.mvtsMotrices[mot][i]
                    if mvt2Potentiel.type == TypeMouvement.Recuperer and mvt2Potentiel.lot == mvt1.lot and mvt2Potentiel.etape == mvt1.etape-1:
                        return (i, mvt2Potentiel)
            else:
                raise Exception(f"Type du mouvement {mvt1} inconnu.")
            
            # S'il n'y a pas de mouvement lié
            warning(f"Aucun mouvement lié à {mvt1}.")
            return (-1, None)

        def inserer_etape(self, lot: Lot, numEtape: int, noeudEtape: int, motriceEtape: Motrice):
            """
            Insère une étape à un lot et attribue le mouvement à une motrice.
            L'ajout ne doit pas être réalisé au début ou à la fin de la suite d'étapes.
            """
            # Insertion de l'étape
            self.etapesLots[lot].insert(numEtape, noeudEtape)
            # Décalage des mouvements existants
            for mot in probleme.motrices:
                for i, mvt in enumerate(self.mvtsMotrices[mot]):
                    if (mvt.lot == lot 
                        and (mvt.etape > numEtape or (mvt.etape == numEtape and mvt.type != TypeMouvement.Deposer))):      # N'incrémente pas l'étape de dépose au noeud cible
                        self.mvtsMotrices[mot][i].etape += 1     # Incrémentation du numéro d'étape
            # Création de mouvements pour attribuer l'étape à une motrice
            mvtRecup = Mouvement(TypeMouvement.Recuperer, lot, numEtape)
            mvtDepose = Mouvement(TypeMouvement.Deposer, lot, numEtape+1)
            self.inserer_mvts_lies(motriceEtape, mvtRecup, mvtDepose)

        def supprimer_etape(self, lot: Lot, numEtape: int):
            # Suppression de l'étape
            self.etapesLots[lot].pop(numEtape)
            # Suppression des mouvements associées et décalages des autres mouvements
            for mot in probleme.motrices:
                indicesMvtsASup = []        # Pour suppression après parcours
                for i, mvt in enumerate(self.mvtsMotrices[mot]):
                    # Enregistrement de la sup. du mvt de récup. à l'étape cible et du mvt de dépose à l'étape suivante
                    if ((mvt.lot == lot and mvt.etape == numEtape and mvt.type == TypeMouvement.Recuperer)
                        or (mvt.lot == lot and mvt.etape == numEtape + 1 and mvt.type == TypeMouvement.Deposer)):
                        indicesMvtsASup.append(i)
                    # Décalage des autres mouvements
                    elif (mvt.lot == lot and mvt.etape > numEtape):
                        self.mvtsMotrices[mot][i].etape -= 1
                # Suppression effective
                cmptSup = 0
                for i in indicesMvtsASup:
                    self.mvtsMotrices[mot].pop(i-cmptSup)
                    cmptSup+=1

        def extraire_mvts_lies(self, mot: Motrice):
            """
            Retourne un tuple contenant deux mouvements liés (Récupérer et Déposer).
            """
            if len(self.mvtsMotrices[mot]) < 2:
                warning(f"{mot} doit être associée à 2 mvts minimum.")
                return None, None
            
            # Sélection d'un mouvement aléatoire
            indexMvt1 = random.randint(0, len(self.mvtsMotrices[mot])-1)
            mvt1 = self.mvtsMotrices[mot][indexMvt1]
            mvt2 = None

            # Si mouvement en Récup
            if mvt1.type == TypeMouvement.Recuperer:
                for i in range(indexMvt1, len(self.mvtsMotrices[mot])):
                    mvtPotentiel = self.mvtsMotrices[mot][i]
                    if mvtPotentiel.type == TypeMouvement.Deposer and mvtPotentiel.lot == mvt1.lot and mvtPotentiel.etape == mvt1.etape+1:
                        mvt2 = mvtPotentiel
                        break
            # Si mouvement en Dépose
            elif mvt1.type == TypeMouvement.Deposer:
                for i in range(indexMvt1-1, -1, -1):
                    mvtPotentiel = self.mvtsMotrices[mot][i]
                    if mvtPotentiel.type == TypeMouvement.Recuperer and mvtPotentiel.lot == mvt1.lot and mvtPotentiel.etape == mvt1.etape-1:
                        mvt2 = mvtPotentiel
                        break
                mvt1, mvt2 = mvt2, mvt1         # Inversion pour toujours avoir la récupération en premier
            else:
                raise Exception(f"Type du mouvement {mvt1} inconnu.")
            
            if mvt2 is None:
                warning(f"Aucun mouvement lié à {self.mvtsMotrices[mot][indexMvt1]}.")
            
            return mvt1, mvt2

    def generer_individu():
        """
        Génère un individu initial. Chaque individu est un tableau de tableau, qui continent
        """
        ind = creator.Individual()
        for lot in probleme.lots:
            ind.etapesLots[lot] = [lot.noeudOrigine, lot.noeudDest]
            mot = random.choice(probleme.motrices)           # Sélection de motrices aléatoires pour amener directement les lots à destination
            mvtRecup = Mouvement(TypeMouvement.Recuperer, lot, 0)       # Les mouvements sont exécutés dans l'ordre, lorsque le lot atteint son étape 0 (c'est à dire, dès le départ).
            mvtDepose = Mouvement(TypeMouvement.Deposer, lot, 1)
            
            ind.mvtsMotrices[mot].append(mvtRecup)      
            ind.mvtsMotrices[mot].append(mvtDepose)
        return ind
    
    def muter_individu(ind: Individu):
        tirageMut = random.choice([0, 1])
        # Mutation des étapes (ajout, suppression)
        if tirageMut == 0:
            lot = random.choice(probleme.lots)
            subTirageMut = random.choice([0, 1, 2]) if len(ind.etapesLots[lot]) > 2 else 0         # Pas de suppression ou variation si seulement origine et dest
            # Ajout d'une étape
            if subTirageMut == 0:
                numEtape = random.randint(1, len(ind.etapesLots[lot]) - 1)      # Sélection d'un étape entre les étapes origine (exclue) et dest (inclue)
                if config.orienterAjoutEtape:
                    noeudsEligibles = list(set(noeudsCapa) & set(nx.shortest_path(probleme.graphe, ind.etapesLots[lot][numEtape-1], ind.etapesLots[lot][numEtape])))    # Intersection des noeuds de capacité non nulle ET sur le plus cours chemin (guidage par une heuristique)
                    noeudsEligibles = noeudsEligibles + random.sample(noeudsCapa, len(noeudsEligibles))    # Ajout d'autres noeuds aléatoires dans les noeuds éligibles
                    noeudEtape = random.choice(noeudsEligibles)                     # Tirage de l'étape
                else:
                    noeudEtape = random.choice(noeudsCapa)
                ind.inserer_etape(lot, numEtape, noeudEtape, random.choice(probleme.motrices))
            # Suppression d'un étape
            elif subTirageMut == 1:  # Suppression uniquement si d'autres étapes que origine et dest
                numEtape = random.randint(1, len(ind.etapesLots[lot]) - 2)       # Sélection d'un étape entre les étapes origine (exclue) et dest (exclue)
                ind.supprimer_etape(lot, numEtape)
            elif subTirageMut == 2:
                numEtape = random.randint(1, len(ind.etapesLots[lot]) - 2)       # Sélection d'un étape entre les étapes origine (exclue) et dest (exclue)
                ind.etapesLots[lot][numEtape] = random.choice(noeudsCapa)      # Sélection d'un noeud de capacité non nulle

        # Mutation des mouvements (déplacement ou échange)
        elif tirageMut == 1:
            subTirageMut = random.choice([0, 1, 2])
            # Déplacement de mouvements entre deux motrices
            if subTirageMut == 0:
                # Tirage des motrices
                if len(probleme.motrices) < 2:
                    return ind,
                mots = random.sample(probleme.motrices, 2); motSource = mots[0]; motCible = mots[1]
                # Tirage des mouvements
                if len(ind.mvtsMotrices[motSource]) == 0:
                    return ind,
                mvtRecup, mvtDepose = ind.extraire_mvts_lies(motSource)
                # Déplacement des mouvements
                ind.mvtsMotrices[motSource].remove(mvtRecup)
                ind.mvtsMotrices[motSource].remove(mvtDepose)
                ind.inserer_mvts_lies(motCible, mvtRecup, mvtDepose)
            # Echange de l'ordre des mouvements dans la séquence d'une même motrice
            elif subTirageMut == 1:
                # Tirage de la motrice et du mouvement
                mot = random.choice(probleme.motrices)
                if len(ind.mvtsMotrices[mot]) == 0:
                    return ind,
                mvt = random.choice(ind.mvtsMotrices[mot])
                # Sélection d'une nouvelle position en respectant le mouvement lié
                indexMvtLie, mvtLie = ind.extraire_mvt_lie(mot, mvt)
                if mvt.type == TypeMouvement.Recuperer:
                    nvIndexMvt = random.randint(0, indexMvtLie-1)
                elif mvt.type == TypeMouvement.Deposer:
                    nvIndexMvt = random.randint(indexMvtLie+1, len(ind.mvtsMotrices[mot])-1)
                # Déplacement à la nouvelle position
                ind.mvtsMotrices[mot].remove(mvt)
                ind.mvtsMotrices[mot].insert(nvIndexMvt, mvt)
            # Ajout d'une attente aléatoire
            elif subTirageMut == 2:
                mot = random.choice(probleme.motrices)
                if len(ind.mvtsMotrices[mot]) == 0:
                    return ind,
                mvt = random.choice(ind.mvtsMotrices[mot])
                mvt.attente += int(round(np.random.exponential(config.lambdaTempsAttente)))

        return ind,

    def croiser_individus(ind1: Individu, ind2: Individu):
        enfant1 = Individu(ind1)
        enfant2 = Individu(ind2)
        # Croisement des étapes
        lot = random.choice(probleme.lots)
        enfant1.etapesLots[lot] = ind2.etapesLots[lot]
        enfant2.etapesLots[lot] = ind1.etapesLots[lot]
        for mot in probleme.motrices:
            # Extraction des données à transférer
            transfertsInd1 = [(i, mvt) for i, mvt in enumerate(ind1.mvtsMotrices[mot]) if mvt.lot == lot]
            transfertsInd2 = [(i, mvt) for i, mvt in enumerate(ind2.mvtsMotrices[mot]) if mvt.lot == lot]
            for i, _ in reversed(transfertsInd1):
                del enfant1.mvtsMotrices[mot][i]
            for i, _ in reversed(transfertsInd2):
                del enfant2.mvtsMotrices[mot][i]
            # Transfert des mvts de ind1 vers ind2
            for i, mvt in transfertsInd1:
                insertAt = min(i, len(enfant2.mvtsMotrices[mot]))
                enfant2.mvtsMotrices[mot].insert(insertAt, mvt)
            # Transfert des mvts de ind2 vers ind1
            for i, mvt in transfertsInd2:
                insertAt = min(i, len(enfant1.mvtsMotrices[mot]))
                enfant1.mvtsMotrices[mot].insert(insertAt, mvt)

        return creator.Individual(enfant1), creator.Individual(enfant2)

    @lru_cache
    def evaluer_individu(ind: Individu):
        sol = simuler_individu(ind, tracage=False)
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
        blocagesItineraires = defaultdict(list)                     # Blocages temporaires générés par les itinéraires des motrices

        tempsAttenteMots = dict.fromkeys(probleme.motrices, 0)      # Temps d'attente restant pour chaque motrice
        tempsAttenteLots = dict.fromkeys(probleme.lots, 0)          # Temps d'attente restant pour chaque lot (lors du désattelage)
        nbWagonsAtteles = dict.fromkeys(probleme.motrices, 0)       # Nombre de wagons attelés pour chaque motrice

        objCoutsLogistiques = 0         # Objectif de coûts d'exploitation de la solution
        objDeplacement = 0              # Objectif de durée de parcours pour l'ensemble des motrices
        tracageMots = {}
        tracageLots = {}
        tracageAttelages = {}
        lotsValides = dict.fromkeys(probleme.lots, False)
        nbLotsValides = 0

        # Initialisation des derniers noeuds
        for mot in probleme.motrices:
            derniersNoeudsMots[mot] = mot.noeudOrigine
        for lot in probleme.lots:
            derniersNoeudsLots[lot] = lot.noeudOrigine

        # Initialisation du traçage (si activé)
        if tracage:
            tracageMots = {mot: [None] * config.horizonTemp for mot in probleme.motrices}       # Copie par compréhension de liste pour éviter de créer des référence vers la même liste avec fromkeys
            tracageLots = {lot: [None] * config.horizonTemp for lot in probleme.lots}
            tracageAttelages = {lot: [None] * config.horizonTemp for lot in probleme.lots}


        # -------------------------------------------------
        # Définition des actions, exécutées à la fin de l'itinéraire des mouvements
        # -------------------------------------------------
        def action_recuperer(mot, lot):
            attelages[lot] = mot
            tempsAttenteMots[mot] += config.dureeAttelage

            # Gestion des dépassements de capacité pour les motrices
            nbWagonsAtteles[mot] += lot.taille
            if nbWagonsAtteles[mot] > mot.capacite:
                objCoutsLogistiques += (nbWagonsAtteles - mot.capacite) * config.coeffDepassementCapaMotrice
            return True

        def action_deposer(mot, lot):
            attelages[lot] = None
            derniersNoeudsLots[lot] = derniersNoeudsMots[mot]                    # Mise à jour de la position du lot avant son désattelage
            tempsAttenteMots[mot] += config.dureeDesattelage
            tempsAttenteLots[lot] += config.dureeDesattelage                     # Le lot est également bloqué durant son désattelage
            nbWagonsAtteles[mot] -= lot.taille
            indicesEtapesActuelles[lot] += 1
            return True
        

        # -------------------------------------------------
        # Fonction pour le nettoyage des blocages temporaires
        # -------------------------------------------------
        def nettoyer_blocages_itineraires(t):
            """Nettoie les blocages échus, pour chaque arc."""
            if t % config.dureeConservationBlocages != 0:
                return
            for arete in list(blocagesItineraires.keys()):
                blocagesItineraires[arete] = [
                    sillon for sillon in blocagesItineraires[arete]
                    if sillon.tempsFin >= t
                ]


        # -------------------------------------------------
        # Boucle temporelle
        # -------------------------------------------------
        objCoutsLogistiques += len(probleme.motrices) * config.coutFixeParMotrice       # Application des coûts fixes
        for t in range(config.horizonTemp):
            # Nettoyage des blocages d'itinéraires
            nettoyer_blocages_itineraires(t)

            # Initialisation de l'occupation
            occupationNoeuds = {}                                       # Nombre d'éléments occupant chaque noeud du réseau
            def ajouter_occupation(noeud, occ):
                if not noeud in occupationNoeuds:
                    occupationNoeuds[noeud] = 0
                occupationNoeuds[noeud] += occ

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
                # Si la motrice est en attente, elle n'est pas évaluée à cette instant. Cependant, elle rentre dans le calcul de l'occupation
                if tempsAttenteMots[mot] > 0:
                    tempsAttenteMots[mot] -= 1
                    ajouter_occupation(derniersNoeudsMots[mot], mot.taille)
                    continue

                # Si tous les mouvements ont été traitées, la motrice a terminé son travail
                if indicesMvtsActuels[mot] >= len(ind.mvtsMotrices[mot]):
                    if mot.retourBase and derniersNoeudsMots[mot] != mot.noeudOrigine:      # Si la motrice doit rentrer à la base
                        noeudCible = mot.noeudOrigine
                        mvtActuel = None                # Le retour à la base est un déplacement spécial en dehors des mouvements de l'individu
                    else:
                        continue        # Plus de traitement pour la motrice

                else:
                    mvtActuel = ind.mvtsMotrices[mot][indicesMvtsActuels[mot]]
                    lotCible = mvtActuel.lot
                    noeudCible = ind.etapesLots[lotCible][mvtActuel.etape]

                # Si la motrice n'est pas à destination, génération d'un nouvel itinéraire
                if derniersNoeudsMots[mot] != noeudCible and not itinerairesActuels.get(mot, None):
                    def cout_arc(u, v, t, mot):
                        # Vérification des blocages initiaux du problème
                        for sillon in probleme.blocages.get((u, v), []):
                            if sillon.tempsDebut <= t < sillon.tempsFin and sillon.motrice != mot:
                                return maxsize     # Nombre très grand
                        # Vérification des blocages ajoutés par les itinéraires
                        for sillon in blocagesItineraires.get((u, v), []):
                            if sillon.tempsDebut <= t < sillon.tempsFin and sillon.motrice != mot:
                                return maxsize
                        return probleme.graphe.edges[u, v]['weight']
                    
                    # Calcul de l'itinéraire
                    itineraire, rebroussements = dijkstra_special(
                        probleme.graphe, 
                        derniersNoeudsMots[mot], 
                        noeudCible, 
                        t, 
                        fonctionCout=lambda u, v, instant: cout_arc(u, v, instant, mot), 
                        dureeRebroussement=config.dureeRebroussement)
                    itinerairesActuels[mot] = itineraire

                    # Si aucun itinéraire n'est disponible, la motrice est en attente à cet instant
                    if not itineraire:
                        continue

                    # S'il y a un temps d'attente attachée à un mouvement: ajout aux attentes de la motrice (avant début de l'itinéraire)
                    if mvtActuel and mvtActuel.attente > 0:
                        tempsAttenteMots[mot] += mvtActuel.attente

                    # Blocage des arcs de l'itinéraire
                    for i in range(len(itineraire)-1):
                        u = itineraire[i]               # Extraction des information
                        v = itineraire[i+1]
                        debutParcours = round(u[1])
                        finParcours = round(v[1])

                        blocagesItineraires[(u[0], v[0])].append(Sillon(u[0], v[0], debutParcours, debutParcours+config.ecartementMinimal, mot))        # Ajout des sillons bloqués  
                        if probleme.graphe.edges[u[0], v[0]]['exploit'] == 'Simple':
                            # Blocage du sens opposé dans le cas où l'exploitation est à voie unique
                            blocagesItineraires[(v[0], u[0])].append(Sillon(v[0], u[0], debutParcours, finParcours+config.ecartementMinimal, mot))
                        # TODO: Traiter les cas où la durée est à 0 (passage immédiatement au mouvement suivant. Dans les faits, avoir un itinéraire complet avec un coût nul est très rare) 

                    # Traitement des rebroussements : blocage de tous les arcs connexes
                    for r in rebroussements:
                        noeudReb = r[0]
                        for voisin in probleme.graphe.neighbors(noeudReb):
                            blocagesItineraires[(noeudReb, voisin)].append(Sillon(noeudReb, voisin, r[1], r[1] + config.dureeRebroussement, mot))
                            blocagesItineraires[(voisin, noeudReb)].append(Sillon(voisin, noeudReb, r[1], r[1] + config.dureeRebroussement, mot))

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

                # Application de l'action liée au mouvement (pas d'action dans le cas d'un retour base)
                if (not mvtActuel is None) and derniersNoeudsMots[mot] == noeudCible:
                    actionValide = False            
                    # Pour Récupérer: vérifie que le lot est bien à l'étape cible, n'est pas en attente, et que la commande a bien commencé
                    if (mvtActuel.type == TypeMouvement.Recuperer 
                        and indicesEtapesActuelles[mvtActuel.lot] == mvtActuel.etape
                        and tempsAttenteLots[mvtActuel.lot] == 0 
                        and t >= mvtActuel.lot.debutCommande):
                        actionValide = action_recuperer(mot, mvtActuel.lot)
                    # Pour Déposer: vérifie que le lot est bien attelé à la motrice
                    elif mvtActuel.type == TypeMouvement.Deposer and attelages[mvtActuel.lot] == mot:
                        actionValide = action_deposer(mot, mvtActuel.lot)

                    # Passage au mouvement suivant
                    if actionValide:
                        indicesMvtsActuels[mot] += 1

            # Simulation des lots
            for lot in probleme.lots:
                # Mise à jour de la position du lot par rapport à son attelage
                if attelages.get(lot, None):
                    derniersNoeudsLots[lot] = derniersNoeudsMots[attelages[lot]]

                # Calcul de l'occupation des noeuds du réseau
                if attelages.get(lot, None) is None:
                    noeudLot = derniersNoeudsLots[lot]
                    ajouter_occupation(noeudLot, lot.taille)

                # Décrémentation du temps d'attente du lot
                if tempsAttenteLots[lot] > 0:
                    tempsAttenteLots[lot] -= 1

                # Si le lot n'a pas encore atteint sa destination (étape suivante), pas de traitements supplémentaires
                if attelages.get(lot, None) or derniersNoeudsLots[lot] != lot.noeudDest:
                    continue

                # Validation du lot
                if not lotsValides[lot]:
                    if t >= lot.finCommande:        # Application de la pénalité de retard
                        objCoutsLogistiques += (t - lot.finCommande) * config.coeffRetard           # La pénalité correspond au nombre d'instants écoulé depuis la fin de la commande
                    nbLotsValides += 1
                    lotsValides[lot] = True

            # Pénalisation des excès de capacité
            for noeud, occ in occupationNoeuds.items():
                objCoutsLogistiques += config.coeffDepassementCapaNoeud * max(occ - probleme.graphe.nodes[noeud]['capacite'], 0)

        # Finalisation de la simulation
        for lot in probleme.lots:
            if not lotsValides[lot]:
                objDeplacement = config.coutMax     # Neutralisation de l'objectif de déplacement : pas d'importance tant que tous les lots ne sont pas livrés
                objCoutsLogistiques += (config.horizonTemp - lot.finCommande) * config.coeffRetard      # Ajout du retard maximal par rapport à l'horizon temporel
        sol = Solution((objDeplacement, objCoutsLogistiques), tracageMots, tracageLots, tracageAttelages, ind.etapesLots)
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
    try:
        for gen in range(nbGenerations):
            progeniture = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
            for ind in progeniture:
                ind.fitness.values = toolbox.evaluate(ind)
            pop = toolbox.select(pop + progeniture, k=len(pop))

            # Affichage des statistiques de la population
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(progeniture), **record)
        
            info(logbook.stream)
    except KeyboardInterrupt:
        print("Arrêt demandé, terminaison de l'algorithme.")

    # Extraction du front de pareto et sélection d'une solution
    frontPareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    sol = simuler_individu(frontPareto[0], tracage=True)        # Sélection d'un individu sur le front
    info(f'Solution sélectionnée: {sol.objs}')
    return sol


# ---------------------------------------
# Affichage
# ---------------------------------------
def couleur_aleatoire():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def create_colored_pin(hex_color, size=(50, 100)):
    """Crée un pin avec contour noir et renvoie un Data URL."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.polygon([
        (w/2, h),            
        (w-6, w-6),          
        (6, w-6)             
    ], fill=hex_color, outline="black", width=3)

    # Encodage en base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def generer_animation(probleme: Probleme, solution: Solution, config: Config,
                      instantDebut=datetime(2025, 1, 1, 8, 0, 0),
                      pasTemps=timedelta(minutes=1)):

    graphe = probleme.graphe
    map = generer_carte(graphe, config.latMin, config.latMax, config.lonMin, config.lonMax, aretesVisibles=False, couleurAretes='black', supprimerCouleursNoeuds=True)

    couleursMots = {mot: couleur_aleatoire() for mot in probleme.motrices}
    features = []

    # Traçage des motrices
    for mot in probleme.motrices:    
        coords = []
        instants = []
        for pas, noeud in enumerate(solution.tracageMots[mot]):
            lat, lon = graphe.nodes[noeud]['lat'], graphe.nodes[noeud]['lon']
            coords.append([lon, lat])  # Folium attend (lon, lat)
            t = instantDebut + pas * pasTemps
            instants.append(t.isoformat())

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "times": instants,
                "icon": "marker",
                "iconstyle": {
                    "iconUrl": create_colored_pin(couleursMots[mot]),
                    "iconSize": [25, 41],
                    "iconAnchor": [12, 41],
                    "popupAnchor": [1, -34]
                },
                "style": {"color": couleursMots[mot]},
                "popup": str(mot)
            }
        }
        features.append(feature)

    # Traçage des lots
    for lot in probleme.lots:
        coords = []
        instants = []
        for pas, noeud in enumerate(solution.tracageLots[lot]):
            lat, lon = graphe.nodes[noeud]['lat'], graphe.nodes[noeud]['lon']
            coords.append([lon, lat])
            t = instantDebut + pas * pasTemps
            instants.append(t.isoformat())

        # On crée une LineString invisible pour l'animation du marqueur
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "times": instants,
                "style": {"color": "rgba(0,0,0,0)"},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": 'red',
                    "fillOpacity": 1,
                    "stroke": True,
                    "color": "black",   # contour noir
                    "weight": 2,        # épaisseur du contour
                    "radius": 10
                },
                "popup": str([str(n) + ' | ' + lib_noeud(probleme.graphe, n) for n in solution.etapesLots[lot]])
            }
        })


    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    TimestampedGeoJson(
        data=geojson,
        period="PT1M",
        add_last_point=True,
        auto_play=True
    ).add_to(map)

    return map



# ---------------------------------------
# Processus principal
# ---------------------------------------
def main():
    # Chargement de la config
    config = Config()
    initialiser_logs(config)

    # Initialisation des données
    graphe = importer_graphe('graphe_ferroviaire.graphml')
    
    # Génération automatique d'un problème
    noeudsCapa = [n for n, d in graphe.nodes(data=True) if d.get('capacite') > 0]
    mots = []
    for m in range(5):
        mots.append(Motrice(len(mots), random.choice(noeudsCapa), retourBase=True))
    lots = []
    for l in range(30):
        dep, arr = random.sample(noeudsCapa, 2)
        arr = 497       # Triage Lyon sud
        lots.append(Lot(len(lots), dep, arr, 0, 1))
    
     # Construction du problème
    probleme = Probleme(graphe, mots, lots)

    # Ajout des blocages
    # probleme.ajouter_blocage(Sillon(400, 132, 0, 1000, probleme.motrices[1]))
    # probleme.ajouter_blocage(Sillon(132, 400, 0, 1000, probleme.motrices[1]))

    # Résolution du problème
    sol = resoudre_probleme(config, probleme)
    
    # Génération de l'animation
    map = generer_animation(probleme, sol, config)
    map.save("animation_trains.html")

main()