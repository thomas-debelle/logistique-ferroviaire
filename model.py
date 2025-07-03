from igraph import Graph
import igraph
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from enum import Enum
from typing import Dict, List
import random
import yaml
import logging
import glob
import os
from math import floor
import datetime
from logging import info, warning, error
import copy
from deap import base, creator, tools, algorithms
import numpy as np
import sys

def pause_and_exit():
    info("Appuyez sur une touche pour continuer...")
    input()
    exit()


class Config:
    def __init__(self) -> None:
        """
        noeudsGare: liste des noeuds du graphe correspondant à des gares, où les trains peuvent s'arrêter et déposer des wagons.
        motrices: dictionnaire associant un noeud de départ à chaque identifiant de motrice.
        """
        self.dossierLogs = 'logs'
        self.fichierLogs = None
        self.nbLogsMax = 10

        self.nbGenerations = 100
        self.taillePopulation = 150
        self.cxpb = 0.85
        self.mutpb = 0.1
        
        self.horizonTemporel = 500
        self.coutUnitaireRequetes = 1
        self.coutUnitaireAttelage = 1
        self.coutUnitaireDeplacement = 1

class Probleme:
    def __init__(self):
        self.graphe: Graph = None
        self.noeudsGare = []
        self.requetes = {}
        self.motrices = {}
        self.wagons = {}

class TypeMission(Enum):
    Recuperer = 1
    Deposer = 2
    Attendre = 3

class Mission:
    def __init__(self, typeMission: TypeMission, wagon: int = -1, noeud: int = -1, duree: int = -1) -> None:
        """
        Pour certaine missions (par exemple celles de type "Récupérer") le paramètre de noeud n'est pas nécessaire et est déterminé automatiquement dans l'algorithme.
        """
        self.typeMission = typeMission
        self.wagon = wagon
        self.noeud = noeud
        self.duree = duree

    def __str__(self) -> str:
        if self.typeMission == TypeMission.Recuperer:
            return f"(Réc. W{self.wagon})"
        elif self.typeMission == TypeMission.Deposer:
            return f"(Dép. W{self.wagon} en N{self.noeud})"
        elif self.typeMission == TypeMission.Attendre:
            return f"(Att. {self.duree} en N{self.noeud})"
        else:
            return ""

    def __repr__(self) -> str:
        return str(self)

""""
Dictionnaire associant une séquence de missions à chaque identifiant de motrice. 
"""
Solution = Dict[int, List[Mission]]

class Requete:
    """
    Classe représentant une requête anonyme, qui doit être accomplie pour un wagon spécifié dans la config du problème.
    """
    def __init__(self, noeud, dureeTransbordement, tempsDebut, tempsFin) -> None:
        """
        noeud: noeud cible de la requête.
        dureeTransbordement: temps d'attente du wagon au noeud cible.
        tempsDebut: début de la fenêtre de temps durant laquelle la requête est valide.
        tempsFin: fin de la fenêtre de temps durant laquelle la requête est valide. 
        """
        self.noeud = noeud
        self.dureeTransbordement = dureeTransbordement
        self.tempsDebut = tempsDebut
        self.tempsFin = tempsFin

    def __repr__(self):
        return '(' + str(self.noeud) + ', ' + str(self.dureeTransbordement) + ', ' + str(self.tempsDebut) + ', ' + str(self.tempsFin) + ')' 



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
    fichiersLogs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(fichiersLogs) > config.nbLogsMax:
        fichiersASupprimer = fichiersLogs[:len(fichiersLogs) - config.nbLogsMax]
        for fichier in fichiersASupprimer:
            os.remove(fichier)
            info(f"Suppression de {fichier}.")


def terminer_programme(config: Config):
    """
    Opérations à la fin du programme.
    """
    if not config.fichierLogs is None:
        config.fichierLogs.close()


def charger_config(cheminFichierConfig: str):
    config = Config()

    with open(cheminFichierConfig, encoding='utf-8-sig') as fichier:
        try:
            donnees = yaml.safe_load(fichier)
        except yaml.YAMLError as exc:
            error(exc)
            pause_and_exit()

        try:
            # Lecture des données de base
            config.dossierLogs = str(donnees['dossierLogs'])
            config.nbLogsMax = int(donnees['nbLogsMax'])

            config.nbGenerations = int(donnees['nbGenerations'])
            config.taillePopulation = int(donnees['taillePopulation'])
            config.cxpb = float(donnees['cxpb'])
            config.mutpb = float(donnees['mutpb'])

            config.horizonTemporel = int(donnees['horizonTemporel'])
            config.coutUnitaireRequetes = int(donnees['coutUnitaireRequetes'])
            config.coutUnitaireAttelage = int(donnees['coutUnitaireAttelage'])
            config.coutUnitaireDeplacement = int(donnees['coutUnitaireDeplacement'])
        except KeyError as exc:
            error(f"{exc.args[0]} est absent de la configuration.")
            pause_and_exit()

        return config
    

def charger_probleme(cheminFichierProbleme: str):
    def charger_graphe(dictGraphe: dict) -> Graph:
        aretes = []
        poids = []

        for src, dsts in dictGraphe.items():        # Construction des arêtes
            for dst, p in dsts.items():
                aretes.append((src, dst))
                poids.append(p)

        noeuds = sorted(set([n for arete in aretes for n in arete]))            # Extraction de tous les noeuds utilisés
        indexNoeuds = {noeud: id for id, noeud in enumerate(noeuds)}            # Création d'un mapping d'indices pour igraph
        indexAretes = [(indexNoeuds[src], indexNoeuds[dst]) for src, dst in aretes]

        g = Graph(directed=True)
        g.add_vertices(len(noeuds))
        g.add_edges(indexAretes)
        g.es['weights'] = [int(p) for p in poids]       # Conversion des poids en entiers (les durées et instants sont des entiers naturels)
        g.vs['name'] = [str(n) for n in noeuds]
        g.vs['label'] = [str(n) for n in noeuds]

        return g
    
    # Construction du problème
    probleme = Probleme()

    with open(cheminFichierProbleme, encoding='utf-8-sig') as fichier:
        try:
            donnees = yaml.safe_load(fichier)
        except yaml.YAMLError as exc:
            error(exc)
            pause_and_exit()

        try:
            # Lecture du graphe
            dictGraphe = {int(k): {int(t): float(w) for t, w in v.items()} for k, v in donnees["graphe"].items()}
            probleme.graphe = charger_graphe(dictGraphe)

            # Lecture des noeuds gare
            if not (donnees['noeudsGare'] is None):
                probleme.noeudsGare = [int(n) for n in donnees['noeudsGare']]

            # Lecture des requêtes
            if not (donnees['requetes'] is None):
                probleme.requetes = {
                    w: [Requete(*r) for r in listeRequetes]
                    for w, listeRequetes in donnees['requetes'].items()
                }

            # Lecture des motrices
            if not (donnees['motrices'] is None):
                probleme.motrices = dict(zip([int(k) for k in list(donnees['motrices'].keys())], [int(v) for v in list(donnees['motrices'].values())]))

            # Lecture des wagons
            if not (donnees['wagons'] is None):
                probleme.wagons = dict(zip([int(k) for k in list(donnees['wagons'].keys())], [int(v) for v in list(donnees['wagons'].values())]))

            # Vérifications supplémentaires
            # Vérifie que tous les wagons soient présents dans les requêtes
            for w in probleme.wagons.keys():
                if not w in probleme.requetes.keys():
                    probleme.requetes[w] = []             # Initialisation des requêtes avec une liste vide

        except KeyError as exc:
            error(f"{exc.args[0]} est absent du problème.")
            pause_and_exit()

        return probleme
    

def resoudre_probleme(config: Config, probleme: Probleme):
    idMotrices = list(probleme.motrices.keys())
    idWagons = list(probleme.wagons.keys())

    def sol_to_ind(solution: Solution):
        donneesInd = [mission for missions in solution.values() for mission in missions]
        structure = [(motrice, len(missions)) for motrice, missions in solution.items()]

        ind = creator.Individual(donneesInd)
        ind.structure = structure                   # Pour reconstruire la solution ultérieurement
        return ind

    def ind_to_sol(ind):
        solution = {}
        i = 0
        for motrice, length in ind.structure:
            solution[motrice] = ind[i:i+length]
            i += length
        return solution
    
    def generer_individu():
        sol: Solution = dict.fromkeys(idMotrices, [])
        return sol_to_ind(reparer_solution(sol))
    
    def reparer_solution(sol: Solution):
        """
        Parcours les missions de chaque motrice pour vérifier la causalité (récupération des wagons avant dépose, etc).
        On vérifie également que toutes les requêtes soient traitées par la solution. Si certaines requêtes ne sont pas traitées, on ajoute des missions à des motrices au hasard.
        Supprime les missions invalides ou redondantes.
        """
        # Parcours de toutes les motrices
        indicesRequetesValidees = dict.fromkeys(idWagons, 0)
        for m in idMotrices:
            nbMissions = len(sol[m])
            wagonsAtteles = dict.fromkeys(idWagons, False)
            posASupprimer = [False] * nbMissions

            # Parcours de toutes les missions à partir de leurs positions
            for pos in range(nbMissions):
                mission = sol[m][pos]
                w = mission.wagon

                # Redondance des mission Récupérer
                if mission.typeMission == TypeMission.Recuperer:
                    if wagonsAtteles[w]:                        # Si le wagon est déjà attelé, la mission est redondante
                        posASupprimer[pos] = True
                        continue
                    wagonsAtteles[w] = True                     # Sinon, on met à jour l'attelage

                # Redondance des missions Déposer
                elif mission.typeMission == TypeMission.Deposer:
                    if not wagonsAtteles[w]:                    # Si le wagon n'est pas attelé, la mission est redondante
                        posASupprimer[pos] = True
                        continue 
                    wagonsAtteles[w] = False                    # Sinon, on met à jour l'attelage et on vérifie d'autres éléments relatifs à la mission
                    noeudMission = mission.noeud
                    noeudRequete = probleme.requetes[w][indicesRequetesValidees[w]].noeud if indicesRequetesValidees[w] < len(probleme.requetes[w]) else -1
                    
                    # Si la mission valide la requête (noeudMission == noeudRequete) alors on avance l'indice de requêtes validées
                    if noeudRequete != -1 and noeudMission == noeudRequete:
                        indicesRequetesValidees[w] += 1

                # Redondance des missions Attendre
                elif mission.typeMission == TypeMission.Attendre:
                    if pos == 0:
                        continue
                    missionPrecedente = sol[m][pos-1]                   # Si la mission précédente était aussi une mission Attendre au même noeud, alors on consolide les mission en une seule
                    if missionPrecedente.typeMission == TypeMission.Attendre and mission.noeud == missionPrecedente.noeud:
                        posASupprimer[pos] = True
                        missionPrecedente.duree += mission.duree        # Ajout de la durée de la mission actuelle à la mission précédente pour réaliser la consolidation
                

            # Suppression des missions redondantes. Les positions sont parcourues dans l'ordre croissant (important pour le décalage des indices).
            posSupprimees = 0
            for i in range(nbMissions):
                if not posASupprimer[i]:
                    continue
                sol[m].pop(i-posSupprimees)                 # Gestion des décalages d'indice lors de la suppression de missions
                posSupprimees += 1

            # Ajout de missions de Dépose finales pour les wagons encore attelés
            for w, b in wagonsAtteles.items():
                if not b:       # Si le wagon n'est pas attelé à cette motrice à la fin de la simulation, on passe au suivant
                    continue
                noeudDepose = random.choice(probleme.noeudsGare)
                sol[m].append(Mission(TypeMission.Deposer, w, noeudDepose))


        # Pour chaque requête non vérifiée, ajout des missions sur une motrice au hasard
        for w in idWagons:
            while indicesRequetesValidees[w] < len(probleme.requetes[w]):
                requete = probleme.requetes[w][indicesRequetesValidees[w]]
                noeudRequete = requete.noeud
                m = random.choice(idMotrices)

                sol[m].append(Mission(TypeMission.Recuperer, w, -1))
                sol[m].append(Mission(TypeMission.Deposer, w, noeudRequete))
                indicesRequetesValidees[w] += 1
        
        return sol
    
    def muter_individu(ind):
        """
        Mutation: ajout (+1), suppression (-1) ou déplacement (0) d'une mission dans la chronologie pour une motrice choisie au hasard.
        La précédence n'est pas vérifiée à la mutation mais à la réparation: les missions inutiles (deux récup successives ou deux déposes successives) sont supprimées.
        """
        sol = ind_to_sol(ind)

        m = random.choice(idMotrices)
        pos = random.randint(0, len(sol[m])-1) if len(sol[m]) > 0 else 0
        op = random.choice([0, 1, 2, 3]) if len(sol[m]) > 0 else 1                # Sélection d'une opération (1 si la motrice n'a aucune mission)
        
        # Suppression de la mission à la position pos
        if op == 0:
            sol[m].pop(pos)                         # Les réparations permettront de respecter la causalité si une récupération est supprimée
        # Insertion d'une mission aléatoire à la position pos
        elif op == 1:
            wagon = random.choice(idWagons)
            subOp = random.choice([-1, 0, +1])      # Sélection d'une sous-opération, avec l'insertion d'une mission Ajout, Attendre (avec durée unitaire) ou Récupérer.
            mission = None
            if subOp == -1:
                mission = Mission(TypeMission.Deposer, wagon, random.choice(probleme.noeudsGare))
            elif subOp == 0:
                mission = Mission(TypeMission.Attendre, wagon, random.choice(probleme.noeudsGare))
                mission.duree = 1
            else:       # subOp == +1
                mission = Mission(TypeMission.Recuperer, wagon, -1)

            sol[m].insert(pos, mission)
        # Echange de deux missions
        else:   # op == 0
            if len(sol[m]) > 1:
                # Extraction de deux positions
                autresPos = list(range(len(sol[m])))
                autresPos.remove(pos)
                pos1 = pos
                pos2 = random.choice(autresPos)

                # Echange des deux missions dans la chronologie. La causalité sera garantie par les réparations.
                sol[m][pos1], sol[m][pos2] = sol[m][pos2], sol[m][pos1]

        return sol_to_ind(reparer_solution(sol)),
    
    def croiser_individus(ind1, ind2):
        """
        Croisement par segments aléatoire + réparation pour chaque motrice.
        """
        sol1 = ind_to_sol(ind1)
        sol2 = ind_to_sol(ind2)

        enfant1 = dict()
        enfant2 = dict()

        # Sélection d'un segment aléatoire dans chaque parent et échange
        for m in idMotrices:
            # Dans le cas où l'une des solution n'inclut qu'une mission, le croisement n'est pas possible
            if len(sol1[m]) < 2 or len(sol2[m]) < 2:
                enfant1[m] = sol1[m]
                enfant2[m] = sol2[m]
                continue
            
            # Sélection de deux segments pour chaque solution
            a1, b1 = sorted(random.sample(range(len(sol1[m])), 2))
            a2, b2 = sorted(random.sample(range(len(sol2[m])), 2))

            # Echange des deux segments
            enfant1[m] = sol1[m][:a1] + sol2[m][a2:b2] + sol1[m][b1:]
            enfant2[m] = sol2[m][:a2] + sol1[m][a1:b1] + sol2[m][b2:]

        return sol_to_ind(reparer_solution(enfant1)), sol_to_ind(reparer_solution(enfant2))

    def evaluer_individu(ind):
        return evaluer_solution(ind_to_sol(ind))
    
    def evaluer_solution(sol: Solution):
        """
        Simule la solution instant par instant pour en calculer le score.
        Les positions cibles des missions de Récupération sont déterminées dynamiquement à partir des positions des wagons.
        Une solution est une séquence de missions imposées à chaque motrice. Si un wagon arrive trop tôt dans son noeud de destination, il est mis en attente.
        """
        # Variables de simulation        # TODO: éviter de tout redéclarer à chaque appel de évaluer_solution.
        dernierNoeudMotrices = copy.deepcopy(probleme.motrices)
        dernierNoeudWagons = copy.deepcopy(probleme.wagons)
        prochainNoeudMotrices = copy.deepcopy(probleme.motrices)
        prochainNoeudWagons = copy.deepcopy(probleme.wagons)
        
        debutAttenteMotrices = {m: -1 for m in idMotrices}                          # Instant de début de l'attente de chaque motrice. Si la valeur est à -1, la motrice n'est pas en attente.
        debutTransbordements = {w: -1 for w in idWagons}                            # Instant de début du transbordement de chaque wagon. Si la valeur est à -1, aucun transbordement n'est en cours.
        wagonsBloques = {w: False for w in idWagons}                                # La valeur associée à chaque wagon passe à True lorsqu'ils sont dans l'attente d'être transbordés. Ne pas confondre les wagons en attente et les wagons en cours de transbordement.
        attelages = {w: -1 for w in idWagons}                                       # Associe une motrice (valeur) à chaque wagon (clé). Si l'attelage est <0, le wagon n'est attelé à aucune motrice.
        majAttelages = set()                                                        # Mises à jours à appliquer aux attelages à la fin du traitement de toutes les motrices

        # Variables de gestion des requêtes et des missions
        indicesMissionsActuelles = {m: 0 for m in idMotrices}                       # Indices des missions actuellement évaluées
        indicesRequetesActuelles = {w: 0 for w in idWagons}                         # Indices des requêtes actuellement évaluées
        debutMissionsActuelles = {m: 0 for m in idMotrices}                         # Instant de début des missions actuellement simulées

        # Objectifs
        objCoutsRequetes = 0                                                        # Coût de l'avance et du retard sur les requêtes (à minimiser)
        objCoutsLogistiques = 0                                                     # Coût logistique total de la solution (à minimiser)
        
        # Boucle principale: simulation de la solution instant par instant
        t = -1      # Garantit que t=0 à la première itération
        nbMissionsTotal = sum([len(l) for l in sol.values()])
        nbRequetesTotal = sum([len(l) for l in probleme.requetes.values()])
        nbMissionsEvalues = 0
        nbRequetesEvaluees = 0
        while (nbMissionsEvalues < nbMissionsTotal or nbRequetesEvaluees < nbRequetesTotal):
            t += 1

            # L'horizon temporel permet d'éviter d'être bloqué indéfiniment dans la boucle (si deux motrices cherchent mutuellement à récupérer un wagon attelé à l'autre, par exemple)
            if t >= config.horizonTemporel:
                return (sys.maxsize, sys.maxsize)           # La solution est complètement invalidée

            # ---------------------------------------------------------
            # Mise à jour des motrices et missions
            # ---------------------------------------------------------
            for m in idMotrices:
                if indicesMissionsActuelles[m] >= len(sol[m]):
                    continue

                # Extraction des informations liées à la motrice
                missionActuelle = sol[m][indicesMissionsActuelles[m]]
                depart = dernierNoeudMotrices[m]
                arrivee = prochainNoeudWagons[missionActuelle.wagon] if missionActuelle.typeMission == TypeMission.Recuperer else missionActuelle.noeud     # Pour une mission de type "Récupérer": la motrice se dirige au même emplacement que son wagon cible. Sinon, elle se rend au noeud renseigné dans la mission.
                wagonsAtteles = [w for w in attelages.keys() if attelages[w] == m]

                # Mise à jour des prochaines positions de la motrice et de ses attelages    # TODO: ne pas faire à chaque itération, seulement au début et au changement de mission. Permet aussi de ne pas recalculer l'arrivée à chaque itération
                prochainNoeudMotrices[m] = arrivee
                for w in wagonsAtteles:            # Extraction des wagons attelés à la motrice
                    prochainNoeudWagons[w] = arrivee

                # Calcul du temps de parcours restant jusqu'au noeud d'arrivée
                tempsParcours = int(probleme.graphe.distances(depart, arrivee, weights='weights')[0][0])
                tempsParcoursRestant = tempsParcours - (t - debutMissionsActuelles[m])


                # Lorsque la motrice atteint sa destination:
                # On tente de réaliser l'action liée à la mission. Si ce n'est pas possible, la mission n'est pas immédiatement validée.
                missionValidee = False
                if tempsParcoursRestant > 0:    # Si tempsRestant < 0, cela signifie que la motrice est arrivée à destination avant t (cela devra être pris en compte s'il y a implémentation d'une trace plus tard).
                    continue            # Si la motrice n'est pas encore à destination (il reste du temps de déplacement) on ne réalise pas d'autre traitement

                # Mise à jour des dernières positions de la motrice et de ses attelages
                dernierNoeudMotrices[m] = arrivee
                for w in wagonsAtteles:            # Extraction des wagons attelés à la motrice
                    dernierNoeudWagons[w] = arrivee

                # Ajout du déplacement à l'objectif de parcours
                objCoutsLogistiques += tempsParcours * config.coutUnitaireDeplacement     # TODO: voir comment cela se comporte si le wagon cible est bloqué lorsque la motrice arrive à destination
                
                # Mise à jour des attelages et validation des missions
                wagonMission = missionActuelle.wagon
                
                # Mission Récupérer
                if (missionActuelle.typeMission == TypeMission.Recuperer):      # Si le wagon n'est pas attelé à une autre motrice ou est déjà attelé, la mission peut être validée
                    wagonEstDisponible = attelages[wagonMission] < 0 and (not wagonsBloques[wagonMission]) and debutTransbordements[wagonMission] < 0     # Vérifie que le wagon ne subit aucune opération
                    if wagonEstDisponible or attelages[wagonMission] == m:      # Si la solution a bien été réparée, la deuxième condition n'est pas nécessaire
                        majAttelages.add((wagonMission, m))                     # Ajout de l'attelage
                        objCoutsLogistiques += config.coutUnitaireAttelage
                        missionValidee = True
                # Mission Déposer
                elif missionActuelle.typeMission == TypeMission.Deposer:
                    if attelages[wagonMission] == m:
                        majAttelages.add((wagonMission, -1))                    # Suppression de l'attelage           
                        objCoutsLogistiques += config.coutUnitaireAttelage
                    missionValidee = True           # La mission est validée dans tous les cas (même si le wagon n'est pas attelé à la motrice)
                # Mission Attendre
                elif missionActuelle.typeMission == TypeMission.Attendre:
                    if debutAttenteMotrices[m] < 0:                     # Mise en attente de la motrice
                        debutAttenteMotrices[m] = t
                    if t - debutAttenteMotrices[m] >= missionActuelle.duree:           # Attend que la durée soit écoulée pour autoriser un déplacement de la motrice
                        missionValidee = True
                        debutAttenteMotrices[m] = -1                    # Remise à 0 du compteur

                # Si la mission a pu être validée, on passe à la suivante. Si, la mission actuelle sera réevaluée au prochain instant (dans le cas d'une mise en attente, par exemple).
                if missionValidee:
                    debutMissionsActuelles[m] = t
                    indicesMissionsActuelles[m] += 1
                    nbMissionsEvalues += 1
                    
            # ---------------------------------------------------------
            # Mise à jour des attelages
            # ---------------------------------------------------------
            for a in majAttelages:      # La mise à jour de l'attelage est réalisée après le traitement de toutes les motrices (pour éviter la reprise immédiate du wagon)
                attelages[a[0]] = a[1]
            majAttelages.clear()

            # ---------------------------------------------------------
            # Mise à jour des wagons et requêtes
            # ---------------------------------------------------------
            for w in idWagons:
                if indicesRequetesActuelles[w] >= len(probleme.requetes[w]):
                    continue

                # Vérifie que le wagon soit arrivé à destination et ne soit plus attelé pour commencer le transbordement. Lorsque le transbordement est terminé, la requête est validée.
                requete = probleme.requetes[w][indicesRequetesActuelles[w]]
                if dernierNoeudWagons[w] != requete.noeud or attelages[w] >= 0:
                    continue

                # Si le wagon est en avance, il est placé en attente de transbordement
                if t < requete.tempsDebut:
                    wagonsBloques[w] = True               # Blocage du wagon en avance
                    objCoutsRequetes += config.coutUnitaireRequetes                   # Application d'une pénalité d'avance
                    continue
                else:
                    wagonsBloques[w] = False              # Déblocage du wagon

                # Début du transbordement
                if debutTransbordements[w] < 0:           # Si le transbordement n'a pas commencé pour ce wagon
                    debutTransbordements[w] = t           # Alors on l'initialise
                
                # Fin du transbordement et validation de la requête
                if t - debutTransbordements[w] >= requete.dureeTransbordement:
                    objCoutsRequetes += max(debutTransbordements[w] - requete.tempsFin, 0) * config.coutUnitaireRequetes          # Ajout d'un coût si la requête est terminée en retard
                    debutTransbordements[w] = -1
                    indicesRequetesActuelles[w] += 1
                    nbRequetesEvaluees += 1
                
        return (objCoutsRequetes, objCoutsLogistiques)


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
    info(f"Taille du front de Pareto : {len(frontPareto)}")
    for ind in frontPareto[:5]:
        info(f"Scores : ({int(ind.fitness.values[0])}, {int(ind.fitness.values[1])})")

    meilleurInd = frontPareto[0]        # ATTENTION: certaines requêtes peuvent être remplies par la solution via le désattelage final, même sans action de dépose
    meilleureSol = ind_to_sol(meilleurInd) 
    return meilleureSol

def afficher_solution(sol: Solution, config: Config, probleme: Probleme):
    idMotrices = probleme.motrices.keys()
    
    info("--------------------------------------------------------")
    info("AFFICHAGE DE LA SOLUTION")
    info("--------------------------------------------------------")
    for m in idMotrices:
        info(f"Motrice {m}:")
        for mis in sol[m]:
            info(f" - {mis}")
        if len(sol[m]) == 0:
            info(" - []")
        

    layout = probleme.graphe.layout('kk')     # TODO: terminer (pour le moment, seulement affichage du problème)
    fig, ax = plt.subplots()
    igraph.plot(
        probleme.graphe,
        target=ax,
        layout=layout,
        vertex_size=30,
        vertex_color="lightblue",
        vertex_label=probleme.graphe.vs['label'],
        edge_arrow_size=0.5
    )
    plt.show()
    pass # TODO: afficher l'évolution de la solution
                


def main():
    # Initialisation de Tkinter (sans interface graphique)
    root = tk.Tk()
    root.withdraw()
    
    # Chargement de la configuration. Les erreurs lors du chargement de la config n'apparaissent pas dans les logs.
    chemierFichierConfig = filedialog.askopenfilename(filetypes=[("Fichiers YAML", ".yaml")], initialdir='.', title="Sélectionnez le fichier contenant la configuration.")
    if len(chemierFichierConfig) == 0: return                   # Arrêt du programme si aucun fichier n'a été sélectionné
    config = charger_config(chemierFichierConfig)
    initialiser_logs(config)

    # Chargement du problème
    cheminFichierProbleme = filedialog.askopenfilename(filetypes=[("Fichiers YAML", ".yaml")], initialdir='.', title="Sélectionnez le fichier contenant le problème.")
    if len(cheminFichierProbleme) == 0: return                   # Arrêt du programme si aucun fichier n'a été sélectionné
    probleme = charger_probleme(cheminFichierProbleme)

    # Résolution du problème
    sol = resoudre_probleme(config, probleme)
    afficher_solution(sol, config, probleme)

    # Fin du programme
    terminer_programme(config)
    pause_and_exit()

main()