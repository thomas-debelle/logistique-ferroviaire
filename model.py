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


class ConfigProbleme:
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
        
        self.nbMissionsMax = 5
        self.horizonTemporel = 500

        self.graphe: Graph = None
        self.noeudsGare = []
        self.requetes = {}
        self.motrices = {}
        self.wagons = {}

        # Constantes
        self.baseScoreRequetes = 10            # Récompense maximale lors de la validation d'une requête dans les temps
        self.coeffPenaliteRetard = 1.0

class TypeMission(Enum):
    Recuperer = 1
    Deposer = 2

class Mission:
    def __init__(self, typeMission: TypeMission, wagon: int, noeud: int = -1) -> None:
        """
        Pour certaine missions (par exemple celles de type "Récupérer") le paramètre de noeud n'est pas nécessaire et est déterminé automatiquement dans l'algorithme.
        """
        self.typeMission = typeMission
        self.wagon = wagon
        self.noeud = noeud

    def __str__(self) -> str:
        return f"({self.typeMission}, {self.wagon}, {self.noeud})"

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



def initialiser_logs(config: ConfigProbleme):
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


def terminer_programme(config: ConfigProbleme):
    """
    Opérations à la fin du programme.
    """
    if not config.fichierLogs is None:
        config.fichierLogs.close()


def charger_config(cheminFichierConfig: str):
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
    

    config = ConfigProbleme()

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

            config.nbMissionsMax = int(donnees['nbMissionsMax'])        # TODO: renommer (car uniquement utilisé à la génération initiale de solutions)
            config.horizonTemporel = int(donnees['horizonTemporel'])


            # Lecture du graphe
            dictGraphe = {int(k): {int(t): float(w) for t, w in v.items()} for k, v in donnees["graphe"].items()}
            config.graphe = charger_graphe(dictGraphe)

            # Lecture des noeuds gare
            if not (donnees['noeudsGare'] is None):
                config.noeudsGare = [int(n) for n in donnees['noeudsGare']]

            # Lecture des requêtes
            if not (donnees['requetes'] is None):
                config.requetes = {
                    w: [Requete(*r) for r in listeRequetes]
                    for w, listeRequetes in donnees['requetes'].items()
                }

            # Lecture des motrices
            if not (donnees['motrices'] is None):
                config.motrices = dict(zip([int(k) for k in list(donnees['motrices'].keys())], [int(v) for v in list(donnees['motrices'].values())]))

            # Lecture des wagons
            if not (donnees['wagons'] is None):
                config.wagons = dict(zip([int(k) for k in list(donnees['wagons'].keys())], [int(v) for v in list(donnees['wagons'].values())]))

            # Vérifications supplémentaires
            # Vérifie que tous les wagons soient présents dans les requêtes
            for w in config.wagons.keys():
                if not w in config.requetes.keys():
                    config.requetes[w] = []             # Initialisation des requêtes avec une liste vide

        except KeyError as exc:
            error(f"{exc.args[0]} est absent de la configuration.")
            pause_and_exit()

        return config
    

def resoudre_probleme(config: ConfigProbleme):
    idMotrices = list(config.motrices.keys())
    idWagons = list(config.wagons.keys())


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
        sol: Solution = dict()

        for m in idMotrices:
            sol[m] = []
            for i in range(random.randint(1, floor(config.nbMissionsMax / 2))):         # /2 car pour chaque récupération, une dépose est ajouée
                wagon = random.choice(idWagons)
                missionRecup = Mission(TypeMission.Recuperer, wagon, -1)                        # Ira récupérer le wagon sur n'importe quel noeud
                missionDepose = Mission(TypeMission.Deposer, wagon, random.choice(config.noeudsGare))
                
                # Insère les nouvelles missions tout en respectant la précédence
                posRecup = random.randint(0, len(sol[m]))                   # Les bornes de randint sont inclues
                sol[m].insert(posRecup, missionRecup)
                posDepose = random.randint(posRecup+1, len(sol[m]))
                sol[m].insert(posDepose, missionDepose)

        return sol_to_ind(reparer_solution(sol))
    
    def reparer_solution(sol: Solution):
        # TODO: parcourir les requêtes et chercher dans les missions si la requête est résolue à un moment donné. Sinon, ajouter des missions.
        # Si un wagon arrive trop tôt à sa cible, la requête est considérée comme résoluee (car le wagon sera alors mis en attente).
        # On ne souhaite vérifier que la séquentialité des wagons. Les éléments temporels seront évalués ultérieurement


        # Parcours des missions de chaque motrice
        indicesRequetes = dict.fromkeys(idWagons, 0)            # Indices des requêtes à vérifier
        for m in idMotrices:
            wagonsAtteles = dict.fromkeys(idWagons, False)
            posASupprimer = []
            for pos in range(len(sol[m])):
                mission = sol[m][pos]
                w = mission.wagon

                # Vérification des missions redondantes
                if mission.typeMission == TypeMission.Recuperer:
                    if wagonsAtteles[w]:
                        posASupprimer.append(pos)           # Si le wagon est déjà attelé, la mission est redondante
                    else:
                        wagonsAtteles[w] = True             # Sinon, on met à jour l'attelage
                elif mission.typeMission == TypeMission.Deposer:
                    if not wagonsAtteles[w]:
                        posASupprimer.append(pos)           # Si le wagon n'est pas attelé, la mission est redondante
                    else:
                        wagonsAtteles[w] = False
                        
                        # Si le déplacement de la mission valide la requête, alors on évalue la requête suivante
                        noeudMission = mission.noeud
                        if indicesRequetes[w] < len(config.requetes[w]) and noeudMission == config.requetes[w][indicesRequetes[w]].noeud:
                            indicesRequetes[w] += 1

            # Ajout de missions de dépose pour les wagons encore attelés
            for w, b in wagonsAtteles.items():
                if not b:       # Si le wagon n'est pas attelé à cette motrice à la fin de la simulation, on passe au suivant
                    continue
                noeudDepose = random.choice(config.noeudsGare)
                sol[m].append(Mission(TypeMission.Deposer, w, noeudDepose))
                

            # Suppression des missions redondantes avec gestion des décalages. Les positions sont triées dans l'ordre croissant.
            posSupprimees = 0               
            for p in posASupprimer:
                sol[m].pop(p-posSupprimees)                 # Gestion des décalages d'indice lors de la suppression de missions
                posSupprimees += 1


        # Pour chaque requête non vérifiée, ajout de missions sur des motrices au hasard 
        for w in idWagons:
            while indicesRequetes[w] < len(config.requetes[w]):
                requete = config.requetes[w][indicesRequetes[w]]
                noeudRequete = requete.noeud
                m = random.choice(idMotrices)

                sol[m].append(Mission(TypeMission.Recuperer, w, -1))
                sol[m].append(Mission(TypeMission.Deposer, w, noeudRequete))
                indicesRequetes[w] += 1
        
        return sol
    
    def muter_individu(ind):
        """
        Mutation: ajout (+1), suppression (-1) ou déplacement (0) d'une mission dans la chronologie pour une motrice choisie au hasard.
        La précédence n'est pas vérifiée à la mutation mais à la réparation: les missions inutiles (deux récup successives ou deux déposes successives) sont supprimées.
        """
        sol = ind_to_sol(ind)

        m = random.choice(idMotrices)
        pos = random.randint(0, len(sol[m])-1) if len(sol[m]) > 0 else 0
        op = random.choice([-1, 0, 1]) if len(sol[m]) > 0 else 1                # Sélection d'une opération (1 si la motrice n'a aucune mission)
        
        if op == -1:
            sol[m].pop(pos)                         # Les réparations permettront de respecter la causalité si une récupération est supprimée
        elif op == +1:
            wagon = random.choice(idWagons)
            missionRecup = Mission(TypeMission.Recuperer, wagon, -1)
            missionDepose = Mission(TypeMission.Deposer, wagon, random.choice(config.noeudsGare))

            sol[m].insert(pos, missionDepose)
            sol[m].insert(pos, missionRecup)        # Insertion de la dépose puis de la récup pour respecter l'ordre. Les missions sont directement consécutives.
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
        # Variable de simulation
        dernierNoeudMotrices = copy.deepcopy(config.motrices)
        dernierNoeudWagons = copy.deepcopy(config.wagons)
        prochainNoeudMotrices = copy.deepcopy(config.motrices)
        prochainNoeudWagons = copy.deepcopy(config.wagons)
        
        attelages = {w: -1 for w in idWagons}                                       # Associe une motrice (valeur) à chaque wagon (clé). Si l'attelage est <0, alors le wagon n'est attelé à aucune motrice.
        debutsTransbordements = {w: -1 for w in idWagons}                           # Instant de début du dernier transbordement de chaque wagon. Si la valeur est à -1, cela signifie qu'aucun transbordement n'est en cours.
        wagonsEnAttente = {w: False for w in idWagons}                              # La valeur associée à chaque wagon passe à True lorsqu'ils sont dans l'attente d'être transbordés.

        # Variable de gestion des requêtes et des missions
        indicesMissionsActuelles = {m: 0 for m in idMotrices}                       # Indices des missions actuellement évaluées
        indicesRequetesActuelles = {w: 0 for w in idWagons}                         # Indices des requêtes actuellement évaluées
        debutsMissionsActuelles = {m: 0 for m in idMotrices}                        # Instant de début des missions actuellement simulées

        # Objectifs
        objRequetes = 0                                                             # Score obtenu en répondant à des requêtes (à maximiser)
        objParcours = 0                                                             # Durée totale de parcours des motrices (à minimiser)
        
        # Boucle principale: simulation de la solution instant par instant
        t = -1      # Garantit que t=0 à la première itération
        nbMissionsTotal = sum([len(l) for l in sol.values()])
        nbRequetesTotal = sum([len(l) for l in config.requetes.values()])
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
                arrivee = prochainNoeudWagons[missionActuelle.wagon] if missionActuelle.typeMission == TypeMission.Recuperer else missionActuelle.noeud     # Pour une mission de type "Récupérer": la motrice se dirige au même emplacement que son wagon cible
                wagonsAtteles = [w for w in attelages.keys() if attelages[w] == m]

                # Mise à jour des prochaines positions de la motrice et de ses attelages    # TODO: ne pas faire à chaque itération, seulement au début et au changement de mission. Permet aussi de ne pas recalculer l'arrivée à chaque itération
                prochainNoeudMotrices[m] = arrivee
                for w in wagonsAtteles:            # Extraction des wagons attelés à la motrice
                    prochainNoeudWagons[w] = arrivee

                # Calcul du temps de parcours restant jusqu'à l'arrivée
                tempsParcours = int(config.graphe.distances(depart, arrivee, weights='weights')[0][0])
                tempsRestant = tempsParcours - (t - debutsMissionsActuelles[m])


                # Lorsque la motrice atteint sa destination:
                # On tente de réaliser l'action liée à la mission. Si ce n'est pas possible, la mission n'est pas immédiatement validée.
                missionValidee = False
                if tempsRestant > 0:    # Si tempsRestant < 0, cela signifie que la motrice est arrivée à destination avant t (cela devra être pris en compte s'il y a implémentation d'une trace plus tard).
                    continue            # Si la motrice n'est pas encore à destination (il reste du temps de déplacement) on ne réalise pas d'autre traitement

                # Mise à jour des dernières positions de la motrice et de ses attelages
                dernierNoeudMotrices[m] = arrivee
                for w in wagonsAtteles:            # Extraction des wagons attelés à la motrice
                    dernierNoeudWagons[w] = arrivee

                # Ajout du déplacement à l'objectif de parcours
                objParcours += tempsParcours     # TODO: voir comment cela se comporte si le wagon cible est bloqué lorsque la motrice arrive à destination
                
                # Mise à jour des attelages
                wagonMission = missionActuelle.wagon
                if (missionActuelle.typeMission == TypeMission.Recuperer):       # Si le wagon n'est pas attelé à une autre motrice ou est déjà attelé, la mission peut être validée
                    wagonEstDisponible = attelages[wagonMission] < 0 and (not wagonsEnAttente[wagonMission]) and debutsTransbordements[wagonMission] < 0     # Vérifie que le wagon ne subit aucune opération
                    if wagonEstDisponible or attelages[wagonMission] == m:
                        attelages[wagonMission] = m
                        missionValidee = True
                elif missionActuelle.typeMission == TypeMission.Deposer:
                    if attelages[wagonMission] == m:
                        attelages[wagonMission] = -1                    # Suppression de l'attelage
                    missionValidee = True                               # La mission est validée dans tous les cas (même si le wagon n'est pas attelé à la motrice)

                # Si la mission a pu être validée, passage à la suivante
                if missionValidee:
                    debutsMissionsActuelles[m] = t
                    indicesMissionsActuelles[m] += 1
                    nbMissionsEvalues += 1

                    if indicesMissionsActuelles[m] >= len(sol[m]):      # Si la dernière mission de la motrice a été terminée, suppression de tous ses attelages.
                        attelages = {k: (-1 if v == m else v) for k, v in attelages.items()}        # Remarque: cela est équivalent à une dépose des wagons sur la dernière position de la motrice.

            # ---------------------------------------------------------
            # Mise à jour des wagons et requêtes
            # ---------------------------------------------------------
            for w in idWagons:
                if indicesRequetesActuelles[w] >= len(config.requetes[w]):
                    continue

                # Vérifie que le wagon soit arrivé à destination et ne soit plus attelé pour commencer le transbordement. Lorsque le transbordement est terminé, la requête est validée.
                requete = config.requetes[w][indicesRequetesActuelles[w]]
                if dernierNoeudWagons[w] != requete.noeud or attelages[w] >= 0:
                    continue

                # Si le wagon est en avance, il est placé en attente de transbordement
                if t < requete.tempsDebut:
                    wagonsEnAttente[w] = True
                    continue                # TODO: voir si besoin d'appliquer une pénalité d'avance (dans ce cas, nécessite de modéliser l'encombrement maximal à chaque noeud)
                else:
                    wagonsEnAttente[w] = False              # Déblocage du wagon

                # Début du transbordement
                if debutsTransbordements[w] < 0:            # Si le transbordement n'a pas commencé pour ce wagon
                    debutsTransbordements[w] = t
                
                # Fin du transbordement et validation de la requête
                if t - debutsTransbordements[w] >= requete.dureeTransbordement:
                    objRequetes += max(debutsTransbordements[w] - requete.tempsFin, 0)          # Ajout d'un coût si la requête est terminée en retard
                    debutsTransbordements[w] = -1
                    indicesRequetesActuelles[w] += 1
                    nbRequetesEvaluees += 1
                
        return (objRequetes, objParcours)


    # Initialisation de l'algorithme génétique
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))     # TODO: configurer les poids
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
    meilleureSol = ind_to_sol(meilleurInd)          # TODO: appliquer un nettoyage, et s'assurer que les scores après nettoyage restent les mêmes.    
    return meilleureSol

def afficher_solution(sol: Solution, config: ConfigProbleme):
    layout = config.graphe.layout('kk')
    fig, ax = plt.subplots()
    igraph.plot(
        config.graphe,
        target=ax,
        layout=layout,
        vertex_size=30,
        vertex_color="lightblue",
        vertex_label=config.graphe.vs['label'],
        edge_arrow_size=0.5
    )
    plt.show()
    pass # TODO: afficher l'évolution de la solution
                
def main():
    # Initialisation de Tkinter (sans interface graphique)
    root = tk.Tk()
    root.withdraw()
    
    # Chargement de la configuration du problème et des logs. Les erreurs lors du chargement de la config n'apparaissent pas dans les logs.
    chemierFichierConfig = filedialog.askopenfilename(filetypes=[("Fichiers YAML", ".yaml")], initialdir='.', title="Sélectionnez le fichier contenant la configuration du problème.")
    if len(chemierFichierConfig) == 0: return                   # Arrêt du programme si aucun fichier n'a été sélectionné
    config = charger_config(chemierFichierConfig)
    initialiser_logs(config)

    # Résolution du problème
    sol = resoudre_probleme(config)
    afficher_solution(sol, config)

    # Fin du programme
    terminer_programme(config)
    pause_and_exit()

main()