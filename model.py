from igraph import Graph
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
        self.nbMissionsMax = 5
        self.dossierLogs = 'logs'
        self.fichierLogs = None
        self.nbLogsMax = 10

        self.graphe: Graph = None
        self.noeudGare = []
        self.requetes = {}
        self.motrices = {}
        self.wagons = {}

class TypeMission(Enum):
    Recuperer = 1
    Deposer = 2

class Mission:
    def __init__(self, typeMission: TypeMission, wagon: int, param: int) -> None:
        self.typeMission = typeMission
        self.wagon = wagon
        self.param = param

    def __str__(self) -> str:
        return f"({self.typeMission}, {self.wagon}, {self.param})"

    def __repr__(self) -> str:
        return str(self)

""""
Dictionnaire associant une séquence de missions à chaque identifiant de motrice. 
Chaque mission est un tuple (TypeMission, int, int), où les entiers sont des paramètres pour la mission à effectuer (pas tous toujours nécessaires).
"""
Solution = Dict[int, List[Mission]]

class Requete:
    """
    Classe représentant une requête anonyme, qui doit être accomplie pour un wagon spécifié dans la config du problème.
    """
    def __init__(self, noeud, tempsDebut, tempsFin) -> None:
        """
        noeud: noeud cible de la requête.
        tempsDebut: début de la fenêtre de temps durant laquelle la requête est valide.
        tempsFin: fin de la fenêtre de temps durant laquelle la requête est valide. 
        """
        self.noeud = noeud
        self.tempsDebut = tempsDebut
        self.tempsFin = tempsFin



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
            config.nbMissionsMax = int(donnees['nbMissionsMax'])        # TODO: supprimer ça (ou alors, le renommer car uniquement utilisé à l'initialisation)
            config.dossierLogs = str(donnees['dossierLogs'])
            config.nbLogsMax = int(donnees['nbLogsMax'])

            # Lecture du graphe
            dictGraphe = {int(k): {int(t): float(w) for t, w in v.items()} for k, v in donnees["graphe"].items()}
            config.graphe = charger_graphe(dictGraphe)

            # Lecture des noeuds gare
            if not (donnees['noeudsGare'] is None):
                config.noeudGare = [int(n) for n in donnees['noeudsGare']]

            # Lecture des requêtes
            if not (donnees['requetes'] is None):
                config.requetes = {w: Requete(r[0], r[1], r[2]) for w, r in donnees['requetes'].items()}

            # Lecture des motrices
            if not (donnees['motrices'] is None):
                config.motrices = dict(zip([int(k) for k in list(donnees['motrices'].keys())], [int(v) for v in list(donnees['motrices'].values())]))

            # Lecture des wagons
            if not (donnees['wagons'] is None):
                config.wagons = dict(zip([int(k) for k in list(donnees['wagons'].keys())], [int(v) for v in list(donnees['wagons'].values())]))

        except KeyError as exc:
            error(f"{exc.args[0]} est absent de la configuration.")
            pause_and_exit()

        return config
    

def resoudre_probleme(config: ConfigProbleme):
    idMotrices = list(config.motrices.keys())
    idWagons = list(config.wagons.keys())
    
    def generer_solution():
        sol: Solution = dict()

        for m in idMotrices:
            sol[m] = []
            for i in range(random.randint(1, floor(config.nbMissionsMax / 2))):         # /2 car pour chaque récupération, une dépose est ajouée
                wagon = random.choice(idWagons)
                missionRecup = Mission(TypeMission.Recuperer, wagon, -1)                        # Ira récupérer le wagon sur n'importe quel noeud
                missionDepose = Mission(TypeMission.Deposer, wagon, random.choice(config.noeudGare))
                
                # Insère les nouvelles missions tout en respectant la précédence
                posRecup = random.randint(0, len(sol[m]))                   # Les bornes de randint sont inclues
                sol[m].insert(posRecup, missionRecup)
                posDepose = random.randint(posRecup+1, len(sol[m]))
                sol[m].insert(posDepose, missionDepose)

        return sol
    
    def muter_solution(sol: Solution):
        """
        Mutation: ajout (+1), suppression (-1) ou déplacement (0) d'une mission dans la chronologie pour une motrice choisie au hasard.
        La précédence n'est pas vérifiée à la mutation mais à la réparation: les missions inutiles (deux récup successives ou deux déposes successives) sont supprimées.
        """
        m = random.choice(idMotrices)
        pos = random.randint(0, len(sol[m])-1) if len(sol[m]) > 0 else 0
        op = random.choice([-1, 0, 1]) if len(sol[m]) > 0 else 1                # Sélection d'une opération (1 si la motrice n'a aucune mission)
        
        if op == -1:
            sol[m].pop(pos)
        elif op == +1:
            wagon = random.choice(idWagons)
            missionRecup = Mission(TypeMission.Recuperer, wagon, -1)
            missionDepose = Mission(TypeMission.Deposer, wagon, random.choice(config.noeudGare))

            sol[m].insert(pos, missionDepose)
            sol[m].insert(pos, missionRecup)        # Insertion de la dépose puis de la récup pour respecter l'ordre. Les missions sont directement consécutives.
        else:   # op == 0
            if len(sol[m]) > 1:
                # Extraction de deux positions
                autresPos = list(range(len(sol[m])))
                autresPos.remove(pos)
                pos1 = pos
                pos2 = random.choice(autresPos)

                # Echange des deux missions dans la chronologie
                sol[m][pos1], sol[m][pos2] = sol[m][pos2], sol[m][pos1]

        return reparer_solution(sol)
    
    def reparer_solution(sol: Solution) -> Solution:
        """
        Répare la solution en supprimant les missions inutiles.
        """
        for m in idMotrices:
            wagonsAtteles = []          # Statut d'attelage des wagons mis à jour à chaque mission de la motrice
            posASupprimer = []
            for pos in range(len(sol[m])):
                if sol[m][pos].typeMission == TypeMission.Recuperer:
                    wagon = sol[m][pos].wagon              # Si le wagon est déjà attelé, il ne peut pas être récupéré. Suppression de la mission.
                    if wagon in wagonsAtteles:
                        posASupprimer.append(pos)
                    else:
                        wagonsAtteles.append(wagon)
                elif sol[m][pos].typeMission == TypeMission.Deposer:
                    wagon = sol[m][pos].wagon
                    if not (wagon in wagonsAtteles):        # Si le wagon n'est pas attelé, il ne peut pas être déposé. Suppression de la mission.
                        posASupprimer.append(pos)
                    else:
                        wagonsAtteles.remove(wagon)

            # Suppression des missions relevées. Les positions à supprimer sont triées dans l'ordre croissant
            posSupprimees = 0               
            for p in posASupprimer:
                sol[m].pop(p-posSupprimees)                 # Gestion des décalages d'indice lors de la suppression de missions
                posSupprimees += 1

        return sol
    
    def croiser_solutions(sol1: Solution, sol2: Solution) -> Solution:
        """
        Croisement par segments + réparation pour chaque motrice.
        """
        enfant1 = dict()
        enfant2 = dict()
        for m in idMotrices:            # TODO: changer la manière de croiser les solutions (la sélection avec nbMin a tendance à réduire la taille des solutions)
            nbMin = min([len(sol1), len(sol2)])
            a, b = sorted(random.sample(range(nbMin), 2))

            enfant1[m] = sol1[m][:a] + sol2[m][a:b] + sol1[m][b:]
            enfant2[m] = sol2[m][:a] + sol1[m][a:b] + sol2[m][b:]

        return reparer_solution(enfant1), reparer_solution(enfant2)


    def evaluer_solution(sol: Solution):
        # Variables pour la simulation de la solution
        nbMissionsTotal = sum([len(l) for l in sol.values()])
        positionsMotrices = copy.deepcopy(config.motrices)                      # Position des motrices à la fin de leur dernière mission. Par défaut, positions initiales dans le problème.
        positionsWagons = copy.deepcopy(config.wagons)                          # Noeud actuel du wagon. Si le wagon se déplace, dernier noeud de passage du wagon.
        positionsCiblesWagons = copy.deepcopy(config.wagons)                    # Noeuds en direction desquels se déplacent les wagons (prochaine position où le wagon est disponible pour l'attelage). Si le wagon ne se déplace pas, noeud actuel du wagon.
        attelages = {m: set() for m in idMotrices}                              # Pour chaque motrice, collection des wagons attelés
        indicesMissionsActuelles = {m: 0 for m in idMotrices}
        motricesEnAttente = set()                                               # Collections des motrices en attente. Ces motrices sont bloquées à leur position actuelle jusqu'à la fin de la prochaine mission de l'ensemble des motrices. Ce comportement permet de gérer les motrices arrivant en avance à une position pour récupérer un wagon.

        # Variable de temps
        instantActuel = 0
        dureeEcouleeMissionsActuelles = {m: 0 for m in idMotrices}              # Durée écoulée depuis que la mission actuelle est active (utilisée pour le calcul des instants)
        
        # Processus principal. Mission actuelle=dernière mission validée. Prochaine mission=prochaine mission validée.
        # TODO: vérifier comment le système se comporte avec deux actions Récupérer ou deux actions Déposer consécutives (probablement aucun impact mais à vérifier)
        # TODO: éviter de recalculer la distance à chaque itération ? (en stockant les valeurs et en recalculant uniquement les wagons impactés par d'autres missions)
        # TODO: organiser et optimiser le code.
        nbMissionsEvalues = 0
        while nbMissionsEvalues < nbMissionsTotal:
            # Recherche de la prochaine mission qui sera réalisée
            tempsRestantMin = sys.maxsize
            motriceProchaineMission = -1
            arriveeProchaineMission = -1
            prochaineMission = None
            for m in idMotrices:
                if (indicesMissionsActuelles[m] >= len(sol[m]) 
                    or m in motricesEnAttente):                                         # Si la motrice m n'a plus de missions ou est en attente, pas de traitement
                    continue
                
                # Extraction des informations pour le calcul d'itinéraire
                i = indicesMissionsActuelles[m]
                mission = sol[m][i]
                depart = positionsMotrices[m]
                if mission.typeMission == TypeMission.Recuperer:
                    arrivee = positionsCiblesWagons[mission.wagon]                      # Extraction de la position du wagon cible
                elif mission.typeMission == TypeMission.Deposer:
                    arrivee = mission.param
                
                # Mise à jour des positions cibles des wagons à partir des attelages
                for w in attelages[m]:
                    positionsCiblesWagons[w] = arrivee

                # Calcul du temps de parcours jusqu'à l'arrivée
                # TODO: gérer les cas où la destination n'est pas accessible (plutôt dans la réparation des solutions ? On pourrait admettre ici que toutes les missions sont réalisables)
                dureeParcours = int(config.graphe.distances(depart, arrivee, weights='weights')[0][0])      # Extraction de la meilleure distance jusqu'à l'arrivée       
                tempsRestant = dureeParcours - dureeEcouleeMissionsActuelles[m]

                # Recherche de la prochaine mission exécutée chronologiquement pour l'ensemble des motrices
                if tempsRestant < tempsRestantMin:
                    tempsRestantMin = tempsRestant
                    motriceProchaineMission = m
                    arriveeProchaineMission = arrivee
                    prochaineMission = mission

            # Mise à jour des positions pour la prochaine mission
            missionConfirmee = True
            positionsMotrices[motriceProchaineMission] = arriveeProchaineMission            # Mise à jour de la position de la motrice
            for w in attelages[motriceProchaineMission]:
                positionsWagons[w] = arriveeProchaineMission

            # Confirmation de la mission et mise à jour de l'attelage
            if prochaineMission.typeMission == TypeMission.Recuperer:                       # Attelage du wagon s'il n'est pas bloqué                   
                wagonBloque = any(prochaineMission.wagon in attelages[m] and m != motriceProchaineMission for m in idMotrices)      # Vérifie que le wagon est disponible pour attelage, ou s'il va l'être. Sinon, placement de la motrice en attente. 
                if wagonBloque:
                    motricesEnAttente.add(motriceProchaineMission)                          # Blocage de la motrice jusqu'à évolution de la situation du réseau.
                    missionConfirmee = False
                else:
                    attelages[motriceProchaineMission].add(prochaineMission.wagon)
            elif prochaineMission.typeMission == TypeMission.Deposer:                       # Suppression de l'attelage
                try:
                    attelages[motriceProchaineMission].remove(prochaineMission.wagon)
                except KeyError:
                    pass        # Evite les erreurs lorsque le wagon n'est pas dans la liste des attelages

            # Passage à la prochaine mission et déblocage des motrices en attente
            if missionConfirmee:
                indicesMissionsActuelles[motriceProchaineMission] += 1
                motricesEnAttente.clear()

            # Mise à jour des durées. Dans le cas de missions se terminant en même temps ou de motrices en attente qui viennent d'être débloquées, la durée à ajouter peut être à 0.
            dureeAAjouter = tempsRestantMin - dureeEcouleeMissionsActuelles[motriceProchaineMission]       # Durée écoulée depuis l'exécution de la dernière mission
            instantActuel += dureeAAjouter
            for m in idMotrices:
                if m == motriceProchaineMission:
                    dureeEcouleeMissionsActuelles[m] = 0                    # Réinitialisation des durées écoulées pour la motrice venant de terminer sa mission
                else:
                    dureeEcouleeMissionsActuelles[m] += dureeAAjouter       # Incrémentation des durées écoulées pour les autres motrices

            # Evaluation des requêtes. Cherche si la mission permet de valider une requête du wagon déplacé
            if prochaineMission.typeMission == TypeMission.Deposer:
                wagonEvalue = prochaineMission.wagon
                for requete in config.requetes[wagonEvalue]:
                    if positionsWagons[wagonEvalue] == requete.noeud:
                        pass            # TODO: réfléchir à la condition. Pour chaque requête, chercher l'instant le plus proche de la fenêtre de temps.

            # Actualisation de la boucle
            nbMissionsEvalues += 1

        # TODO: Calcul de l'objectif de distance parcourue par les motrices
        pass

    sol1 = generer_solution()
    sol2 = generer_solution()

    evaluer_solution(sol1)
    evaluer_solution(sol2)

    for i in range(50):
        sol1 = muter_solution(sol1)
        sol2 = muter_solution(sol2)
        sol1, sol2 = croiser_solutions(sol1, sol2)
    pass

                
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
    resoudre_probleme(config)

    # Fin du programme
    terminer_programme(config)
    pause_and_exit()

main()