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

        self.graphe = None
        self.noeudGare = []
        self.requetes = []
        self.motrices = {}
        self.wagons = {}

class TypeMission(Enum):
    Recuperer = 1
    Deposer = 2

class Mission:
    def __init__(self, typeMission: TypeMission, param1: int, param2: int) -> None:
        self.typeMission = typeMission
        self.param1 = param1
        self.param2 = param2

    def __str__(self) -> str:
        return f"({self.typeMission}, {self.param1}, {self.param2})"

    def __repr__(self) -> str:
        return str(self)

""""
Dictionnaire associant une séquence de missions à chaque identifiant de motrice. 
Chaque mission est un tuple (TypeMission, int, int), où les entiers sont des paramètres pour la mission à effectuer (pas tous toujours nécessaires).
"""
Solution = Dict[int, List[Mission]]

class Requete:
    def __init__(self, idWagon, depart, arrivee, tempsDebut, tempsFin) -> None:
        """
        idWagon: identifiant du wagon à transporter.
        depart: noeud de départ.
        arrivee: noeud d'arrivée.
        tempsDebut: début de la fenêtre de temps durant laquelle la requête est valide. Le wagon doit se trouver au depart à tempsDebut.
        tempsFin: fin de la fenêtre de temps durant laquelle la requête est valide. Le wagon doit se trouver à l'arrivée à tempsFin.
        """
        self.idWagon = idWagon
        self.depart = depart
        self.arrivee = arrivee
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
        g.es['weights'] = poids
        g.vs['name'] = [str(n) for n in noeuds]

        return g
    

    config = ConfigProbleme()

    with open(cheminFichierConfig) as fichier:
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
                config.requetes = [Requete(r[0], r[1], r[2], r[3], r[4]) for r in donnees['requetes']]

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
                missionRecup = Mission(TypeMission.Recuperer, wagon, 0)                        # Ira récupérer le wagon sur n'importe quel noeud
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
            missionRecup = Mission(TypeMission.Recuperer, wagon, 0)
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
                    wagon = sol[m][pos].param1              # Si le wagon est déjà attelé, il ne peut pas être récupéré. Suppression de la mission.
                    if wagon in wagonsAtteles:
                        posASupprimer.append(pos)
                    else:
                        wagonsAtteles.append(wagon)
                elif sol[m][pos].typeMission == TypeMission.Deposer:
                    wagon = sol[m][pos].param1
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
        attelages = dict()          # Pour chaque wagon, séquence des motrices d'affectation indexées par le temps

        # Calcul de l'objectif de distance parcourue par les motrices

        # Calcul de l'objectf de respect des requêtes
        # On souhaite que les wagons passent au plus proche de leurs points de requêtes lors des instants requêtés
        # A réfléchir

        pass        # TODO

    sol1 = generer_solution()
    sol2 = generer_solution()
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