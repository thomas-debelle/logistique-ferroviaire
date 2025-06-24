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

        # Constantes
        self.horizonTemporel = 500
        self.baseScoreRequetes = 10
        self.coeffPenaliteRetard = 1.0

class TypeMission(Enum):
    Recuperer = 1
    Deposer = 2

class Mission:
    def __init__(self, typeMission: TypeMission, wagon: int, noeud: int) -> None:
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
Chaque mission est un tuple (TypeMission, int, int), où les entiers sont des paramètres pour la mission à effectuer (pas tous toujours nécessaires).
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
        # Variable de simulation
        dernierNoeudMotrices = copy.deepcopy(config.motrices)
        dernierNoeudWagons = copy.deepcopy(config.wagons)
        prochainNoeudMotrices = copy.deepcopy(config.motrices)
        prochainNoeudWagons = copy.deepcopy(config.wagons)
        attelages = {w: -1 for w in idWagons}                                       # Associe une motrice (valeur) à chaque wagon (clé). Si l'attelage est <0, alors le wagon n'est attelé à aucune motrice.
        debutsTransbordements = {w: -1 for w in idWagons}                           # Instant de début du dernier transbordement de chaque wagon. Si la valeur est à -1, cela signifie qu'aucun transbordement n'est en cours.

        # Variable de gestion des requêtes et des missions
        indicesMissionsActuelles = {m: 0 for m in idMotrices}                       # Indices des missions actuellement évaluées
        indicesRequetesActuelles = {w: 0 for w in idWagons}                       # Indices des requêtes actuellement évaluées
        debutsMissionsActuelles = {m: 0 for m in idMotrices}                        # Instant de début des missions actuellement simulées
        
        # Boucle principale: simulation de la solution instant par instant
        t = 0
        scoreRequetes = 0.0                                                         # Le score est incrémenté pour chaque requête validé. On cherche à le maximiser.
        nbMissionsTotal = sum([len(l) for l in sol.values()])                       # Tant que toutes les missions n'ont pas été évaluées
        nbMissionsEvalues = 0
        while nbMissionsEvalues < nbMissionsTotal:
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
                if tempsRestant > 0:    # TODO: interpréter et gérer les temps négatifs (pas forcément de modif nécessaire) (lorsqu'une motrice poursuit un wagon qui se déplace, et se rapproche du noeud d'origine de la motrice)
                    continue            # Si la motrice n'est pas encore à destination (il reste du temps de déplacement) on ne réalise pas d'autre traitement

                # Mise à jour des dernières positions de la motrice et de ses attelages
                dernierNoeudMotrices[m] = arrivee
                for w in wagonsAtteles:            # Extraction des wagons attelés à la motrice
                    dernierNoeudWagons[w] = arrivee

                # Mise à jour des attelages
                wagonMission = missionActuelle.wagon
                if missionActuelle.typeMission == TypeMission.Recuperer and attelages[wagonMission] < 0:       # Si le wagon n'est pas déjà attelé, la mission peut être validée
                    attelages[wagonMission] = m
                    missionValidee = True
                elif missionActuelle.typeMission == TypeMission.Deposer and attelages[wagonMission] == m:
                    attelages[wagonMission] = -1                    # Suppression de l'attelage
                    missionValidee = True

                # Si la mission a pu être validée, on passe à la suivante
                if missionValidee:
                    debutsMissionsActuelles[m] = t
                    indicesMissionsActuelles[m] += 1
                    nbMissionsEvalues += 1

            # ---------------------------------------------------------
            # Mise à jour des wagons et requêtes
            # ---------------------------------------------------------
            for w in idWagons:
                if indicesRequetesActuelles[w] >= len(config.requetes[w]):
                    continue

                # Si le wagon atteint la destination et n'est plus attelé, l'opération de transbordement peut commencer. Lorsque le transbordement est terminé, la requête est validée.
                requete = config.requetes[w][indicesRequetesActuelles[w]]
                if dernierNoeudWagons[w] != requete.noeud or attelages[w] > 0 or t < requete.tempsDebut:
                    continue            # TODO: voir si besoin d'appliquer une pénalité d'avance (dans ce cas, nécessite de modéliser l'encombrement maximal à chaque noeud)
                    
                # Application éventuelle d'une pénalité de retard
                if t > requete.tempsFin:
                    pass

                # Début du transbordement
                if debutsTransbordements[w] < 0:
                    debutsTransbordements[w] = t
                
                # Fin du transbordement et validation de la requête
                if t - debutsTransbordements[w] >= requete.dureeTransbordement:
                    penaliteRetard = max(debutsTransbordements[w] - requete.tempsFin, 0) * config.coeffPenaliteRetard
                    scoreRequetes += max(config.baseScoreRequetes - penaliteRetard, 0)      # Fin de la requête et calcul du score, avec enregistrement éventuel d'une pénalité
                    debutsTransbordements[w] = -1
                    indicesRequetesActuelles[w] += 1
                    
            t += 1
            pass        # TODO: terminer l'évaluation en intégrant un calcul des distances parcourues par les motrices. A faire en parallèle du calcul des requêtes ?


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