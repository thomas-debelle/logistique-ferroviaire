    def evaluer_solution(sol: Solution):
        # TODO: vérifier comment le système se comporte avec deux actions Récupérer ou deux actions Déposer consécutives (probablement aucun impact mais à vérifier)
        # TODO: éviter de recalculer la distance à chaque itération ? (en stockant les valeurs et en recalculant uniquement les wagons impactés par d'autres missions)
        # TODO: organiser et optimiser le code.

        # Variables de simulation
        positionsMotrices = copy.deepcopy(config.motrices)                      # Position des motrices à la fin de leur dernière mission. Par défaut, positions initiales dans le problème.
        positionsWagons = copy.deepcopy(config.wagons)                          # Noeud actuel du wagon. Si le wagon se déplace, dernier noeud de passage du wagon.
        positionsCiblesWagons = copy.deepcopy(config.wagons)                    # Noeuds en direction desquels se déplacent les wagons (prochaine position où le wagon est disponible pour l'attelage). Si le wagon ne se déplace pas, noeud actuel du wagon.
        attelages = {m: set() for m in idMotrices}                              # Pour chaque motrice, collection des wagons attelés
        motricesBloquees = set()                                                # Collections des motrices bloquées. Ces motrices sont en attente à leur position actuelle jusqu'à la confirmation d'une autre mission. Ce comportement permet de gérer les motrices arrivant en avance à une position pour récupérer un wagon.
        wagonsBloques = set()                                                   # 

        # Variable de gestion des requêtes et des missions
        nbMissionsTotal = sum([len(l) for l in sol.values()])                   # Nombre total de missions à simuler
        indicesMissionActuelle = {m: 0 for m in idMotrices}                     # Collection contenant les indices des missions évaluées pour chaque motrice.
        indicesRequeteActuelle = {m: 0 for m in idMotrices}                     # Collection contenant les indices des requêtes évaluées pour chaque motrice.

        # Variables de temps
        instantActuel = 0
        dureeEcouleeMissionsActuelles = {m: 0 for m in idMotrices}              # Durée écoulée depuis que la mission actuelle est active (utilisée pour le calcul des instants)
        
        # Processus principal. Dans la collection des missions actuelles, on recherche et traite celle qui sera terminée le plus prochainement.
        nbMissionsEvalues = 0
        while nbMissionsEvalues < nbMissionsTotal:
            # ---------------------------------------------------------
            # Recherche de la prochaine mission
            # ---------------------------------------------------------
            tempsRestantMin = sys.maxsize
            motriceProchaineMission = -1
            arriveeProchaineMission = -1
            prochaineMission = None
            for m in idMotrices:
                if (indicesMissionActuelle[m] >= len(sol[m]) 
                    or m in motricesBloquees):                                         # Si la motrice m n'a plus de missions ou est en attente, pas de traitement
                    continue
                
                # Extraction des informations pour le calcul d'itinéraire
                i = indicesMissionActuelle[m]
                mis = sol[m][i]
                depart = positionsMotrices[m]
                if mis.typeMission == TypeMission.Recuperer:
                    arrivee = positionsCiblesWagons[mis.wagon]                      # Extraction de la position du wagon cible
                elif mis.typeMission == TypeMission.Deposer:
                    arrivee = mis.noeud
                
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
                    prochaineMission = mis

            # ---------------------------------------------------------
            # Exécution de la prochaine mission
            # ---------------------------------------------------------
            missionConfirmee = True         # Flag de confirmation de la mission. Si une opération liée à la mission ne peut pas être réalisée (par exemple, un attelage), la mission n'est pas confirmée et la motrice reste bloquée jusqu'à ce que la mission puisse être confirmée.
            
            # Mise à jour des positions en fonction des déplacements réalisés pour la mission.
            positionsMotrices[motriceProchaineMission] = arriveeProchaineMission            # Mise à jour de la position de la motrice
            for w in attelages[motriceProchaineMission]:
                positionsWagons[w] = arriveeProchaineMission

            # Mise à jour des attelages
            if prochaineMission.typeMission == TypeMission.Recuperer:                       # Attelage du wagon s'il n'est pas bloqué                   
                wagonBloque = any(prochaineMission.wagon in attelages[m] and m != motriceProchaineMission for m in idMotrices)      # Vérifie que le wagon est disponible pour attelage, ou s'il va l'être. Sinon, placement de la motrice en attente. 
                if wagonBloque:
                    motricesBloquees.add(motriceProchaineMission)                          # Blocage de la motrice jusqu'à évolution de la situation du réseau.
                    missionConfirmee = False
                else:
                    attelages[motriceProchaineMission].add(prochaineMission.wagon)
            elif prochaineMission.typeMission == TypeMission.Deposer:                       # Suppression de l'attelage
                try:
                    attelages[motriceProchaineMission].remove(prochaineMission.wagon)
                except KeyError:
                    pass        # Evite les erreurs lorsque le wagon n'est pas dans la liste des attelages

            # Confirmation de la mission : passage à la prochaine mission et déblocage des motrices en attente
            if missionConfirmee:
                indicesMissionActuelle[motriceProchaineMission] += 1
                motricesBloquees.clear()

            # ---------------------------------------------------------
            # Evaluation de la prochaine requête
            # ---------------------------------------------------------
            if prochaineMission.typeMission == TypeMission.Deposer:         # Cherche si la mission permet de valider la prochaine requête du wagon
                wagonEvalue = prochaineMission.wagon
                prochaineRequete = config.requetes[wagonEvalue][indicesRequeteActuelle[wagonEvalue]]

                # Si le noeud où le wagon a été déposé correspond au noeud de destination de sa requête 
                if prochaineMission.noeud == prochaineRequete.noeud:
                    pass

                if positionsWagons[wagonEvalue]
                for prochaineRequete in config.requetes[wagonEvalue]:
                    if positionsWagons[wagonEvalue] == prochaineRequete.noeud:
                        pass            # TODO: réfléchir à la condition. Pour chaque requête, chercher l'instant le plus proche de la fenêtre de temps.

            # ---------------------------------------------------------
            # Gestion du temps
            # ---------------------------------------------------------
            # Mise à jour des durées. Dans le cas de missions se terminant en même temps ou de motrices en attente venant d'être débloquées, la durée à ajouter peut être à 0.
            dureeAAjouter = tempsRestantMin - dureeEcouleeMissionsActuelles[motriceProchaineMission]       # Durée écoulée depuis l'exécution de la dernière mission
            instantActuel += dureeAAjouter
            for m in idMotrices:
                if m == motriceProchaineMission:
                    dureeEcouleeMissionsActuelles[m] = 0                    # Réinitialisation des durées écoulées pour la motrice venant de terminer sa mission
                else:
                    dureeEcouleeMissionsActuelles[m] += dureeAAjouter       # Incrémentation des durées écoulées pour les autres motrices


            # TODO: actualisation des wagons bloqués
            pass

            # Actualisation de la boucle
            nbMissionsEvalues += 1

        

        # TODO: Calcul de l'objectif de distance parcourue par les motrices
        pass