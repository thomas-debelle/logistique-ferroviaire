## TODO
- Introduire un temps d'affectation pour temporiser. Expliciter les temps d'attente (notamment, est-ce qu'un attelage et une dépose successifs prennent 2 instants ou un instant ? Combien de temps prend l'attelage après l'arrivée dans un noeud ? Est-ce gratuit en terme de temps ?)
- Voir s'il est toujours utile de vérifier la causalité dans les solutions, sachant que les manips supplémentaires rentrent désormais dans les coûts logistiques. Proposer alors une fonction de nettoyage pour éviter les missions inutiles (s'il en reste à la fin de l'optimisation)
- Ajouter des vérifications sur le fait que les cibles des missions de dépose sont des noeuds de gare. Vérifier aussi que les wagons sont initialement sur des noeuds de gare.
- Vérifier la config (que les requêtes correspondent au problème, par exemple)
- Séparer le problème de la configuration.


## Moyen terme
- Considérer le fait que l'attelage de wagon est LIFO. Augmenter les coûts logistique si le LIFO n'est pas respecté.
- Ajouter des capacités max en terme de wagons et de motrices dans les gares.
- Introduire un critère de fin pour l'algorithme ?
- Créer le code permettant de générer des instances du problème à partir du réseau SNCF.
- Évaluer la complexité numérique du MILP équivalent. Donner le nombre de contraintes générées en fonction des différents paramètres du modèle : horizon temporel, taille du réseau, nombre de motrice, taille du planning...
- Parcourir les commentaire dans le rapport.
- Parcourir le cours "Recherche Opérationnelle 3" pour voir s'il n'y a pas des éléments intéressants à intégrer dans le rapport (relaxation du problème?).


## Nouveau système de solution:
- Système avec seulement deux types de missions: R(w) et D(w, n)
- Tant qu'une motrice ne peux pas accomplir sa mission, elle reste en attente à destination (par exemple, si le wagon à récupérer n'est pas encore arrivé).
- Pour les actions R(w), la motrice cherche le prochain noeud sur lequel elle pourra récupérer le wagon w. Pour cela, il est nécessaire d'évaluer tous les déplacements de ce wagon, et les distances pour atteindre ses noeuds successifs de changement d'attelage.
- Construire une fonction d'évaluation qui encourage le rapprochement du wagon de sa destination dans les temps.
- Intégrer également dans l'objectif l'idée de minimiser le trajet parcouru par l'ensemble des motrices sur le réseau.
- Pour les graphes: comment traiter les coûts et temps de parcours ?
- La plupart de la complexité est reportée sur l'évaluation: la correspondance des mouvements aux requêtes y est évaluée.


## Autres ressources

- [Jeux de données SNCF Réseau](https://ressources.data.sncf.com/explore/?sort=modified&q=publisher:'SNCF+R%C3%A9seau,+DIRECTION+FINANCE+ACHATS'+OR+publisher:'SNCF+R%C3%A9seau)
- [Lignes par région administrative](https://ressources.data.sncf.com/explore/dataset/lignes-par-region-administrative/information/?location=7,44.99977,5.9491&basemap=jawg.transports)
- [Vitesse maximale nominale sur ligne](https://ressources.data.sncf.com/explore/dataset/vitesse-maximale-nominale-sur-ligne/table/?location=8,46.81798,2.5351&basemap=jawg.transports)

La plupart des jeux de données SNCF qui représentent le réseau ferroviaire contiennent les lignes au format Shapefile (série de coordonnées géographiques). Reste à trouver comment extraire un temps de parcours à partir de ces données physiques.