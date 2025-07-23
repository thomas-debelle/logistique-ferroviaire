## TODO - développement
- Réécrire le code du modèle avec un mode de résolution permettant de respecter toutes les contraintes.
- Distinguer dans la résolution les différents noeuds pour en définir la capacité
- Intégrer des contraintes de blocage temporaire ou permanente du réseau pour certaines motrices, sur certains arcs, à des instants donnés. Adapter le processus de rechercher du plus court chemin pour éviter ces arcs. Dans l'ensemble, faire en sorte que le réseau soit dynamique.
- Afficher les solutions et leur évolution. Faire en sorte d'afficher le graphe en respectant ls positions géographiques pour les noeuds. (from folium.plugins import TimestampedGeoJson)
- Ajouter des capacités max en terme de wagons et de motrices dans les gares. Augmenter les coûts logistiques si ces capacités ne sont pas respectées.
- Construire les premières expériences. Chercher des informations économiques pour construire une carnet de commandes (si possible).

### Bonus
- Considérer le nombre de motrices utilisées dans le problème comme un coût logistique fixe.
- Considérer le fait que l'attelage de wagon est LIFO. Augmenter les coûts logistique si le LIFO n'est pas respecté.




## Démarche expérimentale
- Tester la complexité du problème sur plusieurs dimensions: nombre de motrices, nombre de wagons, nombre de requêtes, nombre de noeuds. De manière générale, créer une démarche de tests.
- Introduire un critère de fin pour l'algorithme ? Essentiellement intéressant pour le mémoire.
- Évaluer la complexité numérique du MILP équivalent. Donner le nombre de contraintes générées en fonction des différents paramètres du modèle : horizon temporel, taille du réseau, nombre de motrice, taille du planning...
- Parcourir le cours "Recherche Opérationnelle 3" pour voir s'il n'y a pas des éléments intéressants à intégrer dans le rapport (relaxation du problème?).

## Notes générales et réflexions
Les entreprises ferroviaires (EF) sont chacune responsable de leurs motrices et de leur carnet de commandes. SNCF Réseau ne s'occupe que de la répartition des sillons/créneaux horaires en fonction des demandes des EF.
-> Le programme est à destination des EF, car sa fonction principale est de répartir les motrices en fonction des commandes. A partir des résultats du programme, l'EF peut faire des demandes de sillon à 

Les motrices peuvent être soumises à de nombreuses contraintes opérationnelles :
- Type de traction sur le réseau (électrique, diesel, biocarburant).
- Compatibilité du réseau (gabarit, tension) -> vérifier les types de traction des motrices. S'il y a plusieurs types de tractions (et plusieurs types de lignes), l'information doit être inclue dans les arcs du graphe.
- Temps de conduite du conducteur -> besoin de s'arrêter à un lieu où il peut y avoir un changement de conducteur.
- Fenêtres de maintenance -> on peut admettre que la simulation est réalisée entre deux maintenances.

Logiquement, il faudrait également prendre en compte le coût d'emprunt de certains sillons dans le coût logisitique de la solution.
Le mode de cantonnement n'est en principe pas discriminant pour les trains de fret.

## Ressources
- Tarification des sillons et coût marginal: https://afra.fr/actualites-ferroviaire/la-tarification-de-linfrastructure-fait-obstacle-a-la-competitivite-intermodale-du-rail
- Répartition des sillons et enchères combinatoires: https://variances.eu/?p=5388