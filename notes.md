## TODO
- Ajouter des paramètres sur la vitesse adoptée spécifiquement par les trains de frets sur le réseau/les portions autorisées (par exemple, interdiction de rouler sur les LGV). --> Notamment, ne pas importer les lignes et raccordements avec des vitesses trop élevées.
- Considérer les gares et les sites de triages comme des noeuds à ne pas supprimer. Les matérialiser d'une manière spécifique dans le graphe.
- Chercher des données et des informations pour construire une problème de fret réaliste
- Adapter la distance de raccordement en fonction de la zone (pour Paris, la distance est plus faible qu'ailleurs, sinon raccordements à Saint-Lazare).

- Considérer le nombre de motrices utilisées dans le problème comme un coût logistique fixe.
- Intégrer les contraintes de blocage du réseau à des instants donnés. Ces constraintes pourraient s'interpréter comme: si le train souhaite passer par un tronçon à l'instant t, alors son trajet est allongé d'une durée d (la durée modélisant le fait qu'on laisse passer d'autres trains devant). La principale difficulté réside dans le fait que, dans la simulation, on ne sait quel chemin a emprunté une motrice qu'une fois qu'elle arrive à destination. A voir si la poursuite dynamique des wagons reste la meilleure approche.
- Considérer le fait que l'attelage de wagon est LIFO. Augmenter les coûts logistique si le LIFO n'est pas respecté.
- Ajouter des capacités max en terme de wagons et de motrices dans les gares. Augmenter les coûts logistiques si ces capacités ne sont pas respectées.
- Afficher les solutions et leur évolution. Faire en sorte d'afficher le graphe en respectant ls positions géographiques pour les noeuds.

## Moyen terme
- Tester la complexité du problème sur plusieurs dimensions: nombre de motrices, nombre de wagons, nombre de requêtes, nombre de noeuds. De manière générale, créer une démarche de tests.
- Introduire un critère de fin pour l'algorithme ? Essentiellement intéressant pour le mémoire.
- Évaluer la complexité numérique du MILP équivalent. Donner le nombre de contraintes générées en fonction des différents paramètres du modèle : horizon temporel, taille du réseau, nombre de motrice, taille du planning...
- Parcourir le cours "Recherche Opérationnelle 3" pour voir s'il n'y a pas des éléments intéressants à intégrer dans le rapport (relaxation du problème?).