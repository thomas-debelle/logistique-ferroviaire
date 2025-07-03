## TODO
- Créer le code permettant de générer des instances du problème à partir du réseau SNCF.
- Considérer le nombre de motrices utilisées dans le problème comme un coût logistique fixe.
- Considérer le fait que l'attelage de wagon est LIFO. Augmenter les coûts logistique si le LIFO n'est pas respecté.
- Ajouter des capacités max en terme de wagons et de motrices dans les gares. Augmenter les coûts logistiques si ces capacités ne sont pas respectées.
- Afficher les solutions et leur évolution.

## Moyen terme
- Tester la complexité du problème sur plusieurs dimensions: nombre de motrices, nombre de wagons, nombre de requêtes, nombre de noeuds. De manière générale, créer une démarche de tests.
- Introduire un critère de fin pour l'algorithme ?
- Évaluer la complexité numérique du MILP équivalent. Donner le nombre de contraintes générées en fonction des différents paramètres du modèle : horizon temporel, taille du réseau, nombre de motrice, taille du planning...
- Parcourir le cours "Recherche Opérationnelle 3" pour voir s'il n'y a pas des éléments intéressants à intégrer dans le rapport (relaxation du problème?).


## Autres ressources

- [Jeux de données SNCF Réseau](https://ressources.data.sncf.com/explore/?sort=modified&q=publisher:'SNCF+R%C3%A9seau,+DIRECTION+FINANCE+ACHATS'+OR+publisher:'SNCF+R%C3%A9seau)
- [Lignes par région administrative](https://ressources.data.sncf.com/explore/dataset/lignes-par-region-administrative/information/?location=7,44.99977,5.9491&basemap=jawg.transports)
- [Vitesse maximale nominale sur ligne](https://ressources.data.sncf.com/explore/dataset/vitesse-maximale-nominale-sur-ligne/table/?location=8,46.81798,2.5351&basemap=jawg.transports)

La plupart des jeux de données SNCF qui représentent le réseau ferroviaire contiennent les lignes au format Shapefile (série de coordonnées géographiques). Reste à trouver comment extraire un temps de parcours à partir de ces données physiques.