## TODO

- IMPORTANT: corriger la fonction de mutation pour s'assurer que toutes les solutions générées soient valides. Vérifier également le processus du recuit simulé (notamment les deepcopy).

- IMPORTANT: adapter la fonction d'évaluation pour ne pas sanctionner à Inf une solution, et générer des coûts intermédiaires.

- Créer le code permettant de générer des instances du problème à partir du réseau SNCF.

- Évaluer la complexité numérique du MILP. Donner le nombre de contraintes générées en fonction des différents paramètres du modèle : horizon temporel, taille du réseau, nombre de motrice, taille du planning...

- Parcourir les commentaire dans le rapport.

- Parcourir le cours "Recherche Opérationnelle 3" pour voir s'il n'y a pas des éléments intéressants à intégrer dans le rapport (relaxation linéaire ?).

## Jeux de données SNCF

- [Jeux de données SNCF Réseau](https://ressources.data.sncf.com/explore/?sort=modified&q=publisher:'SNCF+R%C3%A9seau,+DIRECTION+FINANCE+ACHATS'+OR+publisher:'SNCF+R%C3%A9seau)

- [Lignes par région administrative](https://ressources.data.sncf.com/explore/dataset/lignes-par-region-administrative/information/?location=7,44.99977,5.9491&basemap=jawg.transports)

- [Vitesse maximale nominale sur ligne](https://ressources.data.sncf.com/explore/dataset/vitesse-maximale-nominale-sur-ligne/table/?location=8,46.81798,2.5351&basemap=jawg.transports)

La plupart des jeux de données SNCF qui représentent le réseau ferroviaire contiennent les lignes au format Shapefile (série de coordonnées géographiques). Reste à trouver comment extraire un temps de parcours à partir de ces données physiques.

## Implémentation des métaheuristiques

- [How the Ant Colony Optimization algorithm works - YouTube](https://www.youtube.com/watch?v=783ZtAF4j5g)
