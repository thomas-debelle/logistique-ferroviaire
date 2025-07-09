## TODO

- Générer les graphes avec une topologie simplifiée. Les raccordements ne doivent avoir lieu qu'aux extrémités des lignes, où on cherche alors la ligne la plus proche (sinon, mauvaise gestion des ponts). A chaque début et fin de ligne: chercher le noeud le plus proche (géodésiquement) et ajuster l'extrémité pour réaliser le raccord. Attention, le raccord ne doit être ajouté que pour des voies très proches.
- Ajouter des paramètres sur la vitesse adoptée spécifiquement par les trains de frets sur le réseau/les portions autorisées (par exemple, interdiction de rouler sur les LGV)


## Constructeur de problèmes

Le constructeur de problème permet de générer un problème à partir de jeux de données SNCF.

## Jeux de données

- [Jeux de données SNCF Réseau](https://ressources.data.sncf.com/explore/?sort=modified&q=publisher:'SNCF+R%C3%A9seau,+DIRECTION+FINANCE+ACHATS'+OR+publisher:'SNCF+R%C3%A9seau)
- [Vitesse maximale nominale sur ligne](https://ressources.data.sncf.com/explore/dataset/vitesse-maximale-nominale-sur-ligne/table/?location=8,46.81798,2.5351&basemap=jawg.transports)
- [Fichier de forme SNCF](https://ressources.data.sncf.com/explore/dataset/formes-des-lignes-du-rfn/information/?basemap=63a416&location=16,47.7477,7.29715)
- [Installation Terminales Embranchées](https://ressources.data.sncf.com/explore/dataset/liste-des-installations-terminales-embranchees/information/) --> Important pour identifier les gares de fret.
- [Données concaténées](https://www.data.gouv.fr/datasets/donnees-reseau-ferroviaire-national-concatenees/)
- [Lexique des acronymes SNCF](https://ressources.data.sncf.com/explore/dataset/lexique-des-acronymes-sncf/)