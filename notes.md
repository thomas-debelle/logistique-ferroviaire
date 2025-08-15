## TODO
- Afficher les solutions et leur évolution. Faire en sorte d'afficher le graphe en respectant ls positions géographiques pour les noeuds. (from folium.plugins import TimestampedGeoJson)
- Générer la liste des sillons empruntés à partir de la solution -> liste d'arêtes avec les temps de parcours. Réutiliser ces liste en sillons bloqués pour générer les parcours d'une seconde compagnie ferroviaire.
- Voir si possible d'ajouter le coût estimé des sillons dans le calcul du coût (simple à intégrer aux objectifs mais difficile à estimer).

## Démarche expérimentale
- Tester la complexité du problème sur plusieurs dimensions: nombre de motrices, nombre de lots, taille du réseau. De manière générale, créer une démarche de tests.
Sillons bloqués