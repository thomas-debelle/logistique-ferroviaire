import matplotlib.pyplot as plt

# Données
lots = [1, 5, 10, 30]
temps_m5 = [0.43, 3.57, 7.98, 25.93]   # Mots = 5
temps_m10 = [0.57, 2.63, 8.43, 27.80]  # Mots = 10

# Création du graphique
plt.figure(figsize=(8,5))
plt.plot(lots, temps_m5, marker='o', label='Mots = 5')
plt.plot(lots, temps_m10, marker='s', label='Mots = 10')

plt.xlabel('Nombre de lots')
plt.ylabel('Temps (minutes)')
plt.title("Temps de résolution en fonction du nombre de lots")
#plt.xscale('log')  # échelle logarithmique pour mieux visualiser la croissance
#plt.yscale('log')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()
