from igraph import Graph

g = Graph(directed=True)
g.add_vertices(6)
g.add_edges([(0, 1), (1, 2), (2, 4), (4, 5)])

print(g.shortest_paths(0, 4))  # retourne une distance
