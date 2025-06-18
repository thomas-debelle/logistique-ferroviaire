from igraph import Graph

g = Graph(directed=True)
g.add_vertices(5)
g.add_edges([(0, 1), (1, 2), (2, 4)])

print(g.shortest_paths(0, 4))  # [[2]]
