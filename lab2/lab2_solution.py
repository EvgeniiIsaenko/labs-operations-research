# ======================
# Лаба №2, транспортная задача (поток)
# ТЗ в ./lab2_docs./lab2_problem.docx
# ======================
import networkx as nx
import matplotlib.pyplot as plt
import random

random.seed(1337)        
N_vertices            = 10 
n_sources             = 2           
final_destinatons     = 2           
n_layers              = 4           

# k-дольный граф
layers = [[] for _ in range(n_layers)]
vertices_per_layer = [2, 3, 3, 2]   # sum(vertices_per_layer) = N_vertices;
for i, size in enumerate(vertices_per_layer):
    layers[i] = list(range(sum(vertices_per_layer[:i]), sum(vertices_per_layer[:i+1])))

sources = layers[0][:n_sources]     # источниками является здесь начальные пункты
final_destinatons   = layers[-1][-final_destinatons:]     # потребителями являются конечные пункты

G = nx.DiGraph()
G.add_nodes_from(range(N_vertices))

for layer_idx in range(n_layers - 1):
    for u in layers[layer_idx]:
        for v in layers[layer_idx+1]:
            # идет добавление ребра ТОЛЬКО в соседнюю долю
            if random.random() < 0.6:
                capacity = random.randint(1, 20)
                G.add_edge(u, v, capacity=capacity)

print("Начальные вершины:", sources)
print("Конечные вершины:", final_destinatons)

# для нахождения максимального потока
super_source = N_vertices
super_final_destinaton = N_vertices + 1
G_super = G.copy()
G_super.add_node(super_source)
G_super.add_node(super_final_destinaton)

for s in sources:
    G_super.add_edge(super_source, s, capacity=float('inf'))
for t in final_destinatons:
    G_super.add_edge(t, super_final_destinaton, capacity=float('inf'))

flow_value, flow_dict = nx.maximum_flow(G_super, super_source, super_final_destinaton)

print(f"Максимальный суммарный поток из источников в конечные: {flow_value}")

# словарик потока без супер вершин
flow_original = {}
for u in G.nodes():
    flow_original[u] = {}
    for v in G.nodes():
        if v in flow_dict.get(u, {}):
            flow_original[u][v] = flow_dict[u][v]


# --------------------------------
# рисунок 1 визуализация 

plt.figure(figsize=(12, 6))
# networkX кукбук spring_layout(): Position nodes using Fruchterman-Reingold force-directed algorithm.
pos = nx.spring_layout(G, seed=1337)

node_colors = []
for node in G.nodes():
    if node in sources:
        node_colors.append('lightgreen')
    elif node in final_destinatons:
        node_colors.append('lightcoral')
    else:
        node_colors.append('lightblue')

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

edge_labels = {(u, v): f"c={d['capacity']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=15, edge_color='gray')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Первый граф со всеми пропускными способностями")
plt.axis('off')
plt.tight_layout()
# plt.savefig("initial_graph.png", dpi=150)
plt.show()

# --------------------------------
# рисунок 2 визуализация 

H = nx.DiGraph()
H.add_nodes_from(G.nodes())
positive_edges = []
flow_labels = {}
for u in flow_original:
    for v, f in flow_original[u].items():
        # ТОЛЬКО ненулевые потоки
        if f > 1e-6:
            positive_edges.append((u, v))
            flow_labels[(u, v)] = f"x={f:.1f}"
H.add_edges_from(positive_edges)

plt.figure(figsize=(12, 6))
nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=800)
nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold')
nx.draw_networkx_edges(H, pos, edgelist=positive_edges, arrowstyle='-|>', arrowsize=15, edge_color='blue', width=2)
nx.draw_networkx_edge_labels(H, pos, edge_labels=flow_labels, font_size=8)

plt.title("Подграф графа рисунка 1 (граф решения)")
plt.axis('off')
plt.tight_layout()
# plt.savefig("solution_graph.png", dpi=150)
plt.show()

# и в конце все ребра которые двигали товары
print("\nНенулевые потоки по рёбрам:")
for (u,v), f in flow_labels.items():
    print(f"  {u} -> {v} : {f}")