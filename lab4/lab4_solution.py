# ======================
# Лаба №4, затраты сырья на производство из имеющегося запасасырья в день
# ТЗ в ./lab4_docs./lab4_problem.docx
# ======================

import pulp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

np.random.seed(1337)

L = 2 # типы сырья
K = 2 # типы товаров
M = 5 # дни горизонта планирования
N = 4 # число вершин
A = np.array([[random.randint(1, 5) for _ in range(K)]         # --
      for _ in range(L)])                                      # сырья на единицу товара

p = np.random.uniform(10, 50, size=(K, M))
gamma = np.random.uniform(0, 20, size=(L, M))
b0 = np.random.uniform(10, 30, size=L)
Q = np.array([100, 80])

# дуги
arcs = [(1,2), (1,3), (2,3), (2,4), (3,4)]
n_arcs = len(arcs)
d = {}
c = {}
for (i,j) in arcs:
    d[(i,j)] = np.random.uniform(30, 100)
    c[(i,j)] = np.random.uniform(10, 50)

model = pulp.LpProblem("Production_Transport", pulp.LpMaximize)

days = range(1, M+1)
goods = range(1, K+1)
resources = range(1, L+1)
nodes = range(1, N+1)

x = pulp.LpVariable.dicts("x", ((k,m) for k in goods for m in days),
                          lowBound=0, cat=pulp.LpInteger)
b = pulp.LpVariable.dicts("b", ((l,m) for l in resources for m in days),
                          lowBound=0, cat=pulp.LpContinuous)
z = pulp.LpVariable.dicts("z", ((i,j) for (i,j) in arcs),
                          lowBound=0, cat=pulp.LpInteger)
lam = pulp.LpVariable.dicts("lam", ((i,j) for (i,j) in arcs),
                            cat=pulp.LpBinary)

# выручка - затраты
model += (pulp.lpSum(p[k-1,m-1] * x[k,m] for k in goods for m in days)
          - pulp.lpSum(c[(i,j)] * lam[i,j] for (i,j) in arcs))

for l in resources:
    for m in days:
        model += (pulp.lpSum(A[l-1,k-1] * x[k,m] for k in goods) <= b[l,m])

model += (pulp.lpSum(x[k,m] for k in goods for m in days)
          == pulp.lpSum(z[1,j] for (i,j) in arcs if i==1))

for i in range(2, N):
    outflow = pulp.lpSum(z[i,j] for (i_,j) in arcs if i_==i)
    inflow = pulp.lpSum(z[j,i] for (j,i_) in arcs if i_==i)
    model += (outflow == inflow)

for k in goods:
    model += (pulp.lpSum(x[k,m] for m in days) <= Q[k-1])

for (i,j) in arcs:
    model += (z[i,j] <= d[(i,j)])          # (6)
    model += (z[i,j] <= 1e6 * lam[i,j]) # (7)

model += (pulp.lpSum(z[i,N] for (i,j) in arcs if j==N)
          == pulp.lpSum(x[k,m] for k in goods for m in days))

for l in resources:
    # для дней
    model += (b[l,1] == b0[l-1])
    for m in range(1, M):
        model += (b[l, m+1] == b[l, m]
                  - pulp.lpSum(A[l-1,k-1] * x[k,m] for k in goods)
                  + gamma[l-1,m-1])

model.solve(pulp.PULP_CBC_CMD(msg=False)) # почему сообщение дебага идет автоматом # мут дебага

status = pulp.LpStatus[model.status]
print(f"Статус: {status}")
if status != 'Optimal':
    print("Оптимальное решение не найдено")
    exit()

print(f"Максимальная прибыль: {pulp.value(model.objective):.2f}")

x_vals = {(k,m): pulp.value(x[k,m]) for k in goods for m in days}
z_vals = {(i,j): pulp.value(z[i,j]) for (i,j) in arcs}
lam_vals = {(i,j): pulp.value(lam[i,j]) for (i,j) in arcs}
b_vals = {(l,m): pulp.value(b[l,m]) for l in resources for m in days}

print("\nПроизводство x_km:")
for k in goods:
    for m in days:
        if x_vals[k,m] > 0:
            print(f"  Товар {k}, день {m}: {x_vals[k,m]:.2f}")

print("\nПеревозки z_ij:")
for (i,j) in arcs:
    if z_vals[i,j] > 0:
        print(f"  {i}->{j}: {z_vals[i,j]:.2f} (lambda={lam_vals[i,j]})")

# ============================
# графы и визуализация

fig = plt.figure(figsize=(18, 14))

graph1 = fig.add_subplot(3, 2, 1)
G = nx.DiGraph()
G.add_nodes_from(nodes)
for (i,j) in arcs:
    G.add_edge(i, j, capacity=d[(i,j)], cost=c[(i,j)])

pos = nx.spring_layout(G, seed=1337)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, font_weight='bold')

edge_labels = {(i,j): f"d={d[(i,j)]:.1f}\nc={c[(i,j)]:.1f}" for (i,j) in arcs}
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
graph1.set_title("Начальный граф")
graph1.axis('off')
# plt.tight_layout()
#plt.savefig("initial_graph.png")
# plt.show()


graph2 = fig.add_subplot(3, 2, 2)
nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=500)
nx.draw_networkx_labels(G, pos, font_weight='bold')

# дуги по которым идет двиение
active_edges = [(i,j) for (i,j) in arcs if z_vals[i,j] > 0]
inactive_edges = [(i,j) for (i,j) in arcs if z_vals[i,j] == 0]
nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, arrows=True,
                       arrowstyle='->', arrowsize=20, edge_color='gray', style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=active_edges, arrows=True,
                       arrowstyle='->', arrowsize=20, edge_color='red', width=2)
edge_labels_sol = {(i,j): f"{z_vals[i,j]:.1f}" for (i,j) in arcs if z_vals[i,j] > 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_sol, font_size=9, label_pos=0.5)
graph2.set_title("Граф решения")
graph2.axis('off')
# plt.tight_layout()
# plt.savefig("solution_graph.png")
# plt.show()

graph3 = fig.add_subplot(3, 2, 3)
for l in resources:
    vals = [b_vals[l,m] for m in days]
    graph3.plot(days, vals, marker='o', label=f'Сырьё {l}')
graph3.set_xlabel('День m')
graph3.set_ylabel('Запас сырья b_lm')
graph3.set_title('Запасы сырья по дням')
graph3.legend()
graph3.grid(True)
# plt.tight_layout()
# plt.savefig("inventory.png")
# plt.show()

graph4 = fig.add_subplot(3, 2, 4)
total_b = [sum(b_vals[l,m] for l in resources) for m in days]
graph4.bar(days, total_b, color='skyblue')
graph4.set_xlabel('День m')
graph4.set_ylabel('Суммарный запас сырья, сумма b_lm')
graph4.set_title('Суммарные запасы сырья')
graph4.grid(True, axis='y')
# plt.tight_layout()
# plt.savefig("total_inventory.png")
# plt.show()

graph5 = fig.add_subplot(3, 2, 5)
for k in goods:
    vals = [x_vals[k,m] for m in days]
    graph5.plot(days, vals, marker='s', label=f'Товар {k}')
graph5.set_xlabel('День m')
graph5.set_ylabel('Объём производства x_km')
graph5.set_title('Производство по дням')
graph5.legend()
graph5.grid(True)
# graph5.tight_layout()
# plt.savefig("production_daily.png")
# plt.show()

graph6 = fig.add_subplot(3, 2, 6)
total_x = [sum(x_vals[k,m] for m in days) for k in goods]
graph6.bar(range(1, K+1), total_x, tick_label=[f'Товар {k}' for k in goods], color='orange')
graph6.set_xlabel('Тип товара k')
graph6.set_ylabel('Суммарный объём, сумма x_km')
graph6.set_title('Общее производство по типам')
graph6.grid(True, axis='y')
# plt.tight_layout()
# plt.savefig("total_production.png")

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# plt.savefig("all_graphs.png", dpi=150)
plt.show()