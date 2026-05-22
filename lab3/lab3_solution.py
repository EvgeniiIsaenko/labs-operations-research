# ======================
# Лаба №3, затраты сырья на производство
# ТЗ в ./lab3_docs./lab3_problem.docx
# ======================
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import random

random.seed(1337)

vertices        = 4 # вершины
types_of_goods  = 2 # виды товара
types_of_raw    = 2 # количество видов сырья

p = [random.randint(50, 100) for _ in range(types_of_goods)]       # прайс товара
b = [random.randint(100, 200) for _ in range(types_of_raw)]        # запас сырья
A = [[random.randint(1, 5) for _ in range(types_of_goods)]         # --
      for _ in range(types_of_raw)]                                # сырья на единицу товара
Q = [random.randint(10, 20) for _ in range(types_of_goods)]        # спрос

edges = []
for i in range(1, vertices + 1):
    for j in range(i + 1, vertices + 1):
        edges.append((i, j))

d = {e: random.randint(50, 150) for e in edges}
c = {e: random.randint(1, 10) for e in edges}


model = pulp.LpProblem("lab3_problem_transpot_graphs", pulp.LpMaximize)

x = [pulp.LpVariable(f"x_{k}", lowBound=0, upBound=Q[k], cat=pulp.LpInteger) for k in range(types_of_goods)]
z = {e: pulp.LpVariable(f"z_{e[0]}_{e[1]}", lowBound=0, upBound=d[e], cat=pulp.LpInteger) for e in edges}
lam = {e: pulp.LpVariable(f"lam_{e[0]}_{e[1]}", cat=pulp.LpBinary) for e in edges}

# Целевая функция: максимизация выручки минус фиксированные затраты на перевозку
model += pulp.lpSum(p[k] * x[k] for k in range(types_of_goods)) - pulp.lpSum(c[e] * lam[e] for e in edges)

# расход не больше запасов
for l in range(types_of_raw):
    model += pulp.lpSum(A[l][k] * x[k] for k in range(types_of_goods)) <= b[l]

# все из одной вершины (1)
model += pulp.lpSum(x[k] for k in range(types_of_goods)) == pulp.lpSum(z[(1, j)] for (i, j) in edges if i == 1)

for i in range(2, vertices):
    inflow = pulp.lpSum(z[(j, i)] for (j, k) in edges if k == i)    # входящие в i
    outflow = pulp.lpSum(z[(i, j)] for (i2, j) in edges if i2 == i) # исходящие из i
    model += inflow == outflow

model += pulp.lpSum(z[(i, vertices)] for (i, j) in edges if j == vertices) == pulp.lpSum(x[k] for k in range(types_of_goods))

for e in edges:
    model += z[e] <= (sum(Q) + sum(b)) * lam[e]

model.solve(pulp.PULP_CBC_CMD(msg=False)) # снова мучу жуть дебага

status = pulp.LpStatus[model.status]
print(f"Статус: {status}")
if status != "Optimal":
    print("Оптимальное решение не найдено")
    exit()

x_opt = [pulp.value(x[k]) for k in range(types_of_goods)]
z_opt = {e: pulp.value(z[e]) for e in edges}
lam_opt = {e: pulp.value(lam[e]) for e in edges}
profit = pulp.value(model.objective)

print("\nОптимальные объёмы производства:")
for k in range(types_of_goods):
    print(f"  Товар {k+1}: {x_opt[k]:.0f} (спрос {Q[k]})")
print(f"\nМаксимальная прибыль: {profit:.2f}")

print("\nПеревозки по дугам (объём / факт):")
for e in edges:
    if z_opt[e] > 1e-3:
        print(f"  {e[0]} -> {e[1]}: {z_opt[e]:.0f}  (λ={int(lam_opt[e])})")


G = nx.DiGraph()
G.add_nodes_from(range(1, vertices+1))
G.add_edges_from(edges)

plt.figure(figsize=(12, 5))

# исходный граф
plt.subplot(1, 3, 1)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos, font_size=12)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=1.5)
edge_labels = {e: f"{c[e]}" for e in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("Исходный граф\n")

# граф объемы превеозок
plt.subplot(1, 3, 2)
G2 = nx.DiGraph()
G2.add_nodes_from(range(1, vertices+1))
active_edges = [e for e in edges if z_opt[e] > 1e-3]
G2.add_edges_from(active_edges)
nx.draw_networkx_nodes(G2, pos, node_color='lightgreen', node_size=500)
nx.draw_networkx_labels(G2, pos, font_size=12)
nx.draw_networkx_edges(G2, pos, edge_color='blue', arrows=True, arrowsize=20, width=2)
edge_labels2 = {e: f"{z_opt[e]:.0f}" for e in active_edges}
nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels2, font_size=9)
plt.title("Объёмы перевозок")

# объемы производства
plt.subplot(1, 3, 3)
categories = [f"Товар {k+1}" for k in range(types_of_goods)]
plt.bar(categories, x_opt, color='orange', edgecolor='black')
plt.ylim(0, max(Q)*1.1)
for i, v in enumerate(x_opt):
    plt.text(i, v + 1, f"{v:.0f}", ha='center')
plt.title("Объёмы производства")
plt.ylabel("Количество единиц")

plt.tight_layout()
plt.show()