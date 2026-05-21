# ======================
# Лаба №1, транспортная задача
# ТЗ в ./lab1_docs./lab1_problem.docx
# ======================
import random
import matplotlib.pyplot as plt
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, LpStatus, PULP_CBC_CMD #пулп цбц цмд чтоб мутило дебаг решения

# =========================================================================
# визуализация
def draw_bipartite_graph(title, edge_labels, node_labels_supply, node_labels_demand, solution_graph=False):
    G = nx.DiGraph()

    for i in range(num_I):
        G.add_node(f"S{i}", bipartite=0)
    for j in range(num_J):
        G.add_node(f"C{j}", bipartite=1)

    edges = []
    labels = {}
    for i in range(num_I):
        for j in range(num_J):
            if solution_graph and flow[i][j] <= 0:
                continue
            edges.append((f"S{i}", f"C{j}"))
            labels[(f"S{i}", f"C{j}")] = edge_labels[i, j]

    G.add_edges_from(edges)

    # правильно раскидывает вершины по линейке, но беспорядочно 
    # pos = nx.bipartite_layout(G, nodes=[f"S{i}" for i in range(num_I)])

    # костыль для расположения по порядку
    pos = {}
    for i in range(num_I):
        pos[f"S{i}"] = (0, -i)
    for j in range(num_J):
        pos[f"C{j}"] = (1, -j)

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=[f"S{i}" for i in range(num_I)], node_color='lightblue', node_size=800, label='Поставщики')
    nx.draw_networkx_nodes(G, pos, nodelist=[f"C{j}" for j in range(num_J)], node_color='lightgreen', node_size=800, label='Потребители')

    # Метки вершин с объёмами
    node_labels = {}
    for i in range(num_I):
        node_labels[f"S{i}"] = f"S{i}\n{supply[i]}"
    for j in range(num_J):
        node_labels[f"C{j}"] = f"C{j}\n{demand[j]}"
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=15, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red' if not solution_graph else 'blue', font_size=8)

    plt.title(title)
    plt.axis('off')
    #plt.legend()
    plt.show()

random.seed(1337)

num_I = 23   
num_J = 15   

supply = [random.randint(5, 50) for _ in range(num_I)]
demand = [random.randint(5, 50) for _ in range(num_J)]

if sum(supply) > sum(demand):
    supply[-1] = max(1, supply[-1] - (sum(supply) - sum(demand)))

print("Поставщики (a_i):", supply)
print("Потребители (b_j):", demand)
print(f"Сумма предложения = {sum(supply)}, сумма спроса = {sum(demand)}")

# случайная стоимость перевозки из а в б
costs = [[random.randint(1, 20) for _ in range(num_J)] for _ in range(num_I)]
print("Матрица стоимостей c_ij:")
for row in costs:
    print(row)

# линпрога из модуля pulp
prob = LpProblem("lab1_problem_transpot_graphs", LpMinimize)

# массив всех переменных для линпроги
x = {}
for i in range(num_I):
    for j in range(num_J):
        x[i, j] = LpVariable(f"x_{i}_{j}", lowBound=0, cat=LpInteger)

prob += lpSum(costs[i][j] * x[i, j] for i in range(num_I) for j in range(num_J))

for i in range(num_I):
    prob += lpSum(x[i, j] for j in range(num_J)) == supply[i]
for j in range(num_J):
    prob += lpSum(x[i, j] for i in range(num_I)) <= demand[j]



prob.solve(PULP_CBC_CMD(msg=False)) # мучу жуть дебага 

#print("\nСтатус решения:", LpStatus[prob.status])
print("Минимальные затраты =", prob.objective.value())

# парс
flow = [[0] * num_J for _ in range(num_I)]
for i in range(num_I):
    for j in range(num_J):
        flow[i][j] = int(x[i, j].value())
print("\nМатрица перевозок x_ij:")
for row in flow:
    print(row)

for i in range(num_I):
    shipped = sum(flow[i][j] for j in range(num_J))
    print(f"Поставщик {i + 1}: вывезено {shipped} из {supply[i]}")
for j in range(num_J):
    received = sum(flow[i][j] for i in range(num_I))
    print(f"Потребитель {j + 1}: получено {received} при спросе {demand[j]}")

# граф всех соединений
edge_labels_cost = {(i, j): f"{costs[i][j]}" for i in range(num_I) for j in range(num_J)}
draw_bipartite_graph("Начальный граф транспортной задачи (стоимости перевозки)", edge_labels_cost, supply, demand, solution_graph=False)

# граф оптимальной перевозки
edge_labels_flow = {(i, j): f"{flow[i][j]}" for i in range(num_I) for j in range(num_J) if flow[i][j] > 0}
draw_bipartite_graph("Граф решения (объёмы перевозок)", edge_labels_flow, supply, demand, solution_graph=True)