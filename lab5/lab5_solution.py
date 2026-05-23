# ======================
# Лаба №5, построение минимального остовного дерева
# ТЗ в ./lab5_docs./lab5_problem.docx
# ======================
import pulp

def solve_degree_constrained_mst_pulp(n_nodes, edges, costs):
    if n_nodes == 1:
        return [], 0.0, [0], {0: 0}

    M = n_nodes

    prob = pulp.LpProblem("Degree_Constrained_MST", pulp.LpMinimize)

    x = {}
    for (i, j) in edges:
        x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

    lam = [pulp.LpVariable(f"lam_{i}", cat=pulp.LpBinary) for i in range(n_nodes)]

    y = [pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(n_nodes)]

    z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=M, cat=pulp.LpInteger) for i in range(n_nodes)]

    f0 = [pulp.LpVariable(f"f0_{i}", lowBound=0, upBound=M, cat=pulp.LpInteger) for i in range(n_nodes)]

    f_fwd = {} # направление вперед
    f_rev = {} # направление назад 
    for (i, j) in edges:
        f_fwd[(i, j)] = pulp.LpVariable(f"f_{i}_{j}", lowBound=0, upBound=M, cat=pulp.LpInteger)
        f_rev[(i, j)] = pulp.LpVariable(f"f_{j}_{i}", lowBound=0, upBound=M, cat=pulp.LpInteger)

    prob += pulp.lpSum(costs[(i, j)] * x[(i, j)] for (i, j) in edges)

    for i in range(n_nodes):
        incident = [x[(u, v)] for (u, v) in edges if u == i or v == i]
        prob += pulp.lpSum(incident) == 2 - y[i] + z[i]

    for i in range(n_nodes):
        prob += z[i] <= M * lam[i]

    prob += pulp.lpSum(x.values()) == n_nodes - 1

    prob += pulp.lpSum(f0) == n_nodes

    for i in range(n_nodes):
        inflow = [f0[i]]
        outflow = []

        for (u, v) in edges:
            if v == i:
                inflow.append(f_fwd[(u, v)])
            if u == i:
                outflow.append(f_fwd[(i, v)])
            if u == i:
                inflow.append(f_rev[(i, v)])
            if v == i:
                outflow.append(f_rev[(u, v)])

        prob += pulp.lpSum(inflow) - pulp.lpSum(outflow) == 1

    for (i, j) in edges:
        prob += f_fwd[(i, j)] <= M * x[(i, j)]
        prob += f_rev[(i, j)] <= M * x[(i, j)]

    for i in range(n_nodes):
        prob += f0[i] <= M * lam[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=False)) # мечтают ли дебаггеры об электрологах

    if prob.status != pulp.LpStatusOptimal:
        print("Статус решения:", pulp.LpStatus[prob.status])
        return None

    selected_edges = [(i, j) for (i, j) in edges if pulp.value(x[(i, j)]) > 0.5]
    total_cost = pulp.value(prob.objective)
    roots = [i for i in range(n_nodes) if pulp.value(lam[i]) > 0.5]

    degree = {i: 0 for i in range(n_nodes)}
    for (i, j) in selected_edges:
        degree[i] += 1
        degree[j] += 1

    return selected_edges, total_cost, roots, degree


n = 4
coords = {0: (0,0), 1: (2,0), 2: (2,2), 3: (0,2)}
edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
cost_dict = {}
for (i, j) in edges_list:
    xi, yi = coords[i]
    xj, yj = coords[j]
    cost_dict[(i, j)] = ((xi-xj)**2 + (yi-yj)**2) ** 0.5

result = solve_degree_constrained_mst_pulp(n, edges_list, cost_dict)

if result:
    edges, cost, roots, degrees = result
    print("Минимальное остовное дерево с ограничениями на степени")
    print(f"Рёбра: {edges}")
    print(f"Общая стоимость: {cost:.4f}")
    print(f"Корни: {roots}")
    print(f"Степени вершин: {degrees}")
else:
    print("Нет допустимого решения.")