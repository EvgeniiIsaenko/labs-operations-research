# ======================
# Лаба №6, судоку
# ТЗ в ./lab6_docs./lab6_problem.docx
# ======================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def is_binary_solution(x):
    for value in x:
        if abs(value) > 1e-6 and abs(value - 1) > 1e-6:
            return False
    return True

def get_global_cells(puzzles, positions):
    cells = set()
    for n in range(len(puzzles)):
        row0, col0 = positions[n]
        for i in range(9):
            for j in range(9):
                cells.add((row0 + i, col0 + j))
    cells = sorted(list(cells))
    cell_number = {cell: idx for idx, cell in enumerate(cells)}
    return cells, cell_number

def variable_number(cell_number, row, col, value):
    cell_id = cell_number[(row, col)]
    return cell_id * 9 + value

def check_given_conflicts(puzzles, positions):
    given = {}

    for n in range(len(puzzles)):
        row0, col0 = positions[n]
        sudoku_matrix = puzzles[n]
        for i in range(9):
            for j in range(9):
                value = sudoku_matrix[i][j]
                if value != 0:
                    global_cell = (row0 + i, col0 + j)
                    if global_cell in given and given[global_cell] != value:
                        raise RuntimeError(
                            "Противоречие в пересечении: "
                            f"клетка {global_cell} имеет значения {given[global_cell]} и {value}"
                        )

                    given[global_cell] = value


def build_lp_matrices(puzzles, positions):
    check_given_conflicts(puzzles, positions)

    cells, cell_number = get_global_cells(puzzles, positions)

    n_vars = len(cells) * 9

    rows = []
    cols = []
    values = []
    b_eq = []

    def add_equation(var_list, rhs):
        row_id = len(b_eq)

        for var in var_list:
            rows.append(row_id)
            cols.append(var)
            values.append(1)

        b_eq.append(rhs)

    for row, col in cells:
        var_list = []

        for value in range(9):
            var_list.append(variable_number(cell_number, row, col, value))

        add_equation(var_list, 1)

    for n in range(len(puzzles)):
        row0, col0 = positions[n]

        for i in range(9):
            for value in range(9):
                var_list = []

                for j in range(9):
                    row = row0 + i
                    col = col0 + j
                    var_list.append(variable_number(cell_number, row, col, value))

                add_equation(var_list, 1)

        for j in range(9):
            for value in range(9):
                var_list = []

                for i in range(9):
                    row = row0 + i
                    col = col0 + j
                    var_list.append(variable_number(cell_number, row, col, value))

                add_equation(var_list, 1)

        # 3x3
        for block_i in range(3):
            for block_j in range(3):
                for value in range(9):
                    var_list = []

                    for i in range(block_i * 3, block_i * 3 + 3):
                        for j in range(block_j * 3, block_j * 3 + 3):
                            row = row0 + i
                            col = col0 + j
                            var_list.append(variable_number(cell_number, row, col, value))

                    add_equation(var_list, 1)

        for i in range(9):
            for j in range(9):
                value = puzzles[n][i][j]

                if value != 0:
                    row = row0 + i
                    col = col0 + j
                    var = variable_number(cell_number, row, col, value - 1)
                    add_equation([var], 1)

    A_eq = coo_matrix(
        (values, (rows, cols)),
        shape=(len(b_eq), n_vars)
    ).tocsr()

    b_eq = np.array(b_eq)

    bounds = []

    for _ in range(n_vars):
        bounds.append((0, 1))

    return A_eq, b_eq, bounds, cells, cell_number

def solve_sudoku_system(puzzles, positions, title, draw=True):
    A_eq, b_eq, bounds, cells, cell_number = build_lp_matrices(
        puzzles,
        positions
    )

    n_vars = A_eq.shape[1]

    best_result = None

    for attempt in range(20):
        rng = np.random.default_rng(attempt)
        c = rng.random(n_vars)

        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if result.success and is_binary_solution(result.x):
            best_result = result
            break

    if best_result is None:
        if result.success:
            raise RuntimeError(
                "что-то пошло не так, скорее всего дробное число"
            )
        else:
            raise RuntimeError("Решение не найдено: " + result.message)

    solutions, global_solution = extract_solutions(
        best_result,
        puzzles,
        positions,
        cells,
        cell_number
    )

    print("\n" + title)
    print("Статус:", best_result.message)

    if draw:
        draw_sudoku_system(puzzles, positions, title, solution_values=global_solution)

    return solutions

def extract_solutions(result, puzzles, positions, cells, cell_number):
    x = result.x

    global_solution = {}

    for row, col in cells:
        values = []

        for value in range(9):
            var = variable_number(cell_number, row, col, value)
            values.append(x[var])

        digit = int(np.argmax(values)) + 1
        global_solution[(row, col)] = digit

    solutions = []

    for n in range(len(puzzles)):
        row0, col0 = positions[n]
        sudoku = np.zeros((9, 9), dtype=int)

        for i in range(9):
            for j in range(9):
                sudoku[i, j] = global_solution[(row0 + i, col0 + j)]

        solutions.append(sudoku)

    return solutions, global_solution

def print_sudoku(sudoku):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("_" * 21)

        row_text = ""

        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_text += "| "

            row_text += str(sudoku[i][j]) + " "

        print(row_text)

def draw_sudoku_system(puzzles, positions, title, solution_values=None):
    cells, _ = get_global_cells(puzzles, positions)
    
    min_row = min(row for row, col in cells)
    max_row = max(row for row, col in cells)
    min_col = min(col for row, col in cells)
    max_col = max(col for row, col in cells)
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # fig не используем т.к. выводим все по очереди, просто чтобы замутить return -> tuple[fig,...] 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width * 0.8, height * 0.8))
    
    colors = ["blue", "red", "green"]
    
    def draw_grid(ax, global_vals):
        ax.set_xlim(min_col, max_col + 1)
        ax.set_ylim(max_row + 1, min_row)
        
        for n in range(len(puzzles)):
            row0, col0 = positions[n]
            main_color = colors[n % len(colors)]
            
            for t in range(10):
                if t % 3 == 0:
                    linewidth = 2.5
                    color = main_color
                else:
                    linewidth = 0.7
                    color = "gray"
                
                ax.plot([col0, col0 + 9], [row0 + t, row0 + t], color=color, linewidth=linewidth)
                ax.plot([col0 + t, col0 + t], [row0, row0 + 9], color=color, linewidth=linewidth)
        
        already_written = set()
        
        for n in range(len(puzzles)):
            row0, col0 = positions[n]
            
            for i in range(9):
                for j in range(9):
                    global_cell = (row0 + i, col0 + j)
                    
                    if global_cell in already_written:
                        continue
                    
                    if global_vals is None:
                        value = puzzles[n][i][j]
                    else:
                        value = global_vals.get(global_cell, 0)
                    
                    if value != 0:
                        ax.text(
                            col0 + j + 0.5,
                            row0 + i + 0.5,
                            str(value),
                            ha="center",
                            va="center",
                            fontsize=13,
                            color="black"
                        )
                    already_written.add(global_cell)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    
    draw_grid(ax1, None)
    ax1.set_title("Условие", fontsize=14)
    
    draw_grid(ax2, solution_values)
    ax2.set_title("Решение", fontsize=14)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

easy = [
    [2, 0, 5, 0, 0, 9, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 3, 0, 7],
    [7, 0, 0, 8, 5, 6, 0, 1, 0],
    # первые три
    [4, 5, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 8, 5],
    # вторые три
    [0, 2, 0, 4, 1, 8, 0, 0, 6],
    [6, 0, 8, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 2, 0, 0, 7, 0, 8]
    # последние три
]

medium = [
    [0, 0, 6, 0, 9, 0, 2, 0, 0],
    [0, 0, 0, 7, 0, 2, 0, 0, 0],
    [0, 9, 0, 5, 0, 8, 0, 7, 0],
    # первые три
    [9, 0, 0, 0, 3, 0, 0, 0, 6],
    [7, 5, 0, 0, 0, 0, 0, 1, 9],
    [1, 0, 0, 0, 4, 0, 0, 0, 5],
    # вторые три
    [0, 1, 0, 3, 0, 9, 0, 8, 0],
    [0, 0, 0, 2, 0, 1, 0, 0, 0],
    [0, 0, 9, 0, 8, 0, 1, 0, 0]
    # последние три
]

hard = [
    [0, 0, 0, 8, 0, 0, 0, 0, 0],
    [7, 8, 9, 0, 1, 0, 0, 0, 6],
    [0, 0, 0, 0, 0, 6, 1, 0, 0],
    # первые три
    [0, 0, 7, 0, 0, 0, 0, 5, 0],
    [5, 0, 8, 7, 0, 9, 3, 0, 4],
    [0, 4, 0, 0, 0, 0, 2, 0, 0],
    # вторые три
    [0, 0, 3, 2, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 7, 0, 4, 3, 9],
    [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # последние три
]

tuple_blue_1 = [
    [0, 0, 0, 0, 0, 2, 5, 0, 6],
    [7, 1, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # первые три   
    [0, 0, 9, 0, 0, 0, 4, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 8, 0, 0, 0, 3],
    # вторые три   
    [0, 0, 0, 2, 0, 0, 0, 5, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 2, 0, 0, 9, 0, 0, 0]
    # последние три  
]

tuple_red_1 = [
    [0, 0, 0, 4, 0, 0, 8, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 8, 0, 0, 0, 3, 0, 0, 0],
    # первые три              
    [2, 0, 0, 0, 5, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 9, 0, 0, 0, 5, 0, 0],
    # вторые три       
    [0, 6, 0, 0, 0, 0, 0, 2, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 7, 3, 0, 0, 0, 0, 0]
    # последние три
]

tuple_blue_2 = [
    [4, 0, 8, 3, 0, 0, 0, 0, 2],
    [7, 6, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 5, 0, 0, 0, 8, 7, 1],
    # первые три   
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 5, 0, 9, 7, 4, 0, 6, 0],
    [0, 7, 0, 0, 0, 6, 0, 0, 0],
    # вторые три   
    [0, 4, 1, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 1, 0, 4],
    [5, 0, 0, 0, 0, 1, 9, 0, 0]
    # последние три    
]

tuple_red_2 = [
    [0, 0, 0, 6, 9, 0, 0, 4, 2],
    [1, 0, 4, 8, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 0, 8, 0, 1],
    # первые три   
    [8, 6, 0, 2, 0, 7, 0, 0, 4],
    [0, 0, 0, 3, 8, 6, 0, 0, 0],
    [3, 0, 0, 9, 4, 5, 0, 2, 8],
    # вторые три       
    [2, 0, 8, 0, 0, 0, 1, 0, 5],
    [7, 0, 0, 0, 0, 0, 4, 0, 3],
    [6, 1, 0, 0, 5, 9, 0, 0, 0]
    # последние три
]


triplet_left = [
    [6, 0, 8, 0, 1, 0, 0, 0, 0],
    [0, 0, 7, 0, 4, 0, 8, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 9],
    # первые три   
    [0, 6, 0, 7, 0, 0, 0, 3, 0],
    [0, 7, 0, 5, 0, 4, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 9, 5, 0],
    # вторые три   
    [0, 0, 0, 0, 7, 8, 0, 0, 0],
    [5, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 6, 0, 0, 0]
    # последние три
]

triplet_right = [
    [0, 0, 0, 0, 6, 0, 3, 0, 9],
    [7, 0, 3, 0, 4, 0, 1, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 0],
    # первые три   
    [0, 7, 0, 0, 0, 6, 0, 9, 0],
    [0, 0, 0, 7, 0, 4, 0, 8, 0],
    [0, 5, 4, 0, 0, 9, 0, 0, 0],
    # вторые три   
    [0, 0, 0, 4, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 5],
    [0, 0, 0, 3, 0, 0, 4, 0, 0]
    # последние три    
]

triplet_down = [
    [0, 0, 0, 1, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    # первые три   
    [4, 0, 0, 0, 0, 0, 0, 0, 6],
    [0, 0, 0, 6, 5, 2, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 2, 0, 0],
    # вторые три   
    [0, 0, 0, 5, 0, 3, 0, 0, 0],
    [0, 8, 7, 0, 0, 0, 6, 5, 0],
    [0, 4, 0, 0, 0, 0, 0, 8, 0]
    # последние три
]

solve_sudoku_system([easy], [(0, 0)], "Easy")
solve_sudoku_system([medium], [(0, 0)], "Medium")
solve_sudoku_system([hard], [(0, 0)], "Hard")

solve_sudoku_system(
    [tuple_blue_1, tuple_red_1],
    [(0, 0), (3, 3)],
    "Двойное судоку №1"
)

solve_sudoku_system(
    [tuple_blue_2, tuple_red_2],
    [(0, 0), (6, 6)],
    "Двойное судоку №2"
)

solve_sudoku_system(
    [triplet_left, triplet_right, triplet_down],
    [(0, 0), (0, 12), (6, 6)],
    "Тройное судоку"
)
