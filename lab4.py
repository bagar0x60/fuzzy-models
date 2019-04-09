import numpy as np
from itertools import combinations
import csv
import sys

def load_deffuzed_table(filename):
    reader = csv.reader(filename)
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        return np.array([[float(x) for x in row[1:]] for row in reader])

def weights_matrix(n):
    result = []
    for c in combinations(range(n + 2), 2):
        w1, w2, w3 = c[0], c[1] - c[0] - 1, n + 1 - c[1]
        result.append([w1, w2, w3])
    return np.array(result) / n

def domination_confidence(synthesizing_function):
    result = [3*[0] for _ in range(3)]
    rows_count = np.size(synthesizing_function, 0)
    for i in range(3):
        for j in range(3):
            result[i][j] = np.sum(synthesizing_function[:,i] > synthesizing_function[:,j]) / rows_count
    return result

def process(table, n):
    print(f'Величина дискретизации: {n}')
    table = np.array(table)
    print("Деффузифицированная таблица")
    print(table)

    wm = weights_matrix(n)
    print("Весовые коэффициенты:")
    print(wm)

    synthesizing_function = wm @ table.T
    print("Синтезирующая функция:")
    print(synthesizing_function)

    expected_value = np.mean(synthesizing_function, 0)
    standard_deviation = np.std(synthesizing_function, 0)

    print("E[Q]:")
    print(expected_value)
    print("S:")
    print(standard_deviation)

    best = np.argmax(expected_value) + 1
    print(f"Объект {best} лучший по математическому ожиданию.")

    dc = domination_confidence(synthesizing_function)
    print("Вероятности доминирования:")
    other1, other2 = {1, 2, 3} - {best}
    for i, j in [(best, other1), (best, other2), (other1, other2)]:
        print(f"P(Q{i} > Q{j}) = {dc[i - 1][j - 1]}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Использование: python lab4.py <Table> <N>")
        exit(1)

    table_file_name = sys.argv[1]
    table = load_deffuzed_table(table_file_name)
    n = int(sys.argv[2])

    process(table, n)