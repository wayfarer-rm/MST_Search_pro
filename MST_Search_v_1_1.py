import tkinter as tk
from tkinter import ttk
import numpy as np
import time
from deap import base, creator, tools, algorithms
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# Ініціалізація DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Основний алгоритм
def main(coordinates, crossover_prob, mutation_prob, generations, population_size, ax):
    dist_matrix = distance_matrix(coordinates, coordinates)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(coordinates)), len(coordinates))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalMST(individual):
        mst_matrix = minimum_spanning_tree(dist_matrix[individual][:, individual])
        total_length = mst_matrix.sum()
        return (total_length,)

    toolbox.register("evaluate", evalMST)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    start_time = time.time()
    pop, log = algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob, generations, stats=stats, halloffame=hof, verbose=True)
    end_time = time.time()
    execution_time = end_time - start_time

    best_ind = hof[0]
    best_mst_length = evalMST(best_ind)[0]
    print(f"Best individual is {best_ind} with fitness {best_ind.fitness.values} and MST length {best_mst_length}")

    # Візуалізація найкращого рішення
    ax.clear()
    ax.set_facecolor('lightgrey')  # Сірий фон для графіка
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='b')  # Всі точки
    mst_matrix = minimum_spanning_tree(dist_matrix).toarray()
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if mst_matrix[i][j] != 0:
                ax.plot([coordinates[i][0], coordinates[j][0]], [coordinates[i][1], coordinates[j][1]], 'r-')
    
    return execution_time, best_mst_length

# Функція запуску з GUI
def run():
    coord_input = coord_entry.get()
    coordinates = np.array([list(map(float, point.split(','))) for point in coord_input.split(';')])
    crossover_prob = float(crossover_prob_entry.get())
    mutation_prob = float(mutation_prob_entry.get())
    generations = int(generations_entry.get())
    population_size = int(population_size_entry.get())

    # Виконання генетичного алгоритму і візуалізація результатів
    execution_time, best_mst_length = main(coordinates, crossover_prob, mutation_prob, generations, population_size, ax)
    execution_time_label.config(text=f"Час виконання: {execution_time:.2f} секунд, Довжина MST: {best_mst_length}")
    canvas.draw()

# Налаштування головного вікна Tkinter
root = tk.Tk()
root.title("MST Search v.1.1")

# Дозволяємо масштабування вікна
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)
root.grid_rowconfigure(6, weight=1)

# Поля вводу для користувача
coord_label = tk.Label(root, text="Координати (x,y; x,y; ...):")
coord_label.grid(row=0, column=0, sticky='ew')
coord_entry = tk.Entry(root)
coord_entry.grid(row=0, column=1, sticky='ew')

crossover_prob_label = tk.Label(root, text="Ймовірність схрещування:")
crossover_prob_label.grid(row=1, column=0, sticky='ew')
crossover_prob_entry = tk.Entry(root)
crossover_prob_entry.grid(row=1, column=1, sticky='ew')

mutation_prob_label = tk.Label(root, text="Ймовірність мутації:")
mutation_prob_label.grid(row=2, column=0, sticky='ew')
mutation_prob_entry = tk.Entry(root)
mutation_prob_entry.grid(row=2, column=1, sticky='ew')

generations_label = tk.Label(root, text="Кількість поколінь:")
generations_label.grid(row=3, column=0, sticky='ew')
generations_entry = tk.Entry(root)
generations_entry.grid(row=3, column=1, sticky='ew')

population_size_label = tk.Label(root, text="Розмір популяції:")
population_size_label.grid(row=4, column=0, sticky='ew')
population_size_entry = tk.Entry(root)
population_size_entry.grid(row=4, column=1, sticky='ew')

run_button = tk.Button(root, text="Запустити", command=run)
run_button.grid(row=5, column=0, columnspan=2, sticky='ew')

# Підготовка місця для графіка та часу виконання
fig, ax = plt.subplots()
fig.patch.set_facecolor('lightgrey')  # Сірий фон для фігури
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky='ew')
execution_time_label = ttk.Label(root, text="")
execution_time_label.grid(row=7, column=0, columnspan=2, sticky='ew')

root.mainloop()
