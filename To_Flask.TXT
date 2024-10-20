from flask import Flask, render_template, request, jsonify
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Ініціалізація DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Основний алгоритм
def main(coordinates, crossover_prob, mutation_prob, generations, population_size):
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
    
    pop, log = algorithms.eaSimple(pop, toolbox, crossover_prob, mutation_prob, generations, stats=None, halloffame=hof, verbose=False)
    
    best_ind = hof[0]
    best_mst_length = evalMST(best_ind)[0]

    # Побудова графіка
    fig, ax = plt.subplots()
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='b')  # Всі точки
    mst_matrix = minimum_spanning_tree(dist_matrix).toarray()
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if mst_matrix[i][j] != 0:
                ax.plot([coordinates[i][0], coordinates[j][0]], [coordinates[i][1], coordinates[j][1]], 'r-')

    # Збереження графіка у буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return best_mst_length, img_base64

# Головна сторінка з формою
@app.route('/')
def index():
    return render_template('index.html')

# Обробка форми
@app.route('/run', methods=['POST'])
def run():
    coord_input = request.form['coordinates']
    coordinates = np.array([list(map(float, point.split(','))) for point in coord_input.split(';')])
    crossover_prob = float(request.form['crossover_prob'])
    mutation_prob = float(request.form['mutation_prob'])
    generations = int(request.form['generations'])
    population_size = int(request.form['population_size'])

    # Виконання генетичного алгоритму
    best_mst_length, img_base64 = main(coordinates, crossover_prob, mutation_prob, generations, population_size)
    
    return jsonify({'mst_length': best_mst_length, 'graph': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
