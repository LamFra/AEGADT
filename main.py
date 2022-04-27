import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from sklearn.datasets import load_breast_cancer

TRAIN_FILE_NAME = "dataset/winequality-white.csv"
TEST_FILE_NAME = "dataset/winequality-white_test.csv"


# defining various steps required for the genetic algorithm
def initilization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[:int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:, chromosome], y_train)
        predictions = logmodel.predict(X_test.iloc[:, chromosome])
        scores.append(precision_score(y_test, predictions, average='micro') * 0.4 + recall_score(y_test, predictions, average='micro') * 0.3 + f1_score(y_test, predictions, average='micro') * 0.3)
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    population_nextgen = pop_after_sel
    for i in range(len(pop_after_sel)):
        child = pop_after_sel[i]
        child[3:7] = pop_after_sel[(i + 1) % len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j] = not chromosome[j]
        population_nextgen.append(chromosome)
    # print(population_nextgen)
    return population_nextgen


def generations(size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, y_train, y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        # print(scores[:2])
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score


def split_dataset(filename):
    with open(filename, "r"):
        dataset = pd.read_csv(filename, header=0, sep=';')
    features = np.genfromtxt(filename, delimiter=';', dtype=str, max_rows=1)
    features = features[:-1]
    wine = np.array(dataset, dtype=float)
    data, label = np.hsplit(wine, [11, 12])[0], np.hsplit(wine, [11, 12])[1]
    df = pd.DataFrame(data, columns=features)
    return df, label


if __name__ == "__main__":
    # splitting the model into training and testing set
    df, label = split_dataset(TRAIN_FILE_NAME)
    # splitting the model into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.30, random_state=101)
    logmodel = tree.DecisionTreeClassifier(max_depth=4)
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    print("Metric = " + str(accuracy_score(y_test, predictions)))

    chromo, score = generations(size=10, n_feat=11, n_parents=5, mutation_rate=0.10, n_gen=10, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    logmodel.fit(X_train.iloc[:, chromo[-1]], y_train)
    predictions = logmodel.predict(X_test.iloc[:, chromo[-1]])
    print("Metric score after genetic algorithm is= " + str(accuracy_score(y_test, predictions)))