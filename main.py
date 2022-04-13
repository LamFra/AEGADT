import pickle
import random

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

TRAIN_FILE_NAME = "dataset/winequality-white.csv"
TEST_FILE_NAME = "dataset/winequality-white_test.csv"
POPULATION = 50
GENERATION = 3


def split_dataset(filename):
    with open(filename, "r"):
        dataset = pd.read_csv(filename, header=0, sep=';')
    data = np.array(dataset, dtype=int)
    x, y = np.hsplit(data, [11, 12])[0], np.hsplit(data, [11, 12])[1]
    return train_test_split(x, y)


def split_train_set(filename):
    with open(filename, "r"):
        dataset = pd.read_csv(filename, header=0, sep=';')
    data = np.array(dataset, dtype=int)
    return np.hsplit(data, [11, 12])[0], np.hsplit(data, [11, 12])[1]


def save_decision_tree(mod):
    tree.plot_tree(mod)
    tree.export_graphviz(mod, out_file='tree.dot')


def generate_initial_population(population):
    def generate_tree():
        x_tr, x_te, y_tr, y_te = split_dataset(TRAIN_FILE_NAME)
        clf = tree.DecisionTreeClassifier()
        return clf.fit(x_tr, y_tr)

    mod = []
    for _ in range(population):
        mod.append(generate_tree())
    return mod


def score_single_tree():
    x_tr, x_te, y_tr, y_te = split_dataset(TRAIN_FILE_NAME)
    model_name = 'models/decisionTree1.sav'
    try:
        mod = pickle.load(open(model_name, 'rb'))
    except FileNotFoundError:
        clf = tree.DecisionTreeClassifier()
        mod = clf.fit(x_tr, y_tr)
        pickle.dump(mod, open(model_name, 'wb'))
    result = mod.score(x_te, y_te)
    return result


def calculate_accuracy(mod):
    x_test, y_test = split_train_set(TEST_FILE_NAME)
    a = []
    for m in mod:
        a.append(m.score(x_test, y_test))
    d = []
    for i, j in zip(a, mod):
        d.append((i, j))
    d.sort(key=lambda tup: tup[0], reverse=True)
    return d


def next_gen(actual_gen):
    def generate_child(m):
        ch = VotingClassifier(m, voting="hard")
        x_tr, x_te, y_tr, y_te = split_dataset(TRAIN_FILE_NAME)
        return ch.fit(x_tr, y_tr.ravel())

    def random_combination(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        a = []
        for j in indices:
            a.append(pool[j])
        return a

    mod = []
    for c, i in zip(actual_gen, range(len(actual_gen))):
        mod.append(("clf" + str(i), c[1]))

    new_gen = []
    for _ in range(POPULATION):
        new_gen.append(generate_child(random_combination(mod, 3)))

    return new_gen


if __name__ == '__main__':
    s = 0.3
    models = generate_initial_population(POPULATION)
    generation = calculate_accuracy(models)
    save_decision_tree(generation[0][1])
    for i in range(GENERATION-1):
        generation = generation[:int(len(generation) * s)]
        print("Accuracy generation %d: %s" % (i+1, generation[0][0]))
        ng = next_gen(generation)
        array = []
        for n in ng:
            x_tr, x_te, y_tr, y_te = split_dataset(TRAIN_FILE_NAME)
            array.append(n.fit(x_tr, y_tr.ravel()))
        newgen = calculate_accuracy(array)
        generation = newgen

    print("Accuracy generation %d: %s" % (GENERATION, generation[0][0]))
