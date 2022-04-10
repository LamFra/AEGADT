import pickle
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

TRAIN_FILE_NAME = "dataset/winequality-white.csv"
TEST_FILE_NAME = "dataset/winequality-white_test.csv"


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


def nex_gen(actual_gen):
    def generate_child(m):
        child = VotingClassifier(m, voting="hard")
        x_tr, x_te, y_tr, y_te = split_dataset(TRAIN_FILE_NAME)
        return child.fit(x_tr, y_tr.ravel())

    mod = []
    for c, i in zip(actual_gen, range(len(actual_gen))):
        mod.append(("clf" + str(i), c[1]))

    x, y = split_train_set(TEST_FILE_NAME)
    child = generate_child(mod)
    return child.score(x, y)


if __name__ == '__main__':
    initial_population = 10
    s = 0.3
    models = generate_initial_population(initial_population)
    generation = calculate_accuracy(models)
    generation = generation[:int(len(generation) * s)]
    print("New gen: ", nex_gen(generation))
    print("Old gen", generation)
