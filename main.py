import pickle
import numpy as np

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

FILE_NAME = "dataset/winequality-white.csv"


def split_dataset(filename):
    with open(filename, "r") as f:
        dataset = pd.read_csv(filename, header=0, sep=';')
    data = np.array(dataset, dtype=int)
    x, y = np.hsplit(data, [11, 12])[0], np.hsplit(data, [11, 12])[1]
    return train_test_split(x, y)


def save_decision_tree(mod):
    tree.plot_tree(mod)
    tree.export_graphviz(mod, out_file='tree.dot')


def score_single_tree():
    x_tr, x_te, y_tr, y_te = split_dataset(FILE_NAME)
    model_name = 'models/decisionTree1.sav'
    try:
        mod = pickle.load(open(model_name, 'rb'))
    except FileNotFoundError:
        clf = tree.DecisionTreeClassifier()
        mod = clf.fit(x_tr, y_tr)
        pickle.dump(mod, open(model_name, 'wb'))
    result = mod.score(x_te, y_te)
    return result


if __name__ == '__main__':
    print(score_single_tree())