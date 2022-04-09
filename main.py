import numpy as np
import csv
from sklearn import tree

FILE_NAME = "dataset/Skin_NonSkin.txt"


def load_dataset():
    with open(FILE_NAME, "r") as f:
        dataset = list(csv.reader(f, delimiter='\t'))
    data = np.array(dataset, dtype=object)
    return np.hsplit(data, [3, 4])[0], np.hsplit(data, [3, 4])[1]


if __name__ == '__main__':
    X, y = load_dataset()
    print(X)
    print(y)
    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    tree.export_graphviz(clf, out_file='tree.dot')
    """
