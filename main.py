import numpy as np
import csv
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FILE_NAME = "dataset/Skin_NonSkin.txt"


def load_dataset():
    with open(FILE_NAME, "r") as f:
        dataset = list(csv.reader(f, delimiter='\t'))
    data = np.array(dataset, dtype=int)
    return np.hsplit(data, [3, 4])[0], np.hsplit(data, [3, 4])[1]


if __name__ == '__main__':
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    tree.plot_tree(clf)
    # tree.export_graphviz(clf, out_file='tree.dot')
    predictions = clf.predict(x_test)
    # plt.scatter(y_test, predictions)
    print(len(predictions), len(y_test))
    result = []
    for i, j in zip(predictions, y_test):
        result.append(i == j)
    print(result.count(True))
    print(result.count(False))
