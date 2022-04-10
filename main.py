import pickle
import numpy as np
import csv
from sklearn import tree
from sklearn.model_selection import train_test_split

FILE_NAME = "dataset/Skin_NonSkin.txt"


def load_dataset():
    with open(FILE_NAME, "r") as f:
        dataset = list(csv.reader(f, delimiter='\t'))
    data = np.array(dataset, dtype=int)
    return np.hsplit(data, [3, 4])[0], np.hsplit(data, [3, 4])[1]


def save_decision_tree(mod):
    tree.plot_tree(mod)
    tree.export_graphviz(mod, out_file='tree.dot')


if __name__ == '__main__':
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model_name = 'models/decisionTree1.sav'
    model = ""

    try:
        model = pickle.load(open(model_name, 'rb'))
    except FileNotFoundError:
        clf = tree.DecisionTreeClassifier()
        model = clf.fit(x_train, y_train)
        pickle.dump(model, open(model_name, 'wb'))

    result = model.score(x_test, y_test)
    print(result)