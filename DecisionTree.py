from sklearn.model_selection import train_test_split
from main import TRAIN_FILE_NAME
import pandas as pd
import numpy as np
import operator


def split_dataset(filename):
    with open(filename, "r"):
        dataset = pd.read_csv(filename, header=0, sep=';')
    wine = np.array(dataset, dtype=float)
    data, label = np.hsplit(wine, [11, 12])[0], np.hsplit(wine, [11, 12])[1]
    return data, label


class DecisionTree(object):
    ops = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt
    }

    def __init__(self, max_depth):
        self.features = [None for _ in range(2 ** (max_depth - 1) - 1)]
        self.operators = [None for _ in range(2 ** (max_depth - 1) - 1)]
        self.values = [None for _ in range(2 ** (max_depth - 1) - 1)]
        self.labels = [None for _ in range(2 ** (max_depth - 1) - 1)]
        self.max_depth = max_depth

    def set_root(self, feature, operator, value):
        self.features[0] = feature
        self.operators[0] = operator
        self.values[0] = value

    def set_left_child(self, parent_index, feature, operator, value):
        self.features[(2 * parent_index) + 1] = feature
        self.operators[(2 * parent_index) + 1] = operator
        self.values[(2 * parent_index) + 1] = value

    def set_right_child(self, parent_index, feature, operator, value):
        self.features[(2 * parent_index) + 2] = feature
        self.operators[(2 * parent_index) + 2] = operator
        self.values[(2 * parent_index) + 2] = value

    def fit(self, _X_train, _y_train):
        self.features = np.random.choice(len(_X_train[0]), size=len(self.features))
        self.operators = np.random.choice(list(self.ops.values()), size=len(self.operators))
        self.values = [np.random.uniform(min(_X_train[:, i]), max(_X_train[:, i])) for i in self.features]
        self.labels += list(np.random.randint(min(_y_train[:, 0]), max(_y_train[:, 0]), size=2 ** (self.max_depth-1)))


if __name__ == '__main__':
    df, label = split_dataset(TRAIN_FILE_NAME)
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=101)
    t = DecisionTree(max_depth=4)
    t.fit(X_train, y_train)
    print(t.features)
    print(t.operators)
    print(t.values)
    print(t.labels)
    #print(list(t.ops.values()))