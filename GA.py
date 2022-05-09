from DecisionTree import *

TRAIN_FILE_NAME = "dataset/winequality-white.csv"


def initial_population(size, max_depth, _x_train, _y_train):
    decision_trees = [DecisionTree(max_depth=max_depth) for _ in range(size)]
    for ds in decision_trees:
        ds.fit(_x_train, _y_train)
    return decision_trees


def score_population(_population, _x_test, _y_test):
    return [p.f_measure_score(_x_test, _y_test) for p in _population]


if __name__ == '__main__':
    df, label = split_dataset(TRAIN_FILE_NAME)
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20)
    population = initial_population(5, 4, X_train, y_train)
    scores = score_population(population, X_test, y_test)
    print(population)
    print(scores)
