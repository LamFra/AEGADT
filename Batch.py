import numpy as np
from sklearn import metrics, tree, datasets
from sklearn.model_selection import train_test_split

from DecisionTree import split_dataset, TRAIN_FILE_NAME

if __name__ == '__main__':
    MAX_DEPTH = 10
    # df, label = split_dataset(TRAIN_FILE_NAME)
    iris = datasets.load_iris()
    df = iris.data
    label = np.array([np.array([i]) for i in iris.target])
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20)
    scores = []
    for _ in range(1000):
        clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
        clf = clf.fit(X_train, y_train)
        scores.append(metrics.f1_score(y_test, clf.predict(X_test), average="micro"))
    print(max(scores))


