import numpy as np
import csv


FILE_NAME = "dataset/adult_dataset.data"


def load_dataset():
    with open(FILE_NAME, "r") as f:
        dataset = list(csv.reader(f, delimiter=','))

    return np.array(dataset, dtype=object)


if __name__ == '__main__':
    data = load_dataset()
    print(data)
