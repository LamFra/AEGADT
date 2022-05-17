from DecisionTree import *

TRAIN_FILE_NAME = "dataset/winequality-white.csv"


def initial_population(size, max_depth, _x_train, _y_train):
    decision_trees = [DecisionTree(max_depth=max_depth) for _ in range(size)]
    for ds in decision_trees:
        ds.fit(_x_train, _y_train)
    return decision_trees


def score_population(_population, _x_test, _y_test):
    return [p.f_measure_score(_x_test, _y_test) for p in _population]


def roulette_wheel_selection(pop_after_fit, fitness, n_parents):
    population_fitness = sum(fitness)
    chromosome_probabilities = [f / population_fitness for f in fitness]
    indexes = np.random.choice(len(pop_after_fit), size=n_parents, p=chromosome_probabilities)
    return list(np.array(pop_after_fit)[indexes.astype(int)])


def crossover(pop_after_sel, n_child):
    _pop_after_crossover = []
    for _ in range(int(n_child / 2)):
        for a in one_point_crossover(pop_after_sel[np.random.choice(range(len(pop_after_sel)))],
                                     pop_after_sel[np.random.choice(range(len(pop_after_sel)))]):
            _pop_after_crossover.append(a)
    if len(_pop_after_crossover) < n_child:
        _pop_after_crossover.append(one_point_crossover(pop_after_sel[np.random.choice(range(len(pop_after_sel)))],
                                                        pop_after_sel[np.random.choice(range(len(pop_after_sel)))])[
                                        np.random.choice(range(2))])
    return _pop_after_crossover


def mutation(pop_after_cross, _x_train, _y_train):
    pop_mutate = list(
        np.array(pop_after_cross)[np.random.choice([True, False], size=len(pop_after_cross), p=[0.15, 0.85])])
    for ds in pop_mutate:
        ds.random_resetting(_x_train, _y_train)


def create_new_generation(_population, _max_depth, _num_individual, _selection):
    scores = score_population(_population, X_test, y_test)
    pop_sel = roulette_wheel_selection(population, scores, _selection)
    pop_cros = crossover(pop_sel, _num_individual)
    mutation(pop_cros, X_train, y_test)
    return pop_cros


if __name__ == '__main__':
    max_depth = 10
    num_individual = 100
    selection = 20
    df, label = split_dataset(TRAIN_FILE_NAME)
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20)
    population = initial_population(num_individual, max_depth, X_train, y_train)
    print("Generation 0, max score: %f" % max(score_population(population, X_test, y_test)))
    i = 1
    while True:
        new_gen = create_new_generation(population, max_depth, num_individual, selection)
        print("Generation %d, max score: %f" % (i, max(score_population(new_gen, X_test, y_test))))
        i += 1
