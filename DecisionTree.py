class DecisionTree(object):
    def __init__(self, fixed_depth):
        self.features = [None for _ in range(2**fixed_depth)]
        self.operators = [None for _ in range(2**fixed_depth)]
        self.values = [None for _ in range(2**fixed_depth)]

    def set_root(self, feature, operator, value):
        if self.features[0] is None:
            self.features[0] = feature
            self.operators[0] = operator
            self.values[0] = value
        else:
            print("Tree already had root")

    def set_left_child(self, parent_index, feature, operator, value):
        if self.features[parent_index] is not None:
            self.features[(2*parent_index)+1] = feature
            self.operators[(2*parent_index)+1] = operator
            self.values[(2*parent_index)+1] = value
        else:
            print("Can't set child, parent not found")

    def set_right_child(self, parent_index, feature, operator, value):
        if self.features[parent_index] is not None:
            self.features[(2*parent_index)+2] = feature
            self.operators[(2*parent_index)+2] = operator
            self.values[(2*parent_index)+2] = value
        else:
            print("Can't set child, parent not found")


if __name__ == '__main__':
    t = DecisionTree(fixed_depth=3)
    t.set_root("f", "<=", 0.5)
    t.set_left_child(0, "l", ">", 0.2)
    t.set_right_child(0, "r", "<", 0.1)
    print(t.features)
    print(t.operators)
    print(t.values)
