# from random import seed
import math
from random import randint
from csv import reader
from math import log
from sklearn.model_selection import KFold
import numpy as np
import time
from tqdm import tqdm


# Input
def load_csv(filename):
    dataset = list()
    data_id = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset.pop(0)
    for i in range(len(dataset)):
        data_id.append(dataset[i][0])
        dataset[i].pop(0)
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            dataset[i][j] = float(dataset[i][j])
    return dataset


# Divide dataset
def split_train_test(dataset, ratio):
    num = len(dataset)
    train_num = int((1 - ratio) * num)
    dataset_copy = list(dataset)
    traindata = list()
    while len(traindata) < train_num:
        index = randint(0, len(dataset_copy) - 1)
        traindata.append(dataset_copy.pop(index))
    testdata = dataset_copy
    return traindata, testdata


# Sigmoid Function
def ActivationFunction(value):
    e = 2.671
    expo = e ** value
    val = expo / (1 + expo)
    return val


# Split dataset
def data_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# ICR
def calc_cf(groups, class_values):
    a = 0
    b = 0
    c = 0
    d = 0
    b_ncf = 0
    g = 0
    total_size = 0
    for group in groups:
        total_size += len(group)
        # print(group)
    for class_value in class_values:
        for group in groups:
            if g == 0:
                size = len(group)
                proportion = [row[-1] for row in group].count(class_value)
                a = proportion
                b = size - proportion
                if b == 0:
                    b = 0.01
                if a == 0:
                    a = 0.01
                g = 1
            else:
                size = len(group)
                proportion = [row[-1] for row in group].count(class_value)
                c = proportion
                d = size - proportion
                if d == 0:
                    d = 0.01
                if c == 0:
                    c = 0.01
                g = 0
        cf = (a / (a + b)) / (c / (c + d))
        ncf = log(cf)
        if ncf <= 0:
            ncf = - ncf
        ncf = ActivationFunction(ncf)
        if ncf > b_ncf:
            b_ncf = ncf
    return b_ncf


# ICR
def get_split_cf(dataset, features):
    class_values = list(set(row[-1] for row in dataset))
    # print('class_values', class_values)
    b_index, b_value, b_score, b_groups = 0, 0, 0, None

    for index in features:
        values = list(set(dataset[row][index] for row in range(len(dataset))))
        for row in values:
            groups = data_split(index, row, dataset)
            cf = calc_cf(groups, class_values)
            if cf >= b_score:
                b_index, b_value, b_score, b_groups = index, row, cf, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}  # node


# Split node
def split(node, max_depth, min_size, features, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split_cf(left, features)
        split(node['left'], max_depth, min_size, features, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split_cf(right, features)
        split(node['right'], max_depth, min_size, features, depth + 1)


# Build one tree
def build_one_tree(train, max_depth, min_size, n_features):
    features = []
    while len(features) < n_features:
        index = randint(0, len(dataset[0]) - 2)
        if index not in features:
            features.append(index)
    root = get_split_cf(train, features)
    split(root, max_depth, min_size, features, 1)
    return root


# Predict
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Vote
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# RF
class randomForest:
    def __init__(self, trees_num, max_depth, leaf_min_size, sample_ratio, feature_ratio):
        self.trees_num = trees_num
        self.max_depth = max_depth
        self.leaf_min_size = leaf_min_size
        self.samples_split_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.trees = list()

    # Sample
    def sample_split(self, dataset):
        sample = list()
        n_sample = round(len(dataset) * self.samples_split_ratio)
        while len(sample) < n_sample:
            index = randint(0, len(dataset) - 2)
            sample.append(dataset[index])
        return sample

    def build_randomforest(self, train):
        max_depth = self.max_depth
        min_size = self.leaf_min_size
        n_trees = self.trees_num
        n_features = int(self.feature_ratio * (len(train[0]) - 1))
        # for i in range(n_trees):
        for i in tqdm(range(1, n_trees)):
            # print(i, 'th', 'tree')
            sample = self.sample_split(train)
            tree = build_one_tree(sample, max_depth, min_size, n_features)
            # print(tree)
            time.sleep(0.05)
            self.trees.append(tree)
        return self.trees

    def bagging_predict(self, onetestdata):
        predictions = [predict(tree, onetestdata) for tree in self.trees]
        # print(predictions.count(onetestdata[-1]))
        return max(set(predictions), key=predictions.count)

    def accuracy_metric_x(self, testdata, k, b):
        # print(testdata)
        correct = 0
        x = 0
        y = 0
        TF = np.zeros((k, k))
        # print('b', b)
        for i in range(len(testdata)):
            predicted = self.bagging_predict(testdata[i])
            if predicted == testdata[i][-1]:
                correct += 1
            for l in range(len(b)):
                if predicted == testdata[i][-1]:
                    if predicted == b[l]:
                        x = l
                        y = l
                elif predicted == b[l]:
                    x = l
                    for m in range(len(b)):
                        if testdata[i][-1] == b[m]:
                            y = m
            TF[x][y] += 1
        return correct / float(len(testdata)), TF


if __name__ == '__main__':
    # seed(1)
    path = 'musk1.csv'
    k = 5  # k-fold
    C = 1
    print(path)
    data = load_csv(path)
    data_m, data_n = np.shape(data)
    print('(#sample, #features):', np.shape(data))

    dataset = []
    index_0 = [i for i in range(len(data))]
    # print(index_0)
    np.random.shuffle(index_0)
    # print(index_0)
    for i in range(len(index_0)):
        dataset.append(data[index_0[i]])

    max_depth = 30
    min_size = 1
    sample_ratio = 0.7
    trees_num = 100
    feature_ratio = 1
    # feature_ratio = 2 * math.sqrt(data_n) / data_n
    # feature_ratio = (log(data_n, 2) + C) / data_n
    myRF = randomForest(trees_num+1, max_depth, min_size, sample_ratio, feature_ratio)

    b = []
    c = 0
    x = 0
    l = 0
    # print('fold:', k)
    for i in range(len(dataset)):
        if i == 0:
            b.append(dataset[i][-1])
            l += 1
        for a in range(len(b)):
            if b[a] == dataset[i][-1]:
                c += 1
        if c == 0:
            b.append(dataset[i][-1])
            l += 1
        else:
            c = 0

    for i in range(len(b) - 1):
        for j in range(len(b) - i - 1):
            if b[j] > b[j + 1]:
                b[j], b[j + 1] = b[j + 1], b[j]
    TF = np.zeros((l, l))
    print('classes', b)
    print('Start train')
    gkf = KFold(n_splits=k)
    acc_list = []
    start_time = time.process_time()
    for train_index, test_index in gkf.split(dataset):
        traindata, testdata = np.array(dataset)[train_index], np.array(dataset)[test_index]
        x += 1
        print('fold:', x)
        myRF.build_randomforest(traindata)
        acc, tf = myRF.accuracy_metric_x(testdata[:-1], l, b)
        print(tf)
        print(x, 'th acc:', acc)
        TF += tf
        acc_list.append(acc)

    print(k, 'Five results', acc_list)
    print(TF / k)
    print('Average acc', np.mean(acc_list))
    print('acc-var', np.var(acc_list))

    end_time = time.process_time()
    print("the running time is: " + str(end_time - start_time))
