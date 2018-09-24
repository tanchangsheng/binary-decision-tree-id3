import sys
import math
import csv
import numpy as np



# H(Y|X = x) for each value of Y

if len(sys.argv) != 7:
    sys.exit("You are missing an argument")

TRAIN_INPUT = sys.argv[1]
TEST_INPUT = sys.argv[2]
MAX_DEPTH = int(sys.argv[3])
TRAIN_OUT = sys.argv[4]
TEST_OUT = sys.argv[5]
METRICS_OUT = sys.argv[6]


class Node:
    def __init__(self):
        self.attribute_to_split_by = None
        self.index_of_attribute_to_split_by = None
        self.label_ratio = None
        # self.child_tree_labels = 
        self.child_trees = None
        self.label = None

# read csv to numpy array

def read_data_to_np_array(filepath):

    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
        return np.array(data)

# H(Y)
def get_marginal_entropy(label_ratio):

    # when there is only 1 label left
    if len(label_ratio) == 1:

        return 0

    if int(label_ratio[0][1]) > int(label_ratio[1][1]):

        majority_class_count = int(label_ratio[0][1])
        minority_class_count = int(label_ratio[1][1])

    else:

        majority_class_count = int(label_ratio[1][1])
        minority_class_count = int(label_ratio[0][1])
        
    count = majority_class_count + minority_class_count

    entropy = -1 * (majority_class_count / count * math.log(majority_class_count / count, 2) + \
            minority_class_count / count * math.log(minority_class_count / count, 2))

    return entropy

# H(Y|X=x)
def get_specific_entropy(x, y):

    result = 0

    y_ratio = get_label_ratio(y)

    for row in y_ratio:

        unique_y = row[0]
        unique_y_count = int(row[1])
        unique_y_count_prob = unique_y_count / len(y)

        result += unique_y_count_prob * math.log(unique_y_count_prob,2)

    return -1 * result

# H(Y|X)
def overall_entropy(x, y):

    result = 0

    # get each value of x its respective count of rows
    x_ratio = get_label_ratio(x)

    data = np.column_stack((x,y))

    for row in x_ratio:

        unique_x = row[0]
        unique_x_count = int(row[1])
        unique_x_count_prob = unique_x_count / len(x)

        # filter rows of data by x value
        filtered_data = data[unique_x == data[:,0]]
        filtered_x = filtered_data[:,0]
        filtered_y = filtered_data[:,1]
        result += unique_x_count_prob * get_specific_entropy(filtered_x, filtered_y)

    return result


# Information gain I(Y;X) = H(Y) - H(Y|X)
def get_mutual_information(x, y, marginal_entropy):

    return marginal_entropy - overall_entropy(x,y)

def get_majority_label(label_ratio):

    # when there is only 1 label
    if len(label_ratio) == 1:

        return label_ratio[0][0]

    if int(label_ratio[0][1]) > int(label_ratio[1][1]):
        return label_ratio[0][0]
    return label_ratio[1][0]

def get_label_ratio(labels, fixed_labels=None):

    unique, counts = np.unique(labels, return_counts=True)

    # if getting ratio for non Y
    if fixed_labels is None:
        return np.asarray((unique, counts)).T
    
    # if getting ratio for Y and Y has 2 values left
    if len(labels) == 2:
        return np.asarray((unique, counts)).T
    

# split node only if mutual information > 0
# Handle when depth > no. of variables
# Handle when max-depth is zero
# Use a majority vote of the labels at each leaf to make classification decisions.

def train_tree(node, train_x, train_y, attributes, depth, max_depth):

    # store label ratio
    label_ratio = get_label_ratio(train_y)

    node.label_ratio = label_ratio

    print(label_ratio.tolist())
    # print("[15 democrat /13 republican]")
    # print("[%s %s /%s %s]" % (label_ratio))

    # store majority vote at this node
    node.label = get_majority_label(label_ratio)

    # return if max depth reached or H(Y) == 0

    if depth == max_depth:

        return node

    marginal_entropy = get_marginal_entropy(label_ratio)

    # if H(Y) is 0, stop split
    if marginal_entropy == 0:

        return node

    # find the attribute that has highest I(Y;X)
    mutual_information_list = []

    for i in range(len(attributes)):

        attribute = attributes[i]

        # pass in column data with label data to calculate Mutual Information/Information Gain
        mi = get_mutual_information(train_x[:,i], train_y, marginal_entropy)
        mutual_information_list.append([attributes[i], mi, i])

    sorted_mi = sorted(mutual_information_list, key=lambda x:float(x[1]), reverse=True)

    highest_mi = sorted_mi[0][1]

    if highest_mi == 0:

        return node

    # store attribute to split by
    attribute_to_split_by = sorted_mi[0][0]
    index_of_attribute_to_split_by = sorted_mi[0][2]


    node.attribute_to_split_by = attribute_to_split_by
    node.index_of_attribute_to_split_by = index_of_attribute_to_split_by

    # for each value in attribute_to_split_by, create descendent tree
        # - split data into group by values
        # - create descendent tree on the group
    values_dict = {}

    for i in range(len(train_x)):
        row = train_x[i].tolist()
        label = train_y[i]
        value = row[index_of_attribute_to_split_by]
        if value not in values_dict:
            values_dict[value] = {
                "train_x" : [row],
                "train_y" : [label]
            }
        else:

            values_dict[value]["train_x"].append(row)
            values_dict[value]["train_y"].append(label)
            # values_dict[value]["train_x"].append(row)
            # values_dict[value]["train_y"].append(label)

    child_trees = {}

    for value in values_dict:

        value_groups = values_dict[value]

        group_train_x = np.array(value_groups["train_x"])
        group_train_y = np.array(value_groups["train_y"])

        for i in range(depth + 1):
            print("|  ", end='')
        
        print("%s = %s: " % (attribute_to_split_by, value), end='')
        # print(attribute_to_split_by + " = " + value + ": ")

        descendent_node = Node()
        child_tree = train_tree(descendent_node, group_train_x, group_train_y, attributes, depth + 1, max_depth)
        child_trees[value] = child_tree

    node.child_trees = child_trees

    return node


# use a trained tree to predict labels of given data
def predict(trained_tree, test_x):

    # this means that trained_tree.child_trees[value_of_attribute_to_split_by] has no child_trees
    # when there are no child_trees, there is a label
    if trained_tree.child_trees is None:
        
        return trained_tree.label

    # get attribute to split by in tree
    # attribute_to_split_by is the index of the attribute in the data
    index_of_attribute_to_split_by = trained_tree.index_of_attribute_to_split_by

    # get value of test_x for the attribute to split by
    value_of_attribute_to_split_by = test_x[index_of_attribute_to_split_by]

    # if value of attribute to split by exist in child tree, can go into child tree
    if value_of_attribute_to_split_by in trained_tree.child_trees:

        return predict(trained_tree.child_trees[value_of_attribute_to_split_by], test_x)
    
    # else child tree does not exist and return majority label at that node
    else:
        
        return trained_tree.label


def test(trained_tree, data):

    labels = []

    correct_count = 0
    wrong_count = 0

    data = data[1:]

    for row in data:

        pred_y = predict(trained_tree, row)
        # print(pred_y)
        true_y = row[-1]
        labels.append(pred_y)

        if pred_y == true_y:
            correct_count += 1
        else:
            wrong_count += 1

    result = {
        "labels" : labels,
        "error" : wrong_count / len(data)
    }

    return result


def main():

    global TRAIN_INPUT
    global TEST_INPUT
    global MAX_DEPTH
    global TRAIN_OUT
    global TEST_OUT
    global METRICS_OUT

    # read data into 2D numpy
    train_data = read_data_to_np_array(TRAIN_INPUT)
    attributes = train_data[0,:-1]
    train_x = train_data[1:,:-1]
    train_y = train_data[1:,-1]

    # if max depth > number of variables, max depth = number of variables
    if MAX_DEPTH > train_data[0, :].size - 1:
        MAX_DEPTH = train_data[0, :].size - 1

    # train
    root = Node()

    tree = train_tree(root, train_x, train_y, attributes, 0, MAX_DEPTH)

    train_result = test(tree ,train_data)
    # write predicted labels to file
    with open(TRAIN_OUT, "w") as f:

        f.write("\n".join(train_result["labels"]))

    # test 
    test_data = read_data_to_np_array(TEST_INPUT)

    test_result = test(tree, test_data)

    # write predicted labels to file
    with open(TEST_OUT, "w") as f:

        f.write("\n".join(test_result["labels"]))

    # write metrics
    with open(METRICS_OUT, 'w') as f:

        f.write("error(train): %s\n" % train_result["error"])
        f.write("error(test): %s" % test_result["error"])



if __name__ == '__main__':


    main()
