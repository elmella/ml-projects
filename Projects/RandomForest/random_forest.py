"""
Random Forest Lab

Matthew Mella
10/24/23
"""
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        return sample[self.column] >= self.value
        

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    n = data.shape[1]
    # create a mask of the data that matches the question
    mask = np.apply_along_axis(question.match,arr=data,axis=1)
    # return the data that matches and doesn't match the question
    return data[mask].reshape(-1,n), data[~mask].reshape(-1,n)

# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    # initiate best gain and question
    best_gain = 0
    best_question = None

    # return none if there is no way to split without violating min_samples_leaf
    if num_rows(data) < 2*min_samples_leaf:
        return best_gain, best_question
    
    if random_subset:
        sqrt_features = int(np.sqrt(data.shape[1] - 1))
        features = np.random.choice(data.shape[1] - 1, sqrt_features, replace=False)
    else:
        features = range(data.shape[1] - 1)
    
    # iterate through each column except the label column and each unique value
    for col in features:
        for val in np.unique(data[:, col]):

            # create a question and partition the data
            Q = Question(col, val, feature_names)
            left, right = partition(data, Q)

            # skip if the question doesn't split the data evenly enough
            if num_rows(left) < min_samples_leaf or num_rows(right) < min_samples_leaf:
                continue
            
            # calculate the info gain and update best_gain and best_question if necessary
            current_gain = info_gain(data, left, right)
            if current_gain > best_gain:
                best_gain, best_question = current_gain, Q

    return best_gain, best_question

# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self, data):
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch


# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    # return a leaf if the data is empty
    if num_rows(data) < min_samples_leaf or current_depth == max_depth:
        return Leaf(data)
    # find the best split
    gain = find_best_split(data, feature_names, min_samples_leaf, random_subset)
    if gain[0] == 0:
        return Leaf(data)
    # partition the data
    left, right = partition(data, gain[1])
    return Decision_Node(gain[1], build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset), build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset))



# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # if the node is a leaf, return the most common label
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)
    # if the sample matches the question, go left
    if my_tree.question.match(sample):
        return predict_tree(sample, my_tree.left)
    # otherwise, go right
    else:
        return predict_tree(sample, my_tree.right)


def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    # get labels
    labels = dataset[:, -1]
    # apply predict_tree to each sample
    predictions = np.apply_along_axis(predict_tree, arr=dataset, axis=1, my_tree = my_tree)
    # return accuracy
    return (predictions == dataset[:, -1]).sum() / labels.shape[0]

# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    # get votes from each tree
    votes = [predict_tree(sample, tree) for tree in forest]
    # return the most common vote
    return max(set(votes), key=votes.count)

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    # get labels
    labels = dataset[:, -1]
    # apply predict_forest to each sample
    predictions = np.apply_along_axis(predict_forest, arr=dataset, axis=1, forest = forest)
    # return accuracy
    return (predictions == dataset[:, -1]).sum() / labels.shape[0]

# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    # prep data
    data = np.loadtxt('parkinsons.csv', delimiter=',')
    data = data[1:]
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str,
                 comments=None)
    # randomly select 130 samples
    data1 = data[np.random.choice(data.shape[0], 130, replace=False)]
    # split into train and test
    train1, test1 = train_test_split(data1, test_size=30, random_state=42)
    # build trees
    forest = [build_tree(train1, features, min_samples_leaf=15, max_depth=4, random_subset=True) for i in range(5)]
    # time and accuracy
    start = time.time()
    acc1 = analyze_forest(test1, forest)
    end = time.time()

    # try sklearn
    clf = RandomForestClassifier(n_estimators=5, min_samples_leaf=15, max_depth=4, random_state=42)
    clf.fit(train1[:, :-1], train1[:, -1])
    start2 = time.time()
    acc2 = clf.score(test1[:, :-1], test1[:, -1])
    end2 = time.time()

    # sklearn on all data
    train_full, test_full = train_test_split(data, test_size=0.2, random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf2.fit(train_full[:, :-1], train_full[:, -1])
    start3 = time.time()
    acc3 = clf2.score(test_full[:, :-1], test_full[:, -1])
    end3 = time.time()

    return (acc1, end-start), (acc2, end2-start2), (acc3, end3-start3)




## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if not hasattr(my_tree, "question"):#isinstance(my_tree, leaf_class):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

if __name__ == "__main__":
    print(prob7())