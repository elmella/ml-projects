"""Volume 3: Naive Bayes Classifiers."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import math

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Get the number of ham and spam messages
        hamMask = y == 'ham'
        spamMask = y == 'spam'

        # Calculate the probability of ham and spam
        self.p_ham = np.sum(hamMask) / len(y)
        self.p_spam = np.sum(spamMask) / len(y)

        # Get the ham and spam messages
        ham = X[hamMask]
        spam = X[spamMask]

        # Get the words from the ham and spam messages
        hamWords = []
        spamWords = []
        hamWords = [word for sublist in ham.str.split() for word in sublist]
        spamWords = [word for sublist in spam.str.split() for word in sublist]

        # Get the unique words from the ham and spam messages
        self.trainingWords = list(set(hamWords + spamWords))

        # Get the length of the ham and spam messages
        lenHamWords = len(hamWords)
        lenSpamWords = len(spamWords)
        
        # Calculate the probability of each word in the ham and spam messages
        ham_probs = Counter(hamWords)
        spam_probs = Counter(spamWords)

        # Add 1 to each word count and divide by the total number of words + 2
        for word in ham_probs.keys():
            ham_probs[word] = (ham_probs[word] + 1) / (lenHamWords + 2)
            # Add words that are in the spam messages but not in the ham messages
            if word not in spam_probs.keys():
                spam_probs[word] = 1 / (lenSpamWords + 2)

        for word in spam_probs.keys():
            spam_probs[word] = (spam_probs[word] + 1) / (lenSpamWords + 2)
            # Add words that are in the ham messages but not in the spam messages
            if word not in ham_probs.keys():
                ham_probs[word] = 1 / (lenHamWords + 2)

        # Set the ham and spam probabilities as attributes
        self.ham_probs = ham_probs
        self.spam_probs = spam_probs

        return self

    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Split the messages into words
        messages = X.str.split()

        # Create lists to store the probabilities
        ham_probs = []
        spam_probs = []

        # Calculate the log probabilities for each message
        p_ham = np.log(self.p_ham)
        p_spam = np.log(self.p_spam)

        # Define log(0.5)
        logPointFive = np.log(0.5)

        # Calculate the log probabilities for each message
        for message in messages:
            # Set the initial probabilities to the log of the probability of ham and spam
            ham_prob = p_ham
            spam_prob = p_spam
            for word in message:
                # Add the log probability of each word in the message
                if word not in self.trainingWords:
                    ham_prob += logPointFive
                    spam_prob += logPointFive
                else:
                    ham_prob += np.log(self.ham_probs[word])
                    spam_prob += np.log(self.spam_probs[word])
            # Append the probabilities to the lists
            ham_probs.append(ham_prob)
            spam_probs.append(spam_prob)

        # Convert the lists to arrays
        ham_probs = np.array(ham_probs)
        spam_probs = np.array(spam_probs)

        # Stack the arrays and return them
        return np.vstack((ham_probs, spam_probs)).T

    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get the probabilities for each message
        probs = self.predict_proba(X)
        labels = []
        # Compare the probabilities and append the label to the list. Equal probabilities are classified as ham.
        for prob in probs:
            if prob[0] < prob[1]:
                labels.append('spam')
            else:
                labels.append('ham')

        return np.array(labels)

def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    df = pd.read_csv('sms_spam_collection.csv')
    X, y = df['Message'], df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    nb = NaiveBayesFilter()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    correct_spam = np.sum((y_pred == 'spam') & (y_test == 'spam'))
    incorrect_ham = np.sum((y_pred == 'spam') & (y_test == 'ham'))
    return correct_spam / np.sum(y_test == 'spam'), incorrect_ham / np.sum(y_test == 'ham')

# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        
        # Get the number of ham and spam messages
        hamMask = y == 'ham'
        spamMask = y == 'spam'

        # Calculate the probability of ham and spam
        self.p_ham = np.sum(hamMask) / len(y)
        self.p_spam = np.sum(spamMask) / len(y)

        # Get the ham and spam messages
        ham = X[hamMask]
        spam = X[spamMask]

        # Get the words from the ham and spam messages
        hamWords = []
        spamWords = []
        hamWords = [word for sublist in ham.str.split() for word in sublist]
        spamWords = [word for sublist in spam.str.split() for word in sublist]

        self.N_ham = len(hamWords)
        self.N_spam = len(spamWords)
        # Get the unique words from the ham and spam messages
        self.trainingWords = list(set(hamWords + spamWords))

        # Get the length of the ham and spam messages
        
        # Calculate the rate of each word in the ham and spam messages
        ham_rates = Counter(hamWords)
        spam_rates = Counter(spamWords)

        # Add 1 to each word count and divide by the total number of words + 2
        for word in ham_rates.keys():
            ham_rates[word] = (ham_rates[word] + 1) / (self.N_ham + 2)
            # Add words that are in the spam messages but not in the ham messages
            if word not in spam_rates.keys():
                spam_rates[word] = 1 / (self.N_spam + 2)

        for word in spam_rates.keys():
            spam_rates[word] = (spam_rates[word] + 1) / (self.N_spam + 2)
            # Add words that are in the ham messages but not in the spam messages
            if word not in ham_rates.keys():
                ham_rates[word] = 1 / (self.N_ham + 2)

        # Set the ham and spam rates to the calculated rates
        self.ham_rates = ham_rates
        self.spam_rates = spam_rates

        return self

    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Split the messages into words
        messages = X.str.split()

        ham_probs = []
        spam_probs = []

        # Get the log of the probability of ham and spam
        p_ham = np.log(self.p_ham)
        p_spam = np.log(self.p_spam)

        # Set the default rate of ham and spam
        hamNotFound = 1 / (self.N_ham + 2)
        spamNotFound = 1 / (self.N_spam + 2)
        
        # Iterate through each message
        for message in messages:

            # Get the length of the message
            n = len(message)
            
            # set the probability of ham and spam to the log of the probability of ham and spam
            ham_prob = p_ham
            spam_prob = p_spam

            # Iterate through each unique word in the message
            for word in np.unique(message):
                
                # If the word is not in the training set, set the rate to the default rate
                if word not in self.trainingWords:
                    ham_r = hamNotFound
                    spam_r = spamNotFound
                # Otherwise, set the rate to the log of the rate of the word
                else:
                    ham_r = self.ham_rates[word]
                    spam_r = self.spam_rates[word]
                
                # get the count of the word in the message
                ni = message.count(word)

                # Calculate the probability of the word in the message
                ham_eq = ((ham_r * n)**ni * np.exp(-ham_r * n)) / math.factorial(ni)
                spam_eq = ((spam_r * n)**ni * np.exp(-spam_r * n)) / math.factorial(ni)

                # Add the log of the probability of the word to the log of the probability of the message
                ham_prob += np.log(ham_eq)
                spam_prob += np.log(spam_eq)

            # append the log of the probability of the message to the ham and spam probabilities
            ham_probs.append(ham_prob)
            spam_probs.append(spam_prob)

        # Convert the ham and spam probabilities to arrays
        ham_probs = np.array(ham_probs)
        spam_probs = np.array(spam_probs)

        # Return the ham and spam probabilities as a stacked array
        return np.vstack((ham_probs, spam_probs)).T


    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get the probabilities of ham and spam
        probs = self.predict_proba(X)
        labels = []
        # Iterate through each probability
        for prob in probs:
            # If the probability of spam is greater than the probability of ham, append spam
            if prob[0] < prob[1]:
                labels.append('spam')
            # Otherwise, append ham
            else:
                labels.append('ham')

        # Convert the labels to an array
        return np.array(labels)




def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # Read in the data
    df = pd.read_csv('sms_spam_collection.csv')

    # Split the data into training and testing sets
    X, y = df['Message'], df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create a NaiveBayesFilter and fit it to the training data
    pb = PoissonBayesFilter()
    pb.fit(X_train, y_train)

    # Predict the labels of the test data
    y_pred = pb.predict(X_test)

    # Calculate the proportion of spam messages correctly identified and the proportion of ham messages incorrectly identified
    correct_spam = np.sum((y_pred == 'spam') & (y_test == 'spam'))
    incorrect_ham = np.sum((y_pred == 'spam') & (y_test == 'ham'))

    # Return the proportions
    return correct_spam / np.sum(y_test == 'spam'), incorrect_ham / np.sum(y_test == 'ham')

    
# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # Create a CountVectorizer
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)

    # Create a MultinomialNB classifier
    clf = MultinomialNB()
    clf.fit(train_counts, y_train)

    # Transform the test data
    test_counts = vectorizer.transform(X_test)

    # Return the predictions
    return clf.predict(test_counts)