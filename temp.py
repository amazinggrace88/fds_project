import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import itertools
letters = ['a', 'b', 'c', 'd', 'e', 'f']
booleans = [1, 0, 1, 0, 0, 1]
decimals = [0.1, 0.7, 0.4, 0.4, 0.5]

print(list(itertools.chain(letters, booleans, decimals)))  # itertools - list, tuples, iterables 를 연결하는 것
