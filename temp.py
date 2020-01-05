import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import itertools

iterable1 = 'ABCD'
iterable2 = 'xy'
iterable3 = '1234'

for i in iterable1:
    for j in iterable2:
        for k in iterable3:
            print(i+j+k)

print()

iterable1 = 'ABCD'
iterable2 = 'xy'
iterable3 = '1234'

dot = itertools.product(iterable1, iterable2, iterable3)
print(dot)  # itertools.product object
print(list(dot))  # list 로 만들어서 리턴할 수 있다.

