"""
R fds project in python

- DATA EDA
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# python 에서 fds 코드를 통해서 값을 찾아보고, R 에서의 결과와 성능을 비교해보자.

# step 1. 데이터 불러오기
creditcard = pd.read_csv('creditcard_fds.csv')
print(f'head : \n {creditcard.head()}')  # header 있음, 0, 1, 2, 3, ..
print(f'column names : \n {creditcard.columns}')  # 컬럼 -> Time, V1~V28, Amount, Class(정답)
print(f'describe dataset : \n {creditcard.describe()}')  # 정규화 또는 표준화가 필요, time, class 가 있다.

# step 2. dataset EDA
# NA, 이상치 있는지 알아보기
print(f'null count : \n {creditcard.isnull().sum()}')  # null 없는 것을 발견할 수 있음
# graph - boxplot : Amount
plt.boxplot(creditcard['Amount'])
plt.title('boxplot for Amount')
plt.show()

# graph - boxplot : Time
plt.boxplot(creditcard['Time'])
plt.title('boxplot for Time')
plt.show()  # 정규분포 이루고 있음

# Amount 범위, outlier 알아보기
plt.xlabel('Time by seconds')
plt.ylabel('Creditcard Transaction')
plt.show()
# graph
