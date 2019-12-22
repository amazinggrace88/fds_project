"""
R fds project in python

- Isolation Forest
"""
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# bring the data!
creditcard = pd.read_csv('creditcard_fds.csv')

"""
what is Isolation Forest?

- The IsolationForest ‘isolates’ observations by randomly selecting a feature 
and then randomly selecting a split value between the maximum and minimum values of the selected feature.

- 의사결정나무를 이용한 이상탐지 수행방법
- 이상치 기준을 모델로 생성하는 방법
"""

# 시간을 기준으로 나누어주어 과거의 데이터를 통해 미래의 데이터를 유추할 수 있도록 한다.
# row selection(filtering) 사용
print(f'Time\'s max : {max(creditcard.Time)}')  # 172792.0
print(f'Time\'s len : {len(creditcard.Time)}')  # 284807  -> 시간이 순서대로 정렬되어 있으므로, 200000 을 기준으로 train/test set 생성
df_train = creditcard[len(creditcard.Time) <= 200000]  # train_df
df_train = creditcard
df_test = creditcard[len(creditcard.Time) > 200000]  # test_df
print(f'df_train.head : \n {df_train.head()}')
print(f'df_test.head : \n {df_test.head()}')
