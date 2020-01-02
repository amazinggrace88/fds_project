"""
R fds project in python

- Isolation Forest
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix
import itertools
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # bring the data!
    creditcard = pd.read_csv('creditcard_fds.csv')

    """
    what is Isolation Forest?
    
    - The IsolationForest ‘isolates’ observations by randomly selecting a feature 
    and then randomly selecting a split value between the maximum and minimum values of the selected feature.
    
    - 의사결정나무를 이용한 이상탐지 수행방법
    - 이상치 기준을 모델로 생성하는 방법
    """

    # 정규화 - data scaling
    # dataset -> time, class 뺀 정규화용 DataFrame 만들기
    print(creditcard.columns)
    creditcard_scaling = creditcard.iloc[:, 1:30]  # V1~V28, Amount 까지의 creditcard set 만듬
    print(creditcard_scaling.columns)

    # MinMaxScaler
    scaler = MinMaxScaler()
    creditcard_scaling = scaler.fit_transform(creditcard_scaling)
    print('표준화된 creditcard : ', creditcard_scaling)  # array 형태
    creditcard_scaling = DataFrame(creditcard_scaling)
    print('표준화된 creditcard - df 형태: \n', creditcard_scaling)  # df 형태
    creditcard_scaling.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                  'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                  'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    print('표준화된 creditcard - df & column name : \n', creditcard_scaling)  # df 형태가 + column name
    print('표준화 creditcard된 컬럼들의 기술 통계량 : \n', creditcard_scaling.describe())  # min 0, max 1

    # creditcard scaling + Time + Class
    time_column = creditcard['Time']
    class_column = creditcard['Class']
    creditcard_scaling['Time'] = time_column
    creditcard_scaling['Class'] = class_column
    print('표준화된 creditcard + time + class : \n', creditcard_scaling)  # time, class 마지막에 배열되었다.


    # train data / test data slicing
    # 1. 시간을 기준으로 나누어주어 과거의 데이터를 통해 미래의 데이터를 유추할 수 있도록 함.
    print('Time\'s max :', max(creditcard_scaling.Time))  # 172792.0 (열의 갯수와 max 가 다르므로 중복값이 있음을 알 수 있다.)
    print('Time\'s len :', len(creditcard_scaling.Time))  # 284807 (열의 갯수)
    print('Time flow : \n', creditcard_scaling.Time[:10])  # 시간의 순서대로 흐르지만, 중복이 있음을 알 수 있다.
    df_train = creditcard_scaling.iloc[:200000]
    print('df_train shape : ', df_train.shape)  # (200000, 31)
    df_test = creditcard_scaling.iloc[200000:]
    print('df_test shape : ', df_test.shape)  # (84807, 31)

    # 2. undersampling
    # df_train - X_train, y_train / df_test - X_test, y_test
    X_train = df_train.drop(['Class'], axis=1)
    y_train = df_train['Class']
    X_test = df_test.drop(['Class'], axis=1)
    y_test = df_test['Class']
    # RandomUnderSampler 를 통해 train data 만 undersampling 을 수행
    sampler = RandomUnderSampler(random_state=0)
    X_train, y_train = sampler.fit_sample(X_train, y_train)
    print('Class after Undersampling : \n', Counter(y_train))  # Counter({0: 385, 1: 385}) 5:5 비율

    # 3. Isolation Forest 적용
    # 파라미터 : n_estimators - 노드 수, contamination - 이상치 비율
    # (https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html 참고)
    clf = IsolationForest(n_estimators=100, random_state=42)
    clf.fit(X_train)
    predict_outlier = clf.predict(X_train)
    predict_outlier = pd.DataFrame(predict_outlier).replace({1:0, -1:1})
    # cf. -1 : 이상치, 1이 정상치로 분류되어 1이 이상치, 0이 정상치인 것으로 다시 바꿔주었다.

    # 4. 예측 결과 분석
    # recall : 실제 부정거래를 부정거래로 예측하는 비율
    print('Accuracy : \n', accuracy_score(predict_outlier, y_train))
    print('Confusion Matrix : \n', confusion_matrix(predict_outlier, y_train))
    print('Classification Report : \n', classification_report(predict_outlier, y_train))

    # 5. 시각화
    # plot 2d - 차원 축소
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=predict_outlier, cmap='Paired', s=40, edgecolors='white')
    # plt.title("Isolation Forest")
    # plt.show()

    # plot 3d - 차원 축소
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], X_train.iloc[:, 2], c=predict_outlier)
    # ax.set_xlabel('pcomp 1')
    # ax.set_ylabel('pcomp 2')
    # ax.set_zlabel('pcomp 3')
    # plt.show()