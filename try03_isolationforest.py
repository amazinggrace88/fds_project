"""
R fds project in python

- Isolation Forest
"""
import pandas as pd
from pandas import DataFrame
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix


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
    print('< 정규화 - data scaling >')
    # dataset -> time, class 뺀 정규화용 DataFrame 만들기
    print(creditcard.columns)
    creditcard_scaling = creditcard.iloc[:, 1:30]  # V1~V28, Amount 까지의 creditcard set 만듬
    print(creditcard_scaling.columns)

    # MinMaxScaler 생성
    scaler = MinMaxScaler()
    creditcard_scaling = scaler.fit_transform(creditcard_scaling)
    print('표준화된 creditcard : ', creditcard_scaling)  # array 형태
    creditcard_scaling = DataFrame(creditcard_scaling)
    print('표준화된 creditcard - df 형태: \n', creditcard_scaling)  # df 형태
    creditcard_scaling.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                  'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                  'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    print('표준화된 creditcard - df & column name : \n', creditcard_scaling)  # df 형태가 + column name
    print('표준화된 컬럼들의 기술 통계량 : \n', creditcard_scaling.describe())  # min 0, max 1
    print('================================================')
    print()

    # creditcard scaling + Time + Class 생성
    print('< creditcard scaling + Time + Class 생성 >')
    time_column = creditcard['Time']
    class_column = creditcard['Class']
    creditcard_scaling['Time'] = time_column
    creditcard_scaling['Class'] = class_column
    print('표준화된 creditcard + time + class : \n', creditcard_scaling)  # time, class 마지막에 배열되었다.
    print('================================================')
    print()

    # train data / test data slicing
    print('< train data / test data slicing >')
    # 1. 시간을 기준으로 나누어주어 과거의 데이터를 통해 미래의 데이터를 유추할 수 있도록 함.
    print('Time\'s max :', max(creditcard_scaling.Time))  # 172792.0 (열의 갯수와 max 가 다르므로 중복값이 있음을 알 수 있다.)
    print('Time\'s len :', len(creditcard_scaling.Time))  # 284807 (열의 갯수)
    print('Time flow : \n', creditcard_scaling.Time[:10])  # 시간의 순서대로 흐르지만, 중복이 있음을 알 수 있다.
    df_train = creditcard_scaling.iloc[:200000]
    print('df_train shape : ', df_train.shape)  # (200000, 31)
    df_test = creditcard_scaling.iloc[200000:]
    print('df_test shape : ', df_test.shape)  # (84807, 31)
    print('================================================')
    print()

    # 2. undersampling
    print('< UnderSampling >')
    # df_train - X_train, y_train / df_test - X_test, y_test
    X_train = df_train.drop(['Class'], axis=1)
    y_train = df_train['Class']
    X_test = df_test.drop(['Class'], axis=1)
    y_test = df_test['Class']
    # RandomUnderSampler 를 통해 train data 만 undersampling 을 수행
    sampling_strategy = {0: 578, 1: 192}
    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
    X_train, y_train = sampler.fit_sample(X_train, y_train)
    print('Class after UnderSampling : \n', Counter(y_train))  # Counter({0: 578, 1: 192}) - 2 : 1 비율
    print('================================================')
    print()

    # 3. Isolation Forest 적용
    print('< Isolation Forest 적용 >')
    # 파라미터 : n_estimators - 노드 수, contamination - 이상치 비율
    # (https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html 참고)
    clf = IsolationForest(n_estimators=100, random_state=42)
    clf.fit(X_train)
    predict_outlier = clf.predict(X_train)
    predict_outlier = pd.DataFrame(predict_outlier).replace({1:0, -1:1})
    print('predict_outlier = \n', predict_outlier)
    # cf. -1 : 이상치, 1이 정상치로 분류되어 1이 이상치, 0이 정상치인 것으로 다시 바꿔주었다.
    print('================================================')
    print()


    # 4. 예측 결과 분석
    print('< 예측 결과 분석 >')
    # recall : 실제 부정거래를 부정거래로 예측하는 비율
    print('Accuracy : \n', accuracy_score(predict_outlier, y_train))
    print('Confusion Matrix : \n', confusion_matrix(predict_outlier, y_train))
    print('Classification Report : \n', classification_report(predict_outlier, y_train))
    # recall : 1 을 1이라고 예측하는 비율이 0.95 로 높아졌다 !
    print('================================================')
    print()

    # 5. 시각화
    # plot 2d - 차원 축소
    print(type(X_train.iloc[:, 0]))  # Series
    x_0 = X_train.iloc[:, 0]
    x_0 = x_0.to_frame()
    print(type(x_0))  # DataFrame
    print(type(X_train.iloc[:, 1]))  # Series
    x_1 = X_train.iloc[:, 1]
    x_1 = x_1.to_frame()
    print(type(x_1))  # DataFrame
    print(type(predict_outlier))  # DataFrame

    plt.scatter(x_0, x_1, c=predict_outlier, cmap='Paired', edgecolors='white')
    plt.title("Isolation Forest")
    plt.show()
    # paired 된 1 이 너무 많다는 것을 알 수 있다. 역시 과도한 차원축소는 데이터의 왜곡을 가져온다.
    print('================================================')
    print('END')
