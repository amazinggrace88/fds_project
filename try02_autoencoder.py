"""
R fds project in python

- Autoencoder
"""
from collections import Counter
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from functools import partial
import tensorflow as tf
from tensorflow.keras import layers

  
# Mini_batch
def suffle_batch(features, labels, batch_size):
    rnd_idx = np.random.permutation(len(features))  # features 수 만큼의 순열을 만들어준다.
    n_batches = len(features) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_x, batch_y = features[batch_idx], labels[batch_idx]
        yield batch_x, batch_y  # generator object 생성 -> yield 에서 값을 발생시킨다.(generate)

#

if __name__ == '__main__':
    # bring the data!
    creditcard = pd.read_csv('creditcard_fds.csv')
    """
    What is Autoencoder ? 
    - 입력에서 계산된 출력이 입력과 비슷해지도록 비지도 학습으로 훈련되는 신경망
    - 입력값을 기반으로 여기서 특징을 뽑아내고, 뽑아낸 특징으로 다시 원본을 재생하는 네트워크 (출처: https://bcho.tistory.com/1326 [조대협의 블로그])
    - R 에서는 Undercomplete Autoencoder (히든 레이어의 뉴런(노드, 유닛)이 입력층보다 작으므로 입력이 저차원으로 표현되는 Autoencoder) 을 사용하였다.
    (undercomplete 오토인코더는 저차원을 가지는 히든 레이어에 의해 입력을 그대로 출력으로 복사할 수 없기 때문에, 출력이 입력과 같은 것을 출력하기 위해 학습해야 한다. 
    이러한 학습을 통해 undercomplete 오토인코더는 입력 데이터에서 가장 중요한 특성(feature)을 학습하도록 만든다.)
    
    R / Python 에서의 같은 알고리즘을 구현

    - 참고
    link https://d2.naver.com/news/0956269 (Naver Tech Talk: 오토인코더의 모든 것 (2017년 11월)) 
    link https://excelsior-cjh.tistory.com/187
    link https://github.com/ExcelsiorCJH/Hands-On-ML/blob/master/Chap15-Autoencoders/Chap15-Autoencoders.ipynb (핸즈온머신러닝)    
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

    # 3. autoencoder 적용
    """
    R - autoencoder 와 같은 구조로 적용 
    R - karas 패키지의 stacked autoencoder 를 사용했다.
    Python - tf.keras 의 stacked autoencoder 를 사용하려 한다. (옵션은 최대한 R 의 autoencoder 를 따름)
    [ R code ]
    model %>% 
      layer_dense(units = 15, activation = "tanh", input_shape = ncol(x_train)) %>% 
      layer_dense(units = 10, activation = "tanh") %>% 
      layer_dense(units = 15, activation = "tanh") %>% 
      layer_dense(units = ncol(x_train))
    - 다중 퍼셉트론 레이어 : unit 15, unit 10, unit 15 의 나비모양으로 이루어진 뉴런들을 나열한다.
    - 4개의 layer 를 입력층에는 x_train 의 갯수만큼, 1층, 2층, 3층 순으로 뉴런 갯수를 정하여 나비모양(출력층 : x_train 의 갯수)으로 만든다.
    - 활성화 함수 : hyperbolic tan(x)
    """
    # Training Parameters
    """
    train parameter - learning rate / l2_reg 는 임의지정
    """
    learning_rate = 0.01  # 임의 지정
    l2_reg = 0.0001  # 임의 지정
    n_epochs = 100
    batch_size = 32
    n_batches = len(X_train) // batch_size

    # Layer Parameters
    print('X_train.shape = ', X_train.shape)  # (770, 30)
    n_inputs = 30  # X_train.shape[1]
    n_hidden1 = 15
    n_hidden2 = 10
    n_hidden3 = 15
    n_outputs = n_inputs

    # set the layers using partial
    he_init = tf.keras.initializers.he_normal(seed=106)  # He 초기화 한 initializer
    l2_regularizer = tf.keras.regularizers.l2(l=0.01)  # L2 규제 : overfitting 억제
    dense_layer = partial(tf.layers.dense, activation=tf.nn.tanh, kernel_regularizer=l2_regularizer)
    # tf.nn.tanh 를 사용 (R - tanh 함수 사용하였으므로)
    # 성능 개선에서는 tf.nn.elu 사용 예정 - ReLU 의 특성 공유, gradient 가 죽지 않는다는 장점 가짐

    # placeholder 를 통해 변수 정의
    inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
    print(type(inputs))  # <class 'tensorflow.python.framework.ops.Tensor'>

    # stacked autoencoder
    hidden1 = layers.Dense(inputs, n_hidden1)
    # hidden2 = layers.Dense(hidden1, n_hidden2)
    # hidden3 = layers.Dense(hidden2, n_hidden3)
    # outputs = layers.Dense(hidden3, n_outputs, activation=None)

    # loss 질문하기
    # reconstruction_loss = tf.reduce_mean(tf.square(outputs - inputs))  # MSE 인가 ?
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # 이해 안됨.. reg_loss ?
    # loss = tf.add_n([reconstruction_loss] + reg_losses)  # add_n : element 끼리 더함
    #
    # # optimizer
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # R 과 같은 AdamOptimizer 사용
    #
    # # saver
    # saver = tf.train.Saver(max_to_keep=1)  # saver 에 1개의 model 만을 저장한다.
    """
    다음 목표 : 
    https://github.com/NVIDIA/DeepRecommender - autoencoder 를 이용한 검색 엔진도 구현해보자
    """










