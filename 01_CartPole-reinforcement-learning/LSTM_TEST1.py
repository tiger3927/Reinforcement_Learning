"""
@author:tiger
@file:test4.py
@time:2022/07/19
"""
import numpy as np

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 异常处理
        print(e)

look_back = 40
dataset = np.cos(np.arange(1000) * (20 * np.pi / 1000))
# 归一化，y值域为(0,1)
dataset = (dataset + 1) / 2.

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# 建立模型：有状态
batch_size = 1
model3 = Sequential()
model3.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model3.add(Dropout(0.3))
model3.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model3.add(Dropout(0.3))
model3.add(Dense(1))
model3.compile(loss='mean_squared_error', optimizer='adam')
for i in range(1):
    print(i)
    model3.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    print(i)
    model3.reset_states()

# 预测
x = np.vstack((trainX[-1][1:], (trainY[-1])))
preds = []
pred_num = 500
for i in np.arange(pred_num):
    pred = model3.predict(x.reshape((1, -1, 1)), batch_size=batch_size)
    preds.append(pred.squeeze())
    x = np.vstack((x[1:], pred))

# print(preds[:20])
# print(np.array(preds).shape)
plt.figure(figsize=(12, 5))
plt.plot(np.arange(pred_num), np.array(preds), 'r', label='predctions')
cos_y = (np.cos(np.arange(pred_num) * (20 * np.pi / 1000)) + 1) / 2.
plt.plot(np.arange(pred_num), cos_y, label='origin')
plt.legend()
plt.show()