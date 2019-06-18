import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('rossler.csv')

training_set = dataset.iloc[:1100, 0:1].values
X_train=[]
y_train=[]
for i in range(20,1090):
    X_train.append(training_set[i-20:i,0])
    y_train.append(training_set[i,0])
X_train=np.asarray(X_train)
y_train=np.asarray(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
model = Sequential()
model.add(Convolution1D(filters=16, kernel_size=3, activation='relu', input_shape=(20,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history=model.fit(X_train,y_train,epochs=60,batch_size=8,validation_split=0.02)

test_set=dataset.iloc[1100:1130, 0:1].values
X_test=[]
y_test=[]
for i in range(20,30):
    X_test.append(test_set[i-20:i,0])
    y_test.append(test_set[i,0])
X_test=np.asarray(X_test)
y_test=np.asarray(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted=model.predict(X_test)
y_test=np.reshape(y_test,(10,1))
predicted=np.reshape(predicted,(10,1))
plt.plot(y_test, color = 'red', label="real" )
plt.plot(predicted, color = 'blue', label="predicted")
plt.legend()
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()