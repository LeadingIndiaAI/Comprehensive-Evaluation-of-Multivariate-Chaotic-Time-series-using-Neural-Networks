
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset_train = pd.read_csv('rossler.csv')

training_set = dataset_train.iloc[:1100, 0:1].values
test_set=dataset_train.iloc[1100:1160, 0:1].values
test_set=np.reshape(test_set,(1,60))
test_set= np.reshape(test_set, (1, 60, 1))
X_train = []

for i in range(59, 1100):
    X_train.append(training_set[i-59:i+1, 0])
    
X_train = np.array(X_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


model = Sequential()
model.add(LSTM(100, input_shape=(60,1)))
model.add(RepeatVector(60))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mean_squared_error')
# fit model
history=model.fit(X_train, X_train, epochs=60,validation_split=0.03)
pred=model.predict(test_set)
pred=pred[0,:,0]
test_set=test_set[0,:,0]
plt.plot(test_set, color = 'red', label="real" )
plt.plot(pred, color = 'blue', label="predicted")
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()