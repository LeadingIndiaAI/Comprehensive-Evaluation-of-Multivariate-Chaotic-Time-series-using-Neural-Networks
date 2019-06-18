

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
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = [])

# Fitting the ANN to the Training set
history =classifier.fit(X_train, y_train, batch_size =8, nb_epoch = 60,validation_split=0.03)

dataset_test=dataset.iloc[1100:1110, 0:1].values
y_test=dataset.iloc[1100:1110, 0:1].values
dataset_test=pd.DataFrame(dataset_test)
dataset_train=pd.DataFrame(training_set)


dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 20:].values


inputs = inputs.reshape(-1,1)

X_test = []
for i in range(20,30):
    X_test.append(inputs[i-20:i, 0])
X_test = np.array(X_test)

predicted = classifier.predict(X_test)


# Visualising the results
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
