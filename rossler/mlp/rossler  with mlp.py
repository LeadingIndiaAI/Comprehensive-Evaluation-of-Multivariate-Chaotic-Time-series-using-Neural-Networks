

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

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', learning_rate='constant', random_state=0, max_iter=1000)
#train neural network
history=mlp.fit(X_train, y_train)
# calculate prediction of testdata


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

predicted =mlp.predict(X_test)


# Visualising the results
plt.plot(y_test, color = 'red', label="real" )
plt.plot(predicted, color = 'blue', label="predicted")
plt.legend()
plt.show()


