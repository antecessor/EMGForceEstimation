import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import SGD, RMSprop
from keras.layers.advanced_activations import LeakyReLU

import numpy as np

from Utils.SignalUtils import getSignal, calculateSD, prepareFiringSignal, windowingSig, trainTestSplit, \
    calculateCdr, calculateNormalizedOnCorr

sig, force, fs, firings = getSignal(1, 30, "../signals/")
sdSig = calculateSD(sig)
# sdSig = calculateICA(sdSig, 64)
sizeInputSignal = sdSig.shape[1]
preparedFirings = prepareFiringSignal(firings[0], sizeInputSignal)
signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=8)  # should be divideable to 4
labelWindow = calculateCdr(labelWindow)
signalWindow = calculateNormalizedOnCorr(signalWindow)
x_train, x_test, y_train, y_test = trainTestSplit(signalWindow, labelWindow, 0.7)
# convert to one-hot vector
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

signal_size_row = x_train[0].shape[0]
signal_size_col = x_train[0].shape[1]
x_train = np.reshape(x_train, [-1, signal_size_row, signal_size_col])
x_test = np.reshape(x_test, [-1, signal_size_row, signal_size_col])
input_shape = (signal_size_row, signal_size_col)

model = Sequential()
model.add(Conv1D(16, 5, padding='same', input_shape=input_shape))
model.add(LeakyReLU())
model.add(Conv1D(8, 10, padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling1D(10))
model.add(Dropout(0.25))
model.add(Conv1D(16, 5, padding='same'))
model.add(LeakyReLU())
model.add(Conv1D(16, 10, padding='same'))
model.add(LeakyReLU())
model.add(MaxPooling1D(5))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.mean_squared_error, optimizer=RMSprop(), metrics=[metrics.MSE])

batch_size = 64
epochs = 100

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size)

acc = model.evaluate(x_test,
                     y_test,
                     batch_size=batch_size,
                     verbose=0)

out = np.round(model.predict(x_test, batch_size=batch_size))

print("\nTest accuracy: %.1f%%" % (100 * acc))
