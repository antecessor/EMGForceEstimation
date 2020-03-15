import numpy as np

import tensorflow as tf
from keras import metrics
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Lambda, BatchNormalization, GRU, LSTM, ConvLSTM2D, Reshape, Bidirectional, Flatten, MaxPooling3D, TimeDistributed, MaxPooling2D, SimpleRNN, \
    LocallyConnected2D, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical
from pyCompare import blandAltman
from tensorflow import keras
import matplotlib.pyplot as plt

from Utils.SignalUtils import calculateSD, prepareFiringSignal, windowingSig, \
    calculateCdr, trainTestSplit, calculateNormalizedOnCorr, getSignal2, extractPhaseSpace, calculateSTFT, extractSoundFeatures, calculateICA, butter_bandpass_filter, \
    calculateFFTOnWindows, calculateForceAverage
from sklearn.metrics import r2_score

maxFeatures = 256
# sig, force, fs, firings = getSignal(1, 30, "./signals/")
trainSigs = []
trainTarget = []
for i in range(10, 17):
    sig, force, fs, firings = getSignal2("002041436" + str(i), filePath="../signals/Hamid/00MaHaLI1002041436/")
    trainSigs.append(sig)
    trainTarget.append(force)
sig2, force2, fs2, firings2 = getSignal2("00204143605", filePath="../signals/Hamid/00MaHaLI1002041436/")


# sig2, force2, fs2, firings2 = getSignal(2, 30, "./signals/")

def prepareDataWithForce(sig, force):
    sdSig = calculateSD(sig)
    sdSig = butter_bandpass_filter(sdSig, 20, 450, 2048)
    # sdSig = calculateICA(sdSig, 64)


    signalWindow, labelWindow = windowingSig(sdSig, force, windowSize=maxFeatures)  # should be divideable to 4
    signalWindow = calculateSTFT(signalWindow)

    # signalWindow = calculateFFTOnWindows(signalWindow)
    labelWindow = calculateForceAverage(labelWindow)
    # signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
    # signalWindow, labelWindow = calculateICA(signalWindow, labelWindow, 4)

    # signalWindow = extractPhaseSpace(signalWindow)
    # signalWindow = extractSoundFeatures(signalWindow)

    return signalWindow, labelWindow



def prepareData(sig, firings):
    sdSig = calculateSD(sig)
    sdSig = butter_bandpass_filter(sdSig, 20, 450, 2048)
    # sdSig = calculateICA(sdSig, 64)

    sizeInputSignal = sdSig.shape[1]
    preparedFirings = prepareFiringSignal(firings[0], sizeInputSignal, numSignals=12)
    signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=maxFeatures)  # should be divideable to 4
    signalWindow = calculateSTFT(signalWindow)

    # signalWindow = calculateFFTOnWindows(signalWindow)
    labelWindow = calculateCdr(labelWindow)
    # signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
    # signalWindow, labelWindow = calculateICA(signalWindow, labelWindow, 4)

    # signalWindow = extractPhaseSpace(signalWindow)
    # signalWindow = extractSoundFeatures(signalWindow)

    return signalWindow, labelWindow


signalWindow, labelWindow = prepareDataWithForce(trainSigs[0], trainTarget[0])
for index in range(1, len(trainSigs)):
    sig, label = prepareDataWithForce(trainSigs[index], trainTarget[index])
    signalWindow = np.append(signalWindow, sig, axis=0)
    labelWindow = np.append(labelWindow, label, axis=0)

signalWindow2, labelWindow2 = prepareDataWithForce(sig2, force2)

# labelWindow[np.where(labelWindow >= 1)] = 1
# labelWindow2[np.where(labelWindow2 >= 1)] = 1
# num_classes = len(np.unique(labelWindow))
# labelWindow2[np.where(labelWindow2 > num_classes)] = num_classes - 1
labelWindow=labelWindow/np.max(labelWindow)
labelWindow2=labelWindow2/np.max(labelWindow2)

X_train, X_test, Y_train, Y_test = trainTestSplit(signalWindow, labelWindow, 0.75)
X_train2, X_test2, Y_train2, Y_test2 = trainTestSplit(signalWindow2, labelWindow2, 0.75)

# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)
# Y_train2 = to_categorical(Y_train2)
# Y_test2 = to_categorical(Y_test2)

signal_size_row = X_train[0].shape[0]
signal_size_col = X_train[0].shape[1]
components = X_train[0].shape[2]
# X_train = np.reshape(X_train, [-1, signal_size_row, signal_size_col, 1])
# X_test = np.reshape(X_test, [-1, signal_size_row, signal_size_col, 1])
# X_train2 = np.reshape(X_train2, [-1, signal_size_row, signal_size_col, 1])
# X_test2 = np.reshape(X_test2, [-1, signal_size_row, signal_size_col, 1])

input_shape = (signal_size_row, signal_size_col, components)

lstm_out = 5


def counting(args):
    input = args
    var = tf.reduce_sum(input, axis=1, keepdims=False)
    return var



model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(32, 3, activation="relu"))

# model.add(MaxPooling2D(pool_size=2, padding='same'))
# model.add(Lambda(counting,
#                  name='z'))
model.add(Flatten())
model.add(Dense(64, activity_regularizer=l2(0.001)))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Dense(1,activation="linear"))
model.compile(loss=keras.losses.mean_absolute_error, optimizer='nadam',
              metrics=[metrics.RootMeanSquaredError(), metrics.MAE])
print(model.summary())

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test2.shape)

batch_size = 16
epochs = 150
reduce_lr_acc = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=epochs / 10, verbose=1, min_delta=1e-4, mode='max')
model.fit(X_train, Y_train,
          epochs=epochs,
          batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[reduce_lr_acc])

# acc = model.evaluate(X_test,
#                      Y_test,
#                      batch_size=batch_size,
#                      verbose=0)

out = np.round(model.predict(X_train2, batch_size=batch_size))
predicted2 = out.ravel()

out = np.round(model.predict(X_test, batch_size=batch_size))
predicted = out.ravel()

# sns.set(color_codes=True)
# ax = sns.regplot(x=Y_test2, y=predicted2, color="g")
# Y_train2 = np.argmax(Y_train2, axis=1).ravel()
r = r2_score(Y_train2, predicted2)
print("R2:{0}".format(r))
plt.plot(Y_train2)
plt.plot(predicted2)
plt.legend(['Target', 'Estimated'], loc='upper left')
plt.show()

# blandAltman(Y_test, predicted,
#             savePath='SavedFigureAltman.svg',
#             figureFormat='svg')

pass
