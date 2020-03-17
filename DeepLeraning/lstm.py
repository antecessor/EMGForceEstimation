import numpy as np

import tensorflow as tf
from keras import metrics
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Lambda, BatchNormalization, Dropout, LeakyReLU, GRU, LSTM, Flatten
from keras.regularizers import l2
from keras.utils import to_categorical
from pyCompare import blandAltman
from tensorflow import keras
import matplotlib.pyplot as plt
from Utils.SignalUtils import calculateSD, prepareFiringSignal, windowingSig, \
    calculateCdr, trainTestSplit, calculateNormalizedOnCorr, getSignal2, extractPhaseSpace, butter_bandpass_filter, calculateForceAverage, getSignal, getSignal3
from sklearn.metrics import r2_score
import pandas as pd

maxFeatures = 64
# sig, force, fs, firings = getSignal(1, 30, "../signals/")
trainSigs = []
trainTargets = []
for i in range(1, 3):
    sig, force, fs, firings = getSignal3(str(i))
    trainSigs.append(sig)
    trainTargets.append(force)
sig2, force2, fs2, firings2 = getSignal3(str(3))


# sig2, force2, fs2, firings2 = getSignal(2, 30, "./signals/")
def prepareDataWithForce(sig, force):
    sdSig = calculateSD(sig)
    sdSig = butter_bandpass_filter(sdSig, 20, 450, fs2)
    # sdSig = calculateICA(sdSig, 64)

    signalWindow, labelWindow = windowingSig(sdSig, force, windowSize=maxFeatures)  # should be divideable to 4

    # signalWindow = calculateFFTOnWindows(signalWindow)
    labelWindow = calculateForceAverage(labelWindow)
    # signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
    # signalWindow, labelWindow = calculateICA(signalWindow, labelWindow, 4)

    # signalWindow = extractPhaseSpace(signalWindow)
    # signalWindow = extractSoundFeatures(signalWindow)

    return signalWindow, labelWindow


def prepareData(sig, firings):
    sdSig = calculateSD(sig)
    # sdSig = calculateICA(sdSig, 64)
    sdSig = butter_bandpass_filter(sdSig, 20, 450, 2048)
    sizeInputSignal = sdSig.shape[1]
    preparedFirings = prepareFiringSignal(firings[0], sizeInputSignal)
    signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=maxFeatures)  # should be divideable to 4
    # signalWindow = calculateFFTOnWindows(signalWindow)
    labelWindow = calculateCdr(labelWindow)
    # signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
    # signalWindow, labelWindow = calculateICA(signalWindow, labelWindow, 4)

    signalWindow = extractPhaseSpace(signalWindow)
    # signalWindow = extractSoundFeatures(signalWindow)

    return signalWindow, labelWindow


signalWindow, labelWindow = prepareDataWithForce(trainSigs[0], trainTargets[0])
for index in range(1, len(trainSigs)):
    sig, label = prepareDataWithForce(trainSigs[index], trainTargets[index])
    signalWindow = np.append(signalWindow, sig, axis=0)
    labelWindow = np.append(labelWindow, label, axis=0)

signalWindow2, labelWindow2 = prepareDataWithForce(sig2, force2)

# num_classes = len(np.unique(labelWindow))
# labelWindow2[np.where(labelWindow2 > num_classes)] = num_classes - 1
labelWindow = np.abs(labelWindow)
labelWindow2 = np.abs(labelWindow2)

labelWindow = (labelWindow - np.min(labelWindow)) / (np.max(labelWindow) - np.min(labelWindow))
labelWindow2 = (labelWindow2 - np.min(labelWindow2)) / (np.max(labelWindow2) - np.min(labelWindow2))

X_train, X_test, Y_train, Y_test = trainTestSplit(signalWindow, labelWindow, 0.75)
X_train2, X_test2, Y_train2, Y_test2 = trainTestSplit(signalWindow2, labelWindow2, 0.75)

# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)
# Y_train2 = to_categorical(Y_train2)
# Y_test2 = to_categorical(Y_test2)

signal_size_row = X_train[0].shape[0]
signal_size_col = X_train[0].shape[1]
x_train = np.reshape(X_train, [-1, signal_size_row, signal_size_col])
x_test = np.reshape(X_test, [-1, signal_size_row, signal_size_col])
X_train2 = np.reshape(X_train2, [-1, signal_size_row, signal_size_col])
X_test2 = np.reshape(X_test2, [-1, signal_size_row, signal_size_col])
input_shape = (signal_size_row, signal_size_col)

lstm_out = 16


def counting(args):
    input = args
    var = tf.reduce_sum(input, axis=1, keepdims=False) / tf.reduce_max(input)
    return var


model = Sequential()
model.add(Conv1D(64, 5, padding='same', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(GRU(lstm_out, return_sequences=True))
# model.add(LSTM(lstm_out))
# model.add(Lambda(counting,
#                  name='z'))
model.add(Flatten())
model.add(Dense(16, activity_regularizer=l2(0.001)))
# model.add(GRU(lstm_out, return_sequences=True))
# model.add(LSTM(lstm_out))
# model.add(Dense(20, activity_regularizer=l2(0.001)))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss=keras.losses.mean_absolute_error, optimizer='nadam',
              metrics=[metrics.RootMeanSquaredError(), metrics.MAE])
print(model.summary())

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test2.shape)

batch_size = 12
epochs = 50
reduce_lr_acc = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=epochs / 10, verbose=1, min_delta=1e-4, mode='max')
model.fit(X_train, Y_train,
          epochs=epochs,
          batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[reduce_lr_acc])
model.save("ForceEstimation.h5", overwrite=True)
# acc = model.evaluate(X_test,
#                      Y_test,
#                      batch_size=batch_size,
#                      verbose=0)

out = model.predict(X_train2, batch_size=batch_size)
predicted2 = out.ravel()

out = model.predict(X_test, batch_size=batch_size)
predicted = out.ravel()

# sns.set(color_codes=True)
# ax = sns.regplot(x=Y_test2, y=predicted2, color="g")

r1 = r2_score(Y_train2, predicted2)
r2 = r2_score(Y_test, predicted)
print("New Signal R2:{0}".format(r1))
print("Same Signal R2:{0}".format(r2))
plt.plot(Y_train2)
plt.plot(predicted2)
plt.legend(['Target', 'Estimated'], loc='upper left')
plt.show()


out = model.predict(signalWindow2, batch_size=batch_size)
rect_predict = out.ravel()
plt.plot(labelWindow2)
plt.plot(rect_predict)
plt.legend(['Target', 'Estimated'], loc='upper left')
plt.show()

blandAltman(Y_test, predicted,
            savePath='SavedFigureAltman.svg',
            figureFormat='svg')



df=pd.DataFrame(columns=["target","predicted"])
df["predicted"]=predicted2
df["target"]=Y_train2
df.to_excel("result.xlsx")
pass
