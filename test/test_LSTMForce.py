from unittest import TestCase


import tensorflow as tf
from Utils.SignalUtils import getSignal2, calculateSD, butter_bandpass_filter, prepareFiringSignal, windowingSig, calculateCdr, extractPhaseSpace, trainTestSplit, calculateICA, calculateForceAverage, \
    getSignal3
import numpy as np
import matplotlib.pyplot as plt


class TestLSTMForce(TestCase):

    def test_plotForcePredicted(self):
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

        X_train, X_test, Y_train, Y_test = trainTestSplit(signalWindow, labelWindow, 0.95, shuffle=False)
        X_train2, X_test2, Y_train2, Y_test2 = trainTestSplit(signalWindow2, labelWindow2, 0.95, shuffle=False)

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

        model = tf.keras.models.load_model('E:\Workspaces\crDsEMGEstimationPaper\EMGDecompositionDeepLearning\DeepLeraning\ForceEstimation.h5')

        batch_size = 12
        out = model.predict(X_train2, batch_size=batch_size)
        rect_predict = out.ravel()
        plt.plot(labelWindow2)
        plt.plot(rect_predict)
        plt.legend(['Target', 'Estimated'], loc='upper left')
        plt.show()
