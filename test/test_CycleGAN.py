from unittest import TestCase

from sklearn.preprocessing import scale

import numpy as np

from DeepLeraning.CycleGAN import CycleGAN
from DeepLeraning.CycleGANModified import CycleGANModified
from Utils.SignalUtils import calculateSD, butter_bandpass_filter, prepareFiringSignal, windowingSig, getSignal2, trainTestSplit


class TestCycleGAN(TestCase):

    def prepareData(self, sig, firings):
        sdSig = calculateSD(sig)
        sdSig = scale(sdSig, axis=0)

        # sdSig = calculateICA(sdSig, 64)
        sdSig = butter_bandpass_filter(sdSig, 20, 450, 2048)
        sizeInputSignal = sdSig.shape[0]
        preparedFirings = prepareFiringSignal(firings[0], sizeInputSignal, numSignals=12)
        signalWindow, labelWindow = windowingSig(sdSig.transpose(), preparedFirings, windowSize=256)  # should be divideable to 4
        # signalWindow = calculateFFTOnWindows(signalWindow)
        # labelWindow = convertLabel2OneDimentional(labelWindow)
        # signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
        # signalWindow, labelWindow = calculateICA(signalWindow, labelWindow, 18)

        # signalWindow = extractPhaseSpace(signalWindow)
        # signalWindow = extractSoundFeatures(signalWindow)
        return signalWindow, np.asarray(labelWindow)

    def test_trainSignal(self):
        maxFeatures = 10
        # sig, force, fs, firings = getSignal(1, 30, "./signals/")
        trainSigs = []
        trainFirings = []
        for i in range(10, 17):
            sig, force, fs, firings = getSignal2("002041436" + str(i), filePath="../signals/Hamid/00MaHaLI1002041436/")
            trainSigs.append(sig)
            trainFirings.append(firings)
        sig2, force2, fs2, firings2 = getSignal2("00204143607", filePath="../signals/Hamid/00MaHaLI1002041436/")

        # sig2, force2, fs2, firings2 = getSignal(2, 30, "./signals/")

        signalWindow, labelWindow = self.prepareData(trainSigs[0], trainFirings[0])
        for index in range(1, len(trainSigs)):
            sig, label = self.prepareData(trainSigs[index], trainFirings[index])
            signalWindow = np.append(signalWindow, sig, axis=0)
            labelWindow = np.append(labelWindow, label, axis=0)

        signalWindow2, labelWindow2 = self.prepareData(sig2, firings2)

        X_train, X_test, Y_train, Y_test = trainTestSplit(signalWindow, labelWindow, 0.75)
        X_train2, X_test2, Y_train2, Y_test2 = trainTestSplit(signalWindow2, labelWindow2, 0.75)

        X_train = np.reshape(X_train, [-1, X_train.shape[1], X_train.shape[2]])
        # X_test = np.reshape(X_test, [-1, X_test.shape[1], X_test.shape[2], 1])
        Y_train = np.reshape(Y_train, [-1, Y_train.shape[1], Y_train.shape[2]])
        # y_test = np.reshape(Y_test, [-1, Y_test.shape[1], Y_test.shape[2], 1])

        # Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train))
        # X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))

        cycleGAN = CycleGANModified(Y_train.shape[1], Y_train.shape[2])
        cycleGAN.train(x_train=X_train[:, :, range(Y_train.shape[2])], y_train=Y_train, epochs=100)
