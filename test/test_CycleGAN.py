from unittest import TestCase

from DeepLeraning.CycleGAN import CycleGAN

from Utils.SignalUtils import getSignal2, calculateSD, butter_bandpass_filter, prepareFiringSignal, windowingSig, calculateCdr, extractPhaseSpace, trainTestSplit, calculateICA
import numpy as np


class TestCycleGAN(TestCase):

    def prepareData(self, sig, firings):
        sdSig = calculateSD(sig)
        # sdSig = calculateICA(sdSig, 64)
        sdSig = butter_bandpass_filter(sdSig, 20, 450, 2048)
        sizeInputSignal = sdSig.shape[1]
        preparedFirings = prepareFiringSignal(firings[0], sizeInputSignal, numSignals=12)
        signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=256)  # should be divideable to 4
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

        cycleGAN = CycleGAN(X_train.shape[1], X_train.shape[2])
        cycleGAN.train(x_train=X_train,y_train=Y_train,epochs=100)
