from unittest import TestCase

from Utils.SignalUtils import getSignal, calculateSD, prepareFiringSignal, windowingSig, trainTestSplit, calculateICA, \
    calculateWhiten, autocorr, calculateNormalizedOnCorr, extractFeatures
import matplotlib.pyplot as plt
from SaveData import saveDataAsImage
from numpy.linalg import pinv


class TestSignalUtils(TestCase):

    def test_loadSignal(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        TestCase.assertTrue(self, len(sig) > 0)

    def test_SD_signal(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        sdSig = calculateSD(sig)
        TestCase.assertTrue(self, len(sdSig) > 0)

    def test_prepare_firings(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        TestCase.assertTrue(self, len(preparedFirings) > 0)

    def test_windowingSignal(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        sdSig = calculateSD(sig)
        signal, label = windowingSig(sdSig, preparedFirings, windowSize=15)
        TestCase.assertTrue(self, len(signal) > 0)
        TestCase.assertTrue(self, len(label) > 0)

    def test_train_test_split(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        sdSig = calculateSD(sig)
        signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=15)
        X_train, X_test, y_train, y_test = trainTestSplit(signalWindow, labelWindow, 0.7)
        TestCase.assertTrue(self, len(X_train) > 0)
        TestCase.assertTrue(self, len(X_test) > 0)
        TestCase.assertTrue(self, len(y_train, ) > 0)
        TestCase.assertTrue(self, len(y_test) > 0)

    def test_save_data_as_images(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        sdSig = calculateSD(sig)
        signalWindow, labelWindow = windowingSig(sdSig, preparedFirings, windowSize=15)
        saveDataAsImage(signalWindow, labelWindow)

    def test_ica(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        sdSig = calculateSD(sig)
        sdSigIca = calculateICA(sdSig)
        TestCase.assertTrue(self, len(sdSigIca) > 0)

    def test_CKC(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        sdSig = calculateSD(sig)
        whitenSdSig = calculateWhiten(sdSig)
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        signalWindow, labelWindow = windowingSig(whitenSdSig, preparedFirings, windowSize=124)
        index = 50
        Rx = autocorr(signalWindow[index])
        t = pinv(Rx) @ signalWindow[index]
        plt.plot(t[1, :])
        plt.show()
        TestCase.assertTrue(self, len(t) > 0)

    def test_featureSelection(self):
        sig, force, fs, firings = getSignal(1, 30, "../signals/")
        sdSig = calculateSD(sig)
        whitenSdSig = calculateWhiten(sdSig)
        preparedFirings = prepareFiringSignal(firings[0], len(sig[0, 0][0]))
        signalWindow, labelWindow = windowingSig(whitenSdSig, preparedFirings, windowSize=124)
        signalWindow, labelWindow = calculateNormalizedOnCorr(signalWindow, labelWindow)
        featureExtracted=extractFeatures(signalWindow)

        TestCase.assertTrue(self, len(featureExtracted) > 0)
