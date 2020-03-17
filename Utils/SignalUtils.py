import scipy.io
import os
import numpy as np
from scipy.linalg import pinv
from scipy.signal import butter, lfilter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from scipy import signal
from sonopy import mfcc_spec


def getSignal(lib, mvc, filePath="./signals/"):
    if os.path.exists(filePath):
        signals = os.listdir(filePath)
        for signal in signals:
            if signal.__contains__("Lib" + str(lib)) and signal.__contains__(str(mvc) + "MVC"):
                mat = scipy.io.loadmat(filePath + signal)
                force = mat['Force']
                fs = mat['fsamp']
                firings = mat['sFirings']
                sig = mat['sig_out']
                return sig, force, fs, firings


def getSignal2(sign, filePath="signals/Hamid/00MaHaLI1002041436/"):
    if os.path.exists(filePath):
        signals = os.listdir(filePath)
        for signal in signals:
            if signal.__contains__(sign):
                mat = scipy.io.loadmat(filePath + signal)
                fs = mat['fsamp']
                firings = mat['MUPulses']
                force = mat['ref_signal']
                sig = mat['SIG']
                sig = np.delete(sig, 4, axis=0)
                return sig, force, fs, firings


def getSignal3(sign, filePath="../signals/Hamid2/DataReadyPython/"):
    if os.path.exists(filePath):
        signals = os.listdir(filePath)
        for signal in signals:
            if signal.__contains__(sign):
                mat = scipy.io.loadmat(filePath + signal)
                fs = mat['fs']
                force = mat['ref_signal']
                sig = mat['SIG']
                # sig = np.delete(sig, 4, axis=0)
                return sig, force, fs, None


# calculating the SD signal
def calculateSD(sig):
    nRow, nCol = sig.shape
    singleDifferentialSignal = []
    for row in range(nRow - 1):
        for col in range(nCol):
            singleDifferentialSignal.extend(sig[row + 1, col] - sig[row, col])
    singleDifferentialSignal = np.array(singleDifferentialSignal)
    return singleDifferentialSignal


def calculateICA(sdSig, labels, component=7):
    ica = FastICA(n_components=component, max_iter=1000)
    icaRes = []
    labelNew = []
    for index, sig in enumerate(sdSig):
        try:
            if labels[index].shape[0] == component and labels[index].shape[1] == sdSig[0].shape[1]:
                icaRes.append(np.array(ica.fit_transform(sig.transpose())).transpose())
                labelNew.append(labels[index])
        except:
            pass
    return np.array(icaRes), np.array(labelNew)


def extractPhaseSpace(sig):
    allFeature = []
    for numberData in range(len(sig)):
        feature = []
        for channel in range(sig[numberData].shape[0]):
            x = sig[numberData][channel, :]
            x = x / np.max(x)
            dx = np.diff(x, axis=0, prepend=[0])
            dx = dx / np.max(dx)
            feature.append(x)
            feature.append(dx)
        allFeature.append(np.asarray(feature))
    return np.asarray(allFeature)


def extractSoundFeatures(sig):
    allFeature = []
    for numberData in range(len(sig)):
        feature = []
        for channel in range(sig[numberData].shape[0]):
            x = sig[numberData][channel, :]
            powers, filters, mels, mfccs = mfcc_spec(x, 2048, return_parts=True, num_coeffs=len(x))
            feature.append(np.std(mfccs, axis=0))
            feature.append(np.sum(mfccs, axis=0))
            feature.extend(mfccs)
            # feature.append(np.max(powers, axis=0))
            # feature.append(np.min(powers, axis=0))
        allFeature.append(np.asarray(feature))
    return np.asarray(allFeature)


def calculateSTFT(sdSigWindows):
    sigWindowsNew = []
    for sig in sdSigWindows:
        f, t, Zxx = signal.stft(sig, 2048, nperseg=128)
        sigWindowsNew.append(np.abs(Zxx))
    return np.asarray(sigWindowsNew)


def calculateFFTOnWindows(sdSigWindows):
    return np.array([np.array(np.abs(np.fft.fft(sdSig.transpose()))).transpose() for sdSig in sdSigWindows])


def prepareFiringSignal(firings, sizeInputSignal=None, numSignals=None):
    maxIndex = np.max([np.max(firing[0]) for firing in firings])
    numbers = len(firings)
    preparedFirings = np.zeros([numbers, maxIndex + 1], dtype=float)
    for idx, firing in enumerate(firings):
        preparedFirings[idx, firing[0]] = 1
    if sizeInputSignal:
        preparedFirings = preparedFirings[:, 0:sizeInputSignal]
    if numSignals:
        return preparedFirings[0:numSignals, :]
    return preparedFirings


def windowingSig(sig, labels, windowSize=15):
    signalLen = labels.shape[1]
    if len(labels.shape) == 1:
        labelsWindow = [labels[int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - 1, windowSize)]
    else:
        labelsWindow = [labels[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
    signalsWindow = [sig[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]

    return signalsWindow, labelsWindow


def trainTestSplit(sig, label, trainPercent, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def calculateCdr(labels):
    return np.array([np.sum(label) for label in labels])


def calculateForceAverage(labels):
    return np.array([np.median(label) for label in labels])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data.transpose(), axis=1)
    return y.transpose()


def calculateWhiten(sdSig):
    return np.array(whiten(sdSig.transpose())).transpose()


def autocorr(x):
    corr = signal.correlate2d(x, x, boundary='symm', mode='same')
    return corr


def calculateNormalizedOnCorr(signalWindow, label):
    Rx = [autocorr(sigwin) for sigwin in signalWindow]
    mainSignalWindow = []
    mainLalbe = []
    for index, rx in enumerate(Rx):
        signalWindow[index] = pinv(rx) @ signalWindow[index]
        if signalWindow[index].shape[1] == signalWindow[0].shape[1]:
            mainSignalWindow.append(signalWindow[index])
            mainLalbe.append(label[index])

    return np.array(mainSignalWindow), np.array(mainLalbe)


def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)
