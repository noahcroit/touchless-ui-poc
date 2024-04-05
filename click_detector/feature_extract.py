import numpy as np
from scipy import stats



def calcMeanStd(signal: np.ndarray):
    mean = np.mean(signal)
    std = stats.gstd(signal)  # Using scipy.stats for standard deviation
    return mean, std

def calcPeakToPeak(signal: np.ndarray):
    return np.ptp(signal)

def calcSkewness(signal: np.ndarray):
    return stats.skew(signal)

def calcKurtosis(signal: np.ndarray):
    return stats.kurtosis(signal)

def calcAutocorr(signal: np.ndarray, ref: np.ndarray):
    corr = np.correlate(signal, ref, mode='full')
    corr_max = corr[len(corr)//2:].max()
    return corr_max

def getFeatures(signal, autocorr_ref):
    signal_np = np.array(signal)
    mean, std = calcMeanStd(signal_np)
    ptp = calcPeakToPeak(signal_np)
    autocorr = calcAutocorr(signal_np, autocorr_ref)
    skewness = calcSkewness(signal_np)
    kurtosis = calcKurtosis(signal_np)
    features = np.array([mean, std, ptp, autocorr, skewness, kurtosis])
    return features
