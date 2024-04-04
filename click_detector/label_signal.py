import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feature_extract



def loadSignal(csvfile):
    data = pd.read_csv(csvfile)
    signal = data['Signal']
    print(signal)
    return signal

def cropSignal(signal, start, size):
    crop = signal[start : start + size]
    return crop



import argparse
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-s", "--signal", help=".csv file of sampled signal", default='log_distance_2apr2024.csv')
# Read config file (for camera source, model etc)
args = parser.parse_args()
file_signal = args.signal



# load signal from csv file & plot
y = loadSignal(file_signal)
plt.figure(1)
plt.title("Sampled Signal Waveform")
plt.plot(y)
plt.show()



# setup DSP parameters
# create ref signal for correlation feature
framesize = 24
amplitude = 2
frequency = 2 # 2 sinewave within framesize
x_value = np.linspace(0, framesize, framesize)
y_ref = amplitude*np.sin(2*np.pi*frequency*x_value + np.pi/2.5)
plt.figure(1)
plt.title("Ref Signal")
plt.plot(y_ref)
plt.show()



col_names = ['mean', 'std', 'peak', 'autocorr', 'skewness', 'kurtosis', 'label']
df_train = pd.DataFrame([], columns=col_names)

isrunning = True
while isrunning:
    start = int(input("start index to crop: "))
    # crop & plot cropped signal
    y_crop = cropSignal(y, start, framesize)
    plt.figure(2)
    plt.title("Crop Signal Waveform")
    plt.plot(y_crop)
    plt.show()

    # feature extraction
    mean, std = feature_extract.calcMeanStd(y_crop)
    ptp = feature_extract.calcPeakToPeak(y_crop)
    autocorr = feature_extract.calcAutocorr(y_crop, y_ref)
    skewness = feature_extract.calcSkewness(y_crop)
    kurtosis = feature_extract.calcKurtosis(y_crop)
    print("mean=", mean, " std=", std, " peak-to-peak=", ptp)
    print("auto-correlation=", autocorr)
    print("skew=", skewness, " kurtosis=", kurtosis)

    prompt = input("save?[y] : ")
    if prompt == 'y':
        # create feature column
        label = input("label siganl type as : ")
        features = [mean, std, ptp, autocorr, skewness, kurtosis, label]
        df = pd.DataFrame([features], columns=col_names)
        df_train = pd.concat([df_train, df])
        print("panda formatted features=")
        print(df_train)
        print("*"*30)

    # prompt to continue or not
    prompt = input("continue?[y] : ")
    if prompt != 'y':
        isrunning = False


