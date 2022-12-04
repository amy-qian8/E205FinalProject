import PDFs
import staticBayes
import realtimeBayes
from constants import *
from matplotlib import pyplot as plt
import serial

sidney_dec4_trial_sec = [20, 25, 20, 20, 20, 20, 20, 20]
sidney_dec4_trial = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 0, 3, 0, 2, 3, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 3]
amy_dec4_trial_sec = [20, 20, 20, 20, 20, 20, 20, 20]
amy_dec4_trial = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 3, 3, 2, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
kaanthi_dec4_trial_sec = [20, 23, 20, 19, 17, 20, 21, 20]
kaanthi_dec4_trial = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 0, 3, 1, 0, 3, 3, 2, 1, 3, 3, 3, 1, 2, 3, 3, 3, 3, 2, 1, 3, 1, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def addState(sec, state):
    arr = []
    for i in range(0, int(14*(sec/10))): # num pts with 1.4 Hz
        arr.append(state)
    return arr

def ground_truth(seconds):
    arr = []
    states = [0, 2, 1, 2, 3, 2, 0, 1]
    for i in range(0, 8):
        arr = arr + addState(seconds[i], states[i])
    return arr

def determine_error(sec, trial):
    truth = ground_truth(sec)
    correct = 0
    for i in range(0, len(truth)):
        if truth[i] == trial[i]:
            correct+=1
    print(correct/len(truth))
    plt.plot(truth)
    plt.plot(trial)
    plt.show()

def main():
    # Import data and create 4 PDFs using calibration data
    accel_data_all = PDFs.importData()
    statmu, statstd = PDFs.createPDFHistogram("sit", accel_data_all)
    lyingmu, lyingstd = PDFs.createPDFHistogram("lying", accel_data_all)
    walkmu, walkstd = PDFs.createPDFHistogram("walk", accel_data_all)
    jogmu, jogstd = PDFs.createPDFHistogram("jog", accel_data_all)

    # Choose with type of 4 state Bayes filter (static or realtime)
    static = False
    if static:
        testData = accel_data_all["Testing 1"].dropna().to_numpy()
        staticBayes.staticBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, testData)
    else:
        ser = serial.Serial('COM5')
        ser.flushInput()
        realtimeBayes.realtimeBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, ser)

# main()
determine_error(kaanthi_dec4_trial_sec, kaanthi_dec4_trial)