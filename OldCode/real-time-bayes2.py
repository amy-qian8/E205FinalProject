import math
import numpy as np
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt
import pandas as pd
import os
import serial

ser = serial.Serial('COM5')
ser.flushInput()

# Pulling Data from all CSV files into DataFrame
accel_data_all = pd.DataFrame()
for file in os.listdir():
    if file.endswith('.csv'):
        df = pd.read_csv(file).iloc[:, -1]
        # print("DF: ", df)
        accel_data_all = pd.concat([accel_data_all, df], axis=1)
        # print("Data all: ", accel_data_all)
print('Done loading data.')

# Compile all the accel data into one array, rename the column titles
accel_data_all.columns = ["Sitting Energy", "Lying Energy", "Walking Energy", "Jogging Energy", "Testing 1", "Testing 2", "Testing 3"]

def createPDFHistogram(cols):
    # Plot the histogram for Stationary Kaanthi
    # i = range(len(accel_data_all.columns))
    combinedEnergyArr = np.array(accel_data_all.iloc[:, cols].dropna())
    # combinedEnergyArr = np.array(accel_data_all.iloc[:, cols[0]].dropna())
    # combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, cols[1]].dropna()), axis=0)
    # combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, cols[2]].dropna()), axis=0)

    # print("LEN combinedEnergyArr: ", len(combinedEnergyArr))

    plt.hist(combinedEnergyArr)
    # Plotting the PDF (probability density function) to the histogram
    mu, std = norm.fit(combinedEnergyArr)
    x = np.linspace(min(combinedEnergyArr), max(combinedEnergyArr), 100)
    pdf = norm.pdf(x, mu, std) * 10000  # note big scale factor
    plt.plot(x, pdf, 'k', linewidth=2)
    name = list(accel_data_all.columns)

    # Setting the "state" variable so that the graph title and subtitle update dynamically
    if cols == 0:
        state = "sitting"
    elif cols == 1:
        state = "lying"
    elif cols == 2:
        state = "walking"
    elif cols == 3:
        state = "running"

    plt.suptitle("Histogram of " + state + " Data" + " mu: " + str(round(mu)) + " std: " + str(round(std)), fontsize=12)
    plt.title("p(ei|xi = " + state + ")", fontsize=10)
    plt.xlabel("Energy")
    plt.ylabel("Count of data points")
    plt.show()

    return mu, std

# def plot3PDFs():
#     mu = [16970, 16946, 19025]
#     std = [308, 2895, 10391]
#
#     pdf0 = norm.pdf(mu[0], std[0])
#     pdf1 = norm.pdf(mu[1], std[1])
#     pdf2 = norm.pdf(mu[2], std[2])
#     # plt.plot(pdf0, 'k', linewidth=2)
#     # plt.plot(pdf1, 'k', linewidth=2)
#     # plt.plot(pdf2, 'k', linewidth=2)
#     # plt.show()

def bayes_filter(statMU, statSTD, walkMU, walkSTD, jogMU, jogSTD, testEnergy):
    """Given the vehicle's prior state and current speed, what's the likelihood that the vehicle is stopped?"""
    priorBelStat = 0.34  # The prior belief
    priorBelWalk = 0.33  # The prior belief
    priorBelJog = 0.33  # The prior belief

    # S = stat, W = walk, J = jog

    probStoS = 0.6
    probStoW = 0.35
    probStoJ = 0.05

    probWtoS = 0.3
    probWtoW = 0.6
    probWtoJ = 0.1

    probJtoS = 0.1
    probJtoW = 0.4
    probJtoJ = 0.5

    belCorrectionStatNorm = [priorBelStat]
    belCorrectionWalkNorm = [priorBelWalk]
    belCorrectionJogNorm = [priorBelJog]

    for e in testEnergy:
        # PREDICTION STEP: Belief in each state
        belPredictionStat = (probStoS * priorBelStat) + (probWtoS * priorBelWalk) + (probJtoS * priorBelJog)  # Prediction of being stat
        belPredictionWalk = (probStoW * priorBelStat) + (probWtoW * priorBelWalk) + (probJtoW * priorBelJog)  # Prediction of being walk
        belPredictionJog = (probStoJ * priorBelStat) + (probWtoJ * priorBelWalk) + (probJtoJ * priorBelJog)  # Prediction of being jog

        # CORRECTION STEP: Belief in each state
        numeratorStat = norm.pdf(e, statMU, statSTD) * belPredictionStat
        numeratorWalk = norm.pdf(e, walkMU, walkSTD) * belPredictionWalk
        numeratorJog = norm.pdf(e, jogMU, jogSTD) * belPredictionJog

        normFactor = (numeratorStat + numeratorWalk + numeratorJog)
        belCorrectionStatNorm.append(numeratorStat / normFactor)
        belCorrectionWalkNorm.append(numeratorWalk / normFactor)
        belCorrectionJogNorm.append(numeratorJog / normFactor)

        priorBelStat = belCorrectionStatNorm[-1]  # update prior belief for next iteration
        priorBelWalk = belCorrectionWalkNorm[-1]  # update prior belief for next iteration
        priorBelJog = belCorrectionJogNorm[-1]  # update prior belief for next iteration

    return belCorrectionStatNorm, belCorrectionWalkNorm, belCorrectionJogNorm
    
def bayes_filter4(statMU, statSTD, lyingMU, lyingSTD, walkMU, walkSTD, jogMU, jogSTD):
    """Given the vehicle's prior state and current speed, what's the likelihood that the vehicle is stopped?"""
    priorBelStat = 0.25  # The prior belief
    priorBelLying = 0.25  # The prior belief
    priorBelWalk = 0.25  # The prior belief
    priorBelJog = 0.25  # The prior belief

    # S = stat, W = walk, J = jog

    probStoS = 0.6
    probStoL = 0.25
    probStoW = 0.1
    probStoJ = 0.05

    probLtoS = 0.1
    probLtoL = 0.6
    probLtoW = 0.15
    probLtoJ = 0.15

    probWtoS = 0.2
    probWtoL = 0.1
    probWtoW = 0.6
    probWtoJ = 0.1

    probJtoS = 0.1
    probJtoL = 0.05
    probJtoW = 0.15
    probJtoJ = 0.7

    belCorrectionStatNorm = [priorBelStat]
    belCorrectionLyingNorm = [priorBelLying]
    belCorrectionWalkNorm = [priorBelWalk]
    belCorrectionJogNorm = [priorBelJog]

    while True:
        ser_bytes = ser.readline()
        decoded_bytes = ser_bytes[0:len(ser_bytes)-2].decode("utf-8")
        str_list = decoded_bytes.split(",")
        output = [int(i) for i in str_list]
        e = math.sqrt(output[0]**2 + output[1]**2 + output[2]**2)

        # PREDICTION STEP: Belief in each state
        belPredictionStat = (probStoS * priorBelStat) + (probLtoS * priorBelLying) + (probWtoS * priorBelWalk) + (probJtoS * priorBelJog)  # Prediction of being stat
        belPredictionLying = (probStoL * priorBelStat) + (probLtoL * priorBelLying) + (probWtoL * priorBelWalk) + (probJtoL * priorBelJog)  # Prediction of being stat
        belPredictionWalk = (probStoW * priorBelStat) + (probLtoW * priorBelLying) + (probWtoW * priorBelWalk) + (probJtoW * priorBelJog)  # Prediction of being walk
        belPredictionJog = (probStoJ * priorBelStat) + (probLtoJ * priorBelLying) + (probWtoJ * priorBelWalk) + (probJtoJ * priorBelJog)  # Prediction of being jog

        # CORRECTION STEP: Belief in each state
        numeratorStat = norm.pdf(e, statMU, statSTD) * belPredictionStat
        numeratorLying = norm.pdf(e, lyingMU, lyingSTD) * belPredictionLying
        numeratorWalk = norm.pdf(e, walkMU, walkSTD) * belPredictionWalk
        numeratorJog = norm.pdf(e, jogMU, jogSTD) * belPredictionJog

        normFactor = (numeratorStat + numeratorLying + numeratorWalk + numeratorJog)
        statProb = numeratorStat / normFactor
        lyingProb = numeratorLying / normFactor
        walkProb = numeratorWalk / normFactor
        jogProb = numeratorJog / normFactor

        priorBelStat = belCorrectionStatNorm[-1]  # update prior belief for next iteration
        priorBelLying = belCorrectionLyingNorm[-1]  # update prior belief for next iteration
        priorBelWalk = belCorrectionWalkNorm[-1]  # update prior belief for next iteration
        priorBelJog = belCorrectionJogNorm[-1]  # update prior belief for next iteration

        printState(statProb, lyingProb, walkProb, jogProb)

    return belCorrectionStatNorm, belCorrectionLyingNorm, belCorrectionWalkNorm, belCorrectionJogNorm

stateOverTime = []
def printState(stat, lying, walk, jog):
    if max(stat, lying, walk, jog) == stat:
        print("sitting")
        stateOverTime.append(0)
    elif max(stat, lying, walk, jog) == lying:
        print("lying")
        stateOverTime.append(1)
    elif max(stat, lying, walk, jog) == walk:
        print("walk")
        stateOverTime.append(2)
    elif max(stat, lying, walk, jog) == jog:
        print("jog")
        stateOverTime.append(3)
    else:
        print("all equal")
        stateOverTime.append(0)

def getState(state, stat, lying, walk, jog):
    for i in range(len(state)):
        if max(stat[i], lying[i], walk[i], jog[i]) == stat[i]:
            state[i] = 0
        elif max(stat[i], lying[i], walk[i], jog[i]) == lying[i]:
            state[i] = 1
        elif max(stat[i], lying[i], walk[i], jog[i]) == walk[i]:
            state[i] = 2
        elif max(stat[i], lying[i], walk[i], jog[i]) == jog[i]:
            state[i] = 3
        else:
            state[i] = 0

    return state

def getStateWithWindow(state, windowRadius):
    stateWithWindow = state

    for i in range(len(stateWithWindow)):
        if i > windowRadius and i < (len(stateWithWindow) - windowRadius):
            if stateWithWindow[i-1] == stateWithWindow[i+1]:
                stateWithWindow[i] = stateWithWindow[i-1]

    # PREVIOUS TRIAL CODE THAT WORKS BUT DOESN'T YIELD DESIRED RESULT
    # count = 0
    # for i in range(len(stateWithWindow)):
    #     if i > windowRadius:
    #         count = 0
    #         for j in range(windowRadius):
    #             if stateWithWindow[i-1] == stateWithWindow[i-j-1]:
    #                 count+=1
    #         if count == windowRadius: # If the last N data points are all in the same state, then ...
    #             stateWithWindow[i] = stateWithWindow[i-1]

    return stateWithWindow

def plotState(axs, time, state, barHeight):
    for i in range(len(state)):
        if (state[i] == 0):
            axs[1].scatter(time[i], barHeight, c = 'blue')
            # plt.scatter(time[i], barHeight, c = 'blue')
        elif (state[i] == 1):
            axs[1].scatter(time[i], barHeight, c = 'orange')
            # plt.scatter(time[i], barHeight, c = 'orange')
        elif (state[i] == 2):
            axs[1].scatter(time[i], barHeight, c = 'green')
            # plt.scatter(time[i], barHeight, c = 'green')
        # For testing
        elif (state[i] == 3):
            axs[1].scatter(time[i], barHeight, c = 'red')
            # plt.scatter(time[i], barHeight, c = 'red')

def main():
    # Create the PDFs from the pre-collected data
    statmu, statstd = createPDFHistogram(0) #Stat
    lyingmu, lyingstd = createPDFHistogram(1)  # Lying
    walkmu, walkstd = createPDFHistogram(2) #Walk
    jogmu, jogstd = createPDFHistogram(3) #Jog

    try:
        bayes_filter4(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd)
    except KeyboardInterrupt:
        y = np.arange(0, 4, 1)
        y_ticks_labels = ['stationary', 'lying down', 'walking', 'jogging']
        fig, ax = plt.subplots(1, 1)
        ax.plot(stateOverTime)
        # Set number of ticks for x-axis
        ax.set_yticks(y)
        # Set ticks labels for x-axis
        ax.set_yticklabels(y_ticks_labels)
        plt.show()

main()
