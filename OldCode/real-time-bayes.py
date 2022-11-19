import math
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import pandas as pd
import os
import serial

ser = serial.Serial('COM7')
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
accel_data_all.columns = ["Stationary Kaanthi", "Stationary Sidney", "Stationary Amy", "Walking Kaanthi", "Walking Sidney", "Walking Amy", "Running Kaanthi", "Running Sidney", "Running Amy", "Test Kaanthi", "Test Sidney", "Test Amy"]

def createPDFHistogram(cols):
    # Plot the histogram for Stationary Kaanthi
    # i = range(len(accel_data_all.columns))
    energyArr = np.array(accel_data_all.iloc[:, cols])
    combinedEnergyArr = np.array(accel_data_all.iloc[:, cols[0]].dropna())
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, cols[1]].dropna()), axis=0)
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, cols[2]].dropna()), axis=0)

    # print("LEN combinedEnergyArr: ", len(combinedEnergyArr))

    plt.hist(combinedEnergyArr)
    # Plotting the PDF (probability density function) to the histogram
    mu, std = norm.fit(combinedEnergyArr)
    x = np.linspace(min(combinedEnergyArr), max(combinedEnergyArr), 100)
    pdf = norm.pdf(x, mu, std) * 10000  # note big scale factor
    plt.plot(x, pdf, 'k', linewidth=2)
    name = list(accel_data_all.columns)

    # Setting the "state" variable so that the graph title and subtitle update dynamically
    if cols == [0,1,2]:
        state = "stationary"
    elif cols == [3,4,5]:
        state = "walking"
    elif cols == [6,7,8]:
        state = "running"

    plt.suptitle("Histogram of " + state + " Data" + " mu: " + str(round(mu)) + " std: " + str(round(std)), fontsize=12)
    plt.title("p(ei|xi = " + state + ")", fontsize=10)
    plt.xlabel("Energy")
    plt.ylabel("Count of data points")
    # plt.show()

    return mu, std

# def plot3PDFs():
#     mu = [16970, 16946, 19025]
#     std = [308, 2895, 10391]

#     pdf0 = norm.pdf(mu[0], std[0])
#     pdf1 = norm.pdf(mu[1], std[1])
#     pdf2 = norm.pdf(mu[2], std[2])
#     plt.plot(pdf0, 'k', linewidth=2)
#     plt.plot(pdf1, 'k', linewidth=2)
#     plt.plot(pdf2, 'k', linewidth=2)
#     plt.show()

def bayes_filter(statMU, statSTD, walkMU, walkSTD, jogMU, jogSTD):
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

    while True:
        ser_bytes = ser.readline()
        decoded_bytes = ser_bytes[0:len(ser_bytes)-2].decode("utf-8")
        str_list = decoded_bytes.split(",")
        output = [int(i) for i in str_list]
        e = math.sqrt(output[0]**2 + output[1]**2 + output[2]**2)
        print("acc: ", output)
        print("e: ", e)

        # PREDICTION STEP: Belief in each state
        belPredictionStat = (probStoS * priorBelStat) + (probWtoS * priorBelWalk) + (probJtoS * priorBelJog)  # Prediction of being stat
        belPredictionWalk = (probStoW * priorBelStat) + (probWtoW * priorBelWalk) + (probJtoW * priorBelJog)  # Prediction of being walk
        belPredictionJog = (probStoJ * priorBelStat) + (probWtoJ * priorBelWalk) + (probJtoJ * priorBelJog)  # Prediction of being jog

        # CORRECTION STEP: Belief in each state
        numeratorStat = norm.pdf(e, statMU, statSTD) * belPredictionStat
        numeratorWalk = norm.pdf(e, walkMU, walkSTD) * belPredictionWalk
        numeratorJog = norm.pdf(e, jogMU, jogSTD) * belPredictionJog

        normFactor = (numeratorStat + numeratorWalk + numeratorJog)
        statProb = numeratorStat / normFactor
        walkProb = numeratorWalk / normFactor
        jogProb = numeratorJog / normFactor

        priorBelStat = belCorrectionStatNorm[-1]  # update prior belief for next iteration
        priorBelWalk = belCorrectionWalkNorm[-1]  # update prior belief for next iteration
        priorBelJog = belCorrectionJogNorm[-1]  # update prior belief for next iteration

        printState(statProb, walkProb, jogProb)

    return statProb, walkProb, jogProb

def getState(state, stat, walk, jog):
    for i in range(len(state)):
        if (stat[i] >= walk[i]) & (stat[i] >= jog[i]):
            # state = 0 corresponds to stat
            state[i] = 0
        elif (walk[i] >= stat[i]) & (walk[i] >= jog[i]):
            # state = 1 corresponds to walk
            state[i] = 1
        elif (jog[i] >= stat[i]) & (jog[i] >= walk[i]):
            # state = 2 corresponds to jog
            state[i] = 2
        else:
            # default to a 4th state (or should we default to stat?)
            state[i] = 3
    return state

stateOverTime = []
def printState(stat, walk, jog):
    if (stat >= walk) & (stat >= jog):
        # state = 0 corresponds to stat
        print("stationary")
        stateOverTime.append(0)
    elif (walk >= stat) & (walk >= jog):
        # state = 1 corresponds to walk
        print("walk")
        stateOverTime.append(1)
    elif (jog >= stat) & (jog >= walk):
        # state = 2 corresponds to jog
        print("jog")
        stateOverTime.append(2)
    else:
        # default to a 4th state (or should we default to stat?)
        print("all equal")

def getStateWithWindow(state, windowRadius, stat, walk, jog):
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

def plotState(state, barHeight):
    for i in range(len(state)):
        if (state[i] == 0):
            plt.scatter(i, barHeight, c = 'blue')
        elif (state[i] == 1):
            plt.scatter(i, barHeight, c = 'orange')
        elif (state[i] == 2):
            plt.scatter(i, barHeight, c = 'green')
        # For testing
        elif (state[i] == 3):
            plt.scatter(i, barHeight, c = 'red')

def main():
    statmu, statstd = createPDFHistogram([0, 1, 2]) #Stat
    walkmu, walkstd = createPDFHistogram([3, 4, 5]) #Walk
    jogmu, jogstd = createPDFHistogram([6, 7, 8]) #Jog

    energyTestData = accel_data_all["Test Amy"].dropna().to_numpy()

    try:
        bayes_filter(statmu, statstd, walkmu, walkstd, jogmu, jogstd)
    except KeyboardInterrupt:
        y = np.arange(0,3,1)
        y_ticks_labels = ['stationary','walking','jogging']
        fig, ax = plt.subplots(1,1) 
        ax.plot(stateOverTime)
        # Set number of ticks for x-axis
        ax.set_yticks(y)
        # Set ticks labels for x-axis
        ax.set_yticklabels(y_ticks_labels)
        plt.show()

main()