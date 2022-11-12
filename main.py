import csv
import math
import random
import numpy as np
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt
import pandas as pd
import os

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
    state = "running"
    plt.suptitle("Histogram of " + state + " Data" + " mu: " + str(round(mu)) + " std: " + str(round(std)), fontsize=12)
    plt.title("p(ei|xi = " + state + ")", fontsize=10)
    plt.xlabel("Energy [m/s]")
    plt.ylabel("Count of data points")
    plt.show()

    return mu, std

def plot3PDFs():
    mu = [16970, 16946, 19025]
    std = [308, 2895, 10391]

    pdf0 = norm.pdf(mu[0], std[0])
    pdf1 = norm.pdf(mu[1], std[1])
    pdf2 = norm.pdf(mu[2], std[2])
    plt.plot(pdf0, 'k', linewidth=2)
    plt.plot(pdf1, 'k', linewidth=2)
    plt.plot(pdf2, 'k', linewidth=2)
    plt.show()

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

def main():
    statmu, statstd = createPDFHistogram([0, 1, 2]) #Stat
    walkmu, walkstd = createPDFHistogram([3, 4, 5]) #Walk
    jogmu, jogstd = createPDFHistogram([6, 7, 8]) #Jog

    energyTestData = accel_data_all["Test Amy"].dropna().to_numpy()
    print(energyTestData)

    stat, walk, jog = bayes_filter(statmu, statstd, walkmu, walkstd, jogmu, jogstd, energyTestData)
    plt.plot(stat, label="stat")
    plt.plot(walk, label="walk")
    plt.plot(jog, label="jog")
    plt.legend()
    plt.show()

main()
