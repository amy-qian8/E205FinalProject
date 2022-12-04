import math
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from termcolor import colored
import os

os.system('color')

from constants import *

sitOverTime = []
lyingOverTime = []
walkOverTime = []
jogOverTime = []
def realtimeBayes(statMU, statSTD, lyingMU, lyingSTD, walkMU, walkSTD, jogMU, jogSTD, ser):
    # Initial Belief
    priorBelStat = 0.25
    priorBelLying = 0.25
    priorBelWalk = 0.25
    priorBelJog = 0.25
    
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

        sitOverTime.append(statProb)
        lyingOverTime.append(lyingProb)
        walkOverTime.append(walkProb)
        jogOverTime.append(jogProb)
        printState(statProb, lyingProb, walkProb, jogProb)

stateOverTime = []
def printState(stat, lying, walk, jog):
    if max(stat, lying, walk, jog) == stat:
        print(colored("stationary",'blue'))
        stateOverTime.append(0)
    elif max(stat, lying, walk, jog) == lying:
        print(colored("lying", 'yellow'))
        stateOverTime.append(1)
    elif max(stat, lying, walk, jog) == walk:
        print(colored("walk", 'green'))
        stateOverTime.append(2)
    elif max(stat, lying, walk, jog) == jog:
        print(colored("jog", 'red'))
        stateOverTime.append(3)
    else:
        print("all equal")
        stateOverTime.append(0)

def realtimeBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, ser):
    try:
        realtimeBayes(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, ser)
    except KeyboardInterrupt:
        y = np.arange(0, 4, 1)
        y_ticks_labels = ['stationary', 'lying down', 'walking', 'jogging']
        fig, ax = plt.subplots(1, 1)
        ax.plot(sitOverTime, label = 'sitting')
        ax.plot(lyingOverTime, label = 'lying down')
        ax.plot(walkOverTime, label = 'walking')
        ax.plot(jogOverTime, label = 'jogging')

        samplingRate = 1.4
        time = np.linspace(0, (1/samplingRate) * len(sitOverTime), num = len(sitOverTime))

        ax.set_ylabel("Probability")
        ax.set_xlabel("Time (seconds)")
        ax.legend()

        print(stateOverTime)

        plt.show()
