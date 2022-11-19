import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from constants import *

def staticBayes(statMU, statSTD, lyingMU, lyingSTD, walkMU, walkSTD, jogMU, jogSTD, testEnergy):
    # Initial Belief
    priorBelStat = 0.25
    priorBelLying = 0.25
    priorBelWalk = 0.25
    priorBelJog = 0.25
    
    belCorrectionStatNorm = [priorBelStat]
    belCorrectionLyingNorm = [priorBelLying]
    belCorrectionWalkNorm = [priorBelWalk]
    belCorrectionJogNorm = [priorBelJog]

    for e in testEnergy:
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
        belCorrectionStatNorm.append(numeratorStat / normFactor)
        belCorrectionLyingNorm.append(numeratorLying / normFactor)
        belCorrectionWalkNorm.append(numeratorWalk / normFactor)
        belCorrectionJogNorm.append(numeratorJog / normFactor)

        priorBelStat = belCorrectionStatNorm[-1]  # update prior belief for next iteration
        priorBelLying = belCorrectionLyingNorm[-1]  # update prior belief for next iteration
        priorBelWalk = belCorrectionWalkNorm[-1]  # update prior belief for next iteration
        priorBelJog = belCorrectionJogNorm[-1]  # update prior belief for next iteration

    return belCorrectionStatNorm, belCorrectionLyingNorm, belCorrectionWalkNorm, belCorrectionJogNorm

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

    return stateWithWindow

def plotState(axs, time, state, barHeight):
    for i in range(len(state)):
        if (state[i] == 0):
            axs[1].scatter(time[i], barHeight, c = 'blue')
        elif (state[i] == 1):
            axs[1].scatter(time[i], barHeight, c = 'orange')
        elif (state[i] == 2):
            axs[1].scatter(time[i], barHeight, c = 'green')
        elif (state[i] == 3):
            axs[1].scatter(time[i], barHeight, c = 'red')

def staticBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, testData):
    # Call Bayes Filter on the test data
    stat, lying, walk, jog = staticBayes(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, testData)

    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, 1)

    samplingRate = 1.4 #Hz
    emptyState = np.zeros(len(stat))
    state = getState(emptyState, stat, lying, walk, jog)
    time = np.linspace(0, (1/samplingRate) * len(state), num = len(state)) #start, stop, num

    # Subplot 1: Plot the state probablity
    axs[0].plot(time, stat, label="stat")
    axs[0].plot(time, lying, label="lying")
    axs[0].plot(time, walk, label="walk")
    axs[0].plot(time, jog, label="jog")
    axs[0].set_ylabel("Probability")
    axs[0].legend()

    # Subplot 2: Plot the state (with and without window)
    windowRadius = 2
    plotState(axs, time, state, 1.2)
    stateWithWindow = getStateWithWindow(state, windowRadius)
    plotState(axs, time, stateWithWindow, 1.1)
    axs[1].set_xlabel("Time [seconds]")
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_title("Top line is state without window, Bottom line is state with window")

    fig.suptitle("Estimating state based on accel data using Bayes Filter")
    plt.show()
    