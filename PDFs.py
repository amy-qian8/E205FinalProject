# E205 Final Project
# Amy, Kaanthi, Sidney
# Dec 2022

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import pandas as pd
import os

def importData():
    # Pulling Data from all CSV files into DataFrame
    accel_data_all = pd.DataFrame()
    for file in os.listdir():
        if file.endswith('.csv'):
            df = pd.read_csv(file).iloc[:, -1]
            accel_data_all = pd.concat([accel_data_all, df], axis=1)
    print('Done loading data.')

    # Compile all the accel data into one array, rename the column titles
    accel_data_all.columns = ["Sitting Energy", "Lying Energy", "Walking Energy", "Jogging Energy", "Testing 1", "Testing 2", "Testing 3"]

    return accel_data_all

def createPDFHistogram(state, accel_data_all):
    # Setting the "cols" var based on the state parameter that gets passed in
    if state == "sit":
        cols = 0
    elif state == "lying":
        cols = 1
    elif state == "walk":
        cols = 2
    elif state == "jog":
        cols = 3

    energy = np.array(accel_data_all.iloc[:, cols].dropna())
    
    mu, std = norm.fit(energy)
    # plotPDF(mu, std, energy, state)

    return mu, std

def plotPDF(mu, std, energy, state):
    # Plotting the PDF (probability density function) to the histogram
    plt.hist(energy)
    x = np.linspace(min(energy), max(energy), 100)
    pdf = norm.pdf(x, mu, std) * 10000  # note big scale factor
    plt.plot(x, pdf, 'k', linewidth=2)

    plt.suptitle("Histogram of " + state + " Data" + " mu: " + str(round(mu)) + " std: " + str(round(std)), fontsize=12)
    plt.title("p(ei|xi = " + state + ")", fontsize=10)
    plt.xlabel("Energy")
    plt.ylabel("Count of data points")
    plt.show()
