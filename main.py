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
accel_data_all.columns = ["Stationary Kaanthi", "Stationary Sidney", "Stationary Amy", "Walking Kaanthi", "Walking Sidney", "Walking Amy", "Running Kaanthi", "Running Sidney", "Running Amy"]
# print(accel_data_all)

def createPDFHistogram():

    # Plot the histogram for Stationary Kaanthi
    # i = range(len(accel_data_all.columns))
    i = [6, 7, 8]
    energyArr = np.array(accel_data_all.iloc[:, i])
    combinedEnergyArr = np.array(accel_data_all.iloc[:, i[0]].dropna())
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, i[1]].dropna()), axis=0)
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, i[2]].dropna()), axis=0)

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
    return mu, std, combinedEnergyArr

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

createPDFHistogram()
# plot3PDFs()