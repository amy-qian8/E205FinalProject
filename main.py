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
        df = pd.read_csv(file).iloc[:, -1].dropna()
        print(df)
        accel_data_all = pd.concat([accel_data_all, df], axis=1)
print('Done loading data.')

# Compile all the accel data into one array, rename the column titles
accel_data_all.columns = ["Stationary Kaanthi", "Stationary Sidney", "Stationary Amy", "Walking Kaanthi", "Walking Sidney", "Walking Amy", "Running Kaanthi", "Running Sidney", "Running Amy"]
print(accel_data_all)

def createPDFHistogram():

    # Plot the histogram for Stationary Kaanthi
    # i = range(len(accel_data_all.columns))
    i = [0,1,2]
    energyArr = np.array(accel_data_all.iloc[:, i])
    combinedEnergyArr = np.array(accel_data_all.iloc[:, i[0]])
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, i[1]]), axis=0)
    combinedEnergyArr = np.append(combinedEnergyArr, np.array(accel_data_all.iloc[:, i[2]]), axis=0)

    plt.hist(combinedEnergyArr)
    # Plotting the PDF (probability density function) to the histogram
    mu, std = norm.fit(combinedEnergyArr)
    x = np.linspace(min(combinedEnergyArr), max(combinedEnergyArr), 100)
    pdf = norm.pdf(x, mu, std) * 5000  # note big scale factor
    plt.plot(x, pdf, 'k', linewidth=2)
    name = list(accel_data_all.columns)
    plt.suptitle("Histogram of " + "Stationary Data", fontsize=12)
    plt.title("p(ei|xi = stationary)", fontsize=10)
    plt.xlabel("Energy [m/s]")
    plt.ylabel("Count of data points")
    plt.show()
    return mu, std, combinedEnergyArr

createPDFHistogram()