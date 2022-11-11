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
        accel_data_all = pd.concat([accel_data_all, df], axis=1).dropna()
print('Done loading data.')

# Compile all the accel data into one array, rename the column titles
accel_data_all.columns = ["Stationary Kaanthi", "Walking Kaanthi"]
print(accel_data_all)

def createPDFHistogram():

    # Plot the histogram for Stationary Kaanthi
    # i = range(len(accel_data_all.columns))
    i = 1
    energyArr = np.array(accel_data_all.iloc[:, i])
    print(energyArr)
    plt.hist(energyArr)
    # Plotting the PDF (probability density function) to the histogram
    mu, std = norm.fit(energyArr)
    x = np.linspace(min(energyArr), max(energyArr), 100)
    pdf = norm.pdf(x, mu, std) * 10000  # note big scale factor
    plt.plot(x, pdf, 'k', linewidth=2)
    name = list(accel_data_all.columns)
    plt.suptitle("Histogram of " + name[i], fontsize=12)
    plt.title("p(ei|xi = stationary)", fontsize=10)
    plt.xlabel("Energy [m/s]")
    plt.ylabel("Count of data points")
    plt.show()
    return mu, std, energyArr

createPDFHistogram()