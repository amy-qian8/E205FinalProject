import csv
import math
import random
import numpy as np
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt

class AccelData(object):

    def __init__(self, row):
        self.X_accel = row[0]
        self.Y_accel = row[1]
        self.Z_accel = row[2]
        self.Energy = row[3]

# Read in the data
print('Opening the csv file')
with open('E205 Final Project Data - Stationary (Sitting) Kaanthi.csv') as csv_file:
    print('Reading the csv file')
    readCSV = csv.reader(csv_file, delimiter=',')
    accel_data_all = []
    for row in readCSV:
        accel_data_all.append(AccelData(row))
    print('3. Done loading data.')

# Compile all the accel data into one array, removing the column titles
accel_data = accel_data_all[1:]

def createPDFHistogram():
    # Removing the blank rows from the data automatically
    stationaryX_Kaanthi = []
    stationaryY_Kaanthi = []
    stationaryZ_Kaanthi = []
    stationaryE_Kaanthi = []

    for row in accel_data:
        if any(char.strip() for char in row.X_accel):
            stationaryX_Kaanthi.append(row.X_accel)
        if any(char.strip() for char in row.Y_accel):
            stationaryY_Kaanthi.append(row.Y_accel)
        if any(char.strip() for char in row.Z_accel):
            stationaryZ_Kaanthi.append(row.Z_accel)
        if any(char.strip() for char in row.Energy):
            stationaryE_Kaanthi.append(float(row.Energy))

    # Plot the histogram for Stationary Kaanthi
    energyArr = np.array(stationaryE_Kaanthi)
    print(energyArr)
    plt.hist(energyArr)
    # Plotting the PDF (probability density function) to the histogram
    mu, std = norm.fit(energyArr)
    x = np.linspace(min(energyArr), max(energyArr), 100)
    pdf = norm.pdf(x, mu, std) * 10000
    plt.plot(x, pdf, 'k', linewidth=2)
    plt.suptitle("Histogram of Stationary Kaanthi", fontsize=12)
    plt.title("p(ei|xi = stopped)", fontsize=10)
    plt.xlabel("Energy [m/s]")
    plt.ylabel("Count of data points")
    plt.show()
    return mu, std, energyArr

createPDFHistogram()