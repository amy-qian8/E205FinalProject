import PDFs
import staticBayes
import realtimeBayes
from constants import *

def main():
    # Import data and create 4 PDFs using calibration data
    accel_data_all = PDFs.importData()
    statmu, statstd = PDFs.createPDFHistogram("sit", accel_data_all)
    lyingmu, lyingstd = PDFs.createPDFHistogram("lying", accel_data_all)
    walkmu, walkstd = PDFs.createPDFHistogram("walk", accel_data_all)
    jogmu, jogstd = PDFs.createPDFHistogram("jog", accel_data_all)

    # Choose with type of 4 state Bayes filter (static or realtime)
    static = True
    if static:
        testData = accel_data_all["Testing 3"].dropna().to_numpy()
        staticBayes.staticBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd, testData)
    else:
        realtimeBayes.realtimeBayesWrapper(statmu, statstd, lyingmu, lyingstd, walkmu, walkstd, jogmu, jogstd)

main()
