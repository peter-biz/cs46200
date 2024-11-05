#1.	Only using NumPy, Matplotlib package solve the following problem:

# Obtain the file crash.txt from Brightspace. This file contains measurements from a crash test dummy during an NHSA automobile crash test. 
# Column 0 measures time in milliseconds after the crash event and column 1 represents the acceleration measured by a sensor on the dummy head. 
# The file is formatted according to Numpys default specification for textual data (spaces delineate columns, newlines delineate rows). 
# Thus it can easily be read into a matrix via the following:
#  data = numpy.loadtxt(crash.txt)
#  Divide this data into a training and test set of equal size even-numbered rows to the training set and odd-numbered rows to test. 
# Write a function to fit the line using Least Square model with the formulas of slope and intercept from your textbook [Page no. 677]. 
# Plot the training data using the above model function with the lowest SSE on the training
#  data. Repeat the same for the Test Data. Calculate the RMS error for both. 

import numpy as np
import matplotlib.pyplot as plt



def main():
    data = np.loadtxt('crash.txt') # Load the data from the file   
     

if __name__ == "__main__":
    main()