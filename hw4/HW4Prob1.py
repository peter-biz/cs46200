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

# Function to calculate the slope and intercept of the line
def least_square(x, y):  
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x*y)
    slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    intercept = (sum_y - slope*sum_x) / n
    return slope, intercept


# Function to plot the data
def plot_data(x, y, slope, intercept, title):
    plt.plot(x, y, 'ro') # ro is red color
    plt.plot(x, slope*x + intercept, 'b') # b is blue color
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration')
    plt.title(title)
    plt.show()

# Function to calculate the RMS error
def rms_error(x, y, slope, intercept):
    y_pred = slope*x + intercept # Predicted values
    rms = np.sqrt(np.sum((y - y_pred)**2) / len(x))
    return rms

# Main function
def main():
    data = np.loadtxt('hw4\crash.txt') # Load the data from the file   

    # Divide the data into training and test set
    x_train = data[::2, 0] # Even-numbered rows to the training set
    y_train = data[::2, 1] # Odd-numbered rows to test set
    x_test = data[1::2, 0] # Even-numbered rows to the training set
    y_test = data[1::2, 1] # Odd-numbered rows to test set

    # Fit the line using Least Square model
    slope, intercept = least_square(x_train, y_train)

    # Plot the training data
    plot_data(x_train, y_train, slope, intercept, 'Train Data')

    # Calculate the RMS error for training data
    rms_train = rms_error(x_train, y_train, slope, intercept)
    print('RMS error for training data:', rms_train)

    # Plot the test data
    plot_data(x_test, y_test, slope, intercept, 'Test Data')

    # Calculate the RMS error for test data
    rms_test = rms_error(x_test, y_test, slope, intercept)
    print('RMS error for test data:', rms_test)


     

if __name__ == "__main__":
    main()