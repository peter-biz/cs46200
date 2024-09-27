# Problem #2 - Peter Bizoukas, Python 3.8
# 2.	Using the iris data set of assignment 1 [https://archive.ics.uci.edu/dataset/53/iris,
# using python convert .data files into .csv.] split it into 
# a.	80% train and 20% test data
# b.	70% train and 30% test data 
# c.	Compare the accuracy, specificity and sensitivity of a and b and with the ROC curve 
# mention which is a better model. The grader should be able to generate the curve and the stats after running your program. 


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import csv
import pandas as pd


# Conver iris.data to a .csv
with open('.\iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
    read = csv.reader(in_file)
    with open('.\iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

# Read csv data from iris.csv
iris = pd.read_csv(".\iris.csv")

# Plot Style
plt.style.use('_mpl-gallery')

# Figure size
plt.figure(figsize=(10,6))

def train_test(train, test):
    x = iris['Petal Length']
    y = iris['Petal Width']

    # Train data
    train_x = x[:train]
    train_y = y[:train]

    # Test data 
    test_x = x[test:]
    test_y = y[test:]




    

def main():
    train_test(80, 20) # 80% train, 20% test
    train_test(70, 30) # 70% train, 30% test

if __name__ == "__main__":
    main()