# Problem #2 - Peter Bizoukas, Python 3.8
# 2.	Using the iris data set of assignment 1 [https://archive.ics.uci.edu/dataset/53/iris,
# using python convert .data files into .csv.] split it into 
# a.	80% train and 20% test data
# b.	70% train and 30% test data 
# c.	Compare the accuracy, specificity and sensitivity of a and b and with the ROC curve 
# mention which is a better model. The grader should be able to generate the curve and the stats after running your program. 

# Split your data set into 80% training data and 20% test [for splitting use np.split techniques or masking], then use KNN where K = any odd no of your choice [as this is a KNN assignment], 
# here you will be only using Euclidean distance.
# So basically, write one function or more which calculates Euclidean distance of your test data points one by one with all the training data points and assign classes to the test data based on the value of K
# Compare it with existing test data classes - find overall accuracy, specificity and sensitivity. 
# Repeat the same for 70% and 30 % split.
# Compare the ROCs in a single plot to determine the best model. 
# Donot use pandas - you can use csv package to read the data.


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import csv

# Load the Iris dataset manually from .data file
def load_iris_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader if row]  # remove empty rows
    return np.array(data)

# Convert the data to a .csv file
def convert_to_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)




def main():
    iris = load_iris_data('iris.data')
    convert_to_csv(iris, 'iris.csv')
    


if __name__ == "__main__":
    main()