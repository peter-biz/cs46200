# 1.	Using the iris data set of assignment 1 and 2 [https://archive.ics.uci.edu/dataset/53/iris,
# using python convert .data files into .csv.] split it into 
# a.	80% train and 20% test data
# b.	70% train and 30% test data 
# c.	Write in your own words to compare the accuracy of a and b using any graph of your choice on the following algorithms.
# i.	Decision Trees
# ii.	Random Forest

# Use any python packages of your choice. Mention clearly in readme if that requires additional installation of packages. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv

# Conver iris.data to a .csv
with open('.\iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
    read = csv.reader(in_file)
    with open('.\iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

iris = pd.read_csv('.\iris.csv')

# Split data into training and testing sets
def tts(test_size): 
    # Split data into training and testing sets
    X = iris[['Septal Length', 'Septal Width', 'Petal Length', 'Petal Width']]
    y = iris['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

def main():
    tts(0.8)
    tts(0.7)

if __name__ == "__main__":
    main()