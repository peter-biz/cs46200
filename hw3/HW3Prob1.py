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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
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

# Read iris.csv
iris = pd.read_csv('.\iris.csv')

def decision_tree(train, test):
    # Split data into features and target
    X_train = train.iloc[:, :-1] # all rows, all columns except the last one
    y_train = train.iloc[:, -1] # all rows, only the last column
    X_test = test.iloc[:, :-1] # all rows, all columns except the last one
    y_test = test.iloc[:, -1] # all rows, only the last column

    # Create a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train) 

    # Predict the target
    y_pred = clf.predict(X_test) 

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def random_forest(train, test):
    # Split data into features and target
    X_train = train.iloc[:, :-1] # all rows, all columns except the last one
    y_train = train.iloc[:, -1] # all rows, only the last column
    X_test = test.iloc[:, :-1] # all rows, all columns except the last one 
    y_test = test.iloc[:, -1] # all rows, only the last column

    # Create a random forest classifier 
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict the target
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def plot_accuracy(accuracy_a, accuracy_b, title):
    # Data to plot
    n_groups = 2
    accuracy = [accuracy_a, accuracy_b]

    # Create bar graph
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, accuracy, bar_width,
    alpha=opacity,
    color='b',
    label='Accuracy')

    plt.xlabel('Split Data')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy of {title}')
    plt.xticks(index, ('80% Train', '70% Train')) 
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Split data into 80% train and 20% test data
    train, test = train_test_split(iris, test_size=0.2)

    # Calculate accuracy of Decision Trees and Random Forest on 80% train data
    acc_dec_tree_a = decision_tree(train, test)
    acc_random_forest_a = random_forest(train, test)

    # Split data into 70% train and 30% test data
    train, test = train_test_split(iris, test_size=0.3)

    # Calculate accuracy of Decision Trees and Random Forest on 70% train data
    acc_dec_tree_b = decision_tree(train, test)
    acc_random_forest_b = random_forest(train, test)

    # Print accuracy of Decision Trees & Plot
    print("Accuracy of Decision Trees with 80% Train Data:", str(acc_dec_tree_a*100) + "%")
    print("Accuracy of Decision Trees with 70% Train Data:", str(acc_dec_tree_b*100) + "%")
    plot_accuracy(acc_dec_tree_a, acc_dec_tree_b, "Decision Trees")

    # Print accuracy of Random Forest & Plot
    print("Accuracy of Random Forest with 80% Train Data:", str(acc_random_forest_a*100) + "%")
    print("Accuracy of Random Forest with 70% Train Data:", str(acc_random_forest_b*100) + "%")
    plot_accuracy(acc_random_forest_a, acc_random_forest_b, "Random Forest")

if __name__ == "__main__":
    main()