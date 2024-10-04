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
import csv
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance



# Load the Iris dataset manually from .data file
def load_iris_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                data.append(row)
    return np.array(data)

# Convert the data to a .csv file
def convert_to_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Split data into train and test sets
def train_test_split(data, test_size):
    np.random.shuffle(data)  # Shuffle the data
    split_idx = int(len(data) * (1 - test_size))  # Split index
    return data[:split_idx], data[split_idx:]

# KNN function
def knn(train_features, train_labels, test_features, k):
    predictions = []
    for test_instance in test_features:
        distances = [
            (distance.euclidean(test_instance.astype(float), train_instance.astype(float)), label) 
            for train_instance, label in zip(train_features, train_labels)
        ]
        sorted_neighbors = sorted(distances, key=lambda x: x[0])
        neighbors = [neighbor for _, neighbor in sorted_neighbors[:k]]
        class_votes = Counter(neighbors)
        prediction = class_votes.most_common(1)[0][0]
        predictions.append(prediction)
    return np.array(predictions)

# Confusion matrix calculation
def confusion_matrix_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return accuracy, sensitivity, specificity

# ROC curve plot function
def plot_roc_curve(y_true, y_pred, title): 
    fpr, tpr, thresholds = [], [], np.linspace(0, 1, 100) 
    
    for thresh in thresholds:
        _, sensitivity, specificity = confusion_matrix_metrics(y_true, (y_pred >= thresh).astype(int))
        fpr.append(1 - specificity)
        tpr.append(sensitivity)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function to handle both splits
def main():
    # Step 1: Load the dataset and convert it to CSV
    iris_data = load_iris_data("iris.data")
    convert_to_csv(iris_data, "iris.csv")
    
    # Extract features (X) and labels (y)
    features, labels = iris_data[:, :-1], iris_data[:, -1]
    labels = np.where(labels == 'Iris-versicolor', 1, 0)  # Binary encoding for a specific class

    # Split the data into 80% training and 20% test
    train_data, test_data = train_test_split(iris_data, test_size=0.2)
    train_features, train_labels = train_data[:, :-1], train_data[:, -1]
    train_labels = np.where(train_labels == 'Iris-versicolor', 1, 0)  # Ensure training labels are numeric
    test_features, test_labels = test_data[:, :-1], test_data[:, -1]
    test_labels = np.where(test_labels == 'Iris-versicolor', 1, 0)  # Ensure test labels are numeric

    # Split the data into 70% training and 30% test
    train_data_70, test_data_30 = train_test_split(iris_data, test_size=0.3)
    train_features_70, train_labels_70 = train_data_70[:, :-1], train_data_70[:, -1]
    train_labels_70 = np.where(train_labels_70 == 'Iris-versicolor', 1, 0)  # Ensure training labels are numeric
    test_features_30, test_labels_30 = test_data_30[:, :-1], test_data_30[:, -1]
    test_labels_30 = np.where(test_labels_30 == 'Iris-versicolor', 1, 0)  # Ensure test labels are numeric


    # KNN with k = 3
    k = 3
    predictions = knn(train_features, train_labels, test_features, k)
    
    # Calculate metrics for 80-20 split
    accuracy, sensitivity, specificity = confusion_matrix_metrics(test_labels, predictions)
    print(f"KNN with k = {k}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")

    # Calculate metrics for 70-30 split
    predictions_70 = knn(train_features_70, train_labels_70, test_features_30, k)
    accuracy_70, sensitivity_70, specificity_70 = confusion_matrix_metrics(test_labels_30, predictions_70)
    print(f"\nKNN with k = {k} (70-30 split)")
    print(f"Accuracy: {accuracy_70:.2f}")
    print(f"Sensitivity: {sensitivity_70:.2f}")
    print(f"Specificity: {specificity_70:.2f}")



    # Plot ROC curve for 80-20 split
    plot_roc_curve(test_labels, predictions, title=f"KNN with k = {k} (80-20 split)")

    # Plot ROC curve for 70-30 split
    plot_roc_curve(test_labels_30, predictions_70, title=f"KNN with k = {k} (70-30 split)")

if __name__ == "__main__":
    main()

