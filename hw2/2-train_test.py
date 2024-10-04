# Problem #2 - Peter Bizoukas, Python 3.8
# 2. Using the iris data set of assignment 1 [https://archive.ics.uci.edu/dataset/53/iris,
# using python convert .data files into .csv.] split it into 
# a. 80% train and 20% test data
# b. 70% train and 30% test data 
# c. Compare the accuracy, specificity and sensitivity of a and b and with the ROC curve 
# mention which is a better model. The grader should be able to generate the curve and the stats after running your program. 

# Split your data set into 80% training data and 20% test [for splitting use np.split techniques or masking], then use KNN where K = any odd no of your choice [as this is a KNN assignment], 
# here you will be only using Euclidean distance.
# So basically, write one function or more which calculates Euclidean distance of your test data points one by one with all the training data points and assign classes to the test data based on the value of K
# Compare it with existing test data classes - find overall accuracy, specificity and sensitivity. 
# Repeat the same for 70% and 30 % split.
# Compare the ROCs in a single plot to determine the best model. 
# Do not use pandas - you can use csv package to read the data.

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

# KNN function that returns predictions and distances
def knn(train_features, train_labels, test_features, k):
    predictions = []
    distances_to_predictions = []
    
    for test_instance in test_features: # For each test instance
        distances = [ # Calculate distances to all training instances
            (distance.euclidean(test_instance.astype(float), train_instance.astype(float)), label) 
            for train_instance, label in zip(train_features, train_labels)
        ]
        
        sorted_neighbors = sorted(distances, key=lambda x: x[0]) # Sort by distance
        neighbors = [neighbor for _, neighbor in sorted_neighbors[:k]] # Get the labels of the k nearest neighbors
        class_votes = Counter(neighbors) # Count the votes for each class using Counter
        prediction = class_votes.most_common(1)[0][0]  # Predict the most common class
        predictions.append(prediction) # Store prediction
        distances_to_predictions.append([d[0] for d in sorted_neighbors[:k]])  # Store distances for ROC

    return np.array(predictions), np.array(distances_to_predictions)

# Confusion matrix calculation
def confusion_matrix_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    
    if (TP + FN) > 0: # Sensitivity
        sensitivity = TP / (TP + FN)
    else:
        sensitivity = 0.0

    if (TN + FP) > 0: # Specificity
        specificity = TN / (TN + FP)
    else:
        specificity = 0.0
    
    return accuracy, sensitivity, specificity

# Update the ROC curve plot function to handle distances
def plot_roc_curve(y_true, distances, title):
    fpr, tpr, thresholds = [], [], np.linspace(0, 1, 100) # Thresholds for ROC curve
    
    for thresh in thresholds:
        y_pred = np.where(np.mean(distances, axis=1) <= thresh, 1, 0)  # Using mean distances as probability threshold
        _, sensitivity, specificity = confusion_matrix_metrics(y_true, y_pred) # Calculate metrics
        fpr.append(1 - specificity) 
        tpr.append(sensitivity)
    
    plt.plot(fpr, tpr, label=f'ROC Curve - {title}')

# Main function to handle both splits
def main():
    # Load the dataset
    iris_data = load_iris_data("iris.data")
    features, labels = iris_data[:, :-1], iris_data[:, -1]
    labels = np.where(labels == 'Iris-versicolor', 1, 0)  # Convert to binary labels

    # Split 80-20
    train_data, test_data = train_test_split(iris_data, test_size=0.2) # 80-20 split
    train_features, train_labels = train_data[:, :-1], train_data[:, -1] # Extract features and labels
    train_labels = np.where(train_labels == 'Iris-versicolor', 1, 0) # Convert labels to binary
    test_features, test_labels = test_data[:, :-1], test_data[:, -1]
    test_labels = np.where(test_labels == 'Iris-versicolor', 1, 0)

    # KNN with k=3
    k = 3
    predictions, distances = knn(train_features, train_labels, test_features, k) 

    # Metrics
    accuracy, sensitivity, specificity = confusion_matrix_metrics(test_labels, predictions)
    print(f"KNN with k = {k}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")

    # Repeat for 70-30 split
    train_data_70, test_data_30 = train_test_split(iris_data, test_size=0.3) # 70-30 split
    train_features_70, train_labels_70 = train_data_70[:, :-1], train_data_70[:, -1] 
    train_labels_70 = np.where(train_labels_70 == 'Iris-versicolor', 1, 0)
    test_features_30, test_labels_30 = test_data_30[:, :-1], test_data_30[:, -1]
    test_labels_30 = np.where(test_labels_30 == 'Iris-versicolor', 1, 0)

    predictions_70, distances_70 = knn(train_features_70, train_labels_70, test_features_30, k)
    accuracy_70, sensitivity_70, specificity_70 = confusion_matrix_metrics(test_labels_30, predictions_70)
    
    print(f"\nKNN with k = {k} (70-30 split)")
    print(f"Accuracy: {accuracy_70:.2f}")
    print(f"Sensitivity: {sensitivity_70:.2f}")
    print(f"Specificity: {specificity_70:.2f}")

      # Print which model is better
    if accuracy > accuracy_70:
        print("\n80%-20% Split is more accurate")
    elif accuracy == accuracy_70:
        print("\nBoth splits are equally accurate")
    else:
        print("\n70%-30% Split is more accurate")

    # Plot ROC curves
    plt.figure()
    plot_roc_curve(test_labels, distances, '80%-20% Split') # Plot ROC curve for 80-20 split
    plot_roc_curve(test_labels_30, distances_70, '70%-30% Split') # Plot ROC curve for 70-30 split
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')  # Random classifier line
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

  
    

if __name__ == "__main__":
    main()
