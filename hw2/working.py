import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
# Split data into train and test sets
def train_test_split(data, test_size):
    np.random.shuffle(data) # Shuffle the data
    split_idx = int(len(data) * (1 - test_size)) # Split index
    return data[:split_idx], data[split_idx:] 
# Sigmoid function for logistic regression
def sigmoid(z): # Sigmoid is a function that maps any real value into another value between 0 and 1
    return 1 / (1 + np.exp(-z)) 
# Logistic regression model
def logistic_regression(X, y, lr=0.01, num_iter=1000): # lr is the learning rate and num_iter is the number of iterations
    weights = np.zeros(X.shape[1]) # Initialize weights to zeros
    for i in range(num_iter): 
        predictions = sigmoid(np.dot(X, weights)) 
        gradient = np.dot(X.T, predictions - y) / len(y) 
        weights -= lr * gradient # Update weights
    return weights
# Prediction function
def predict(X, weights): 
    return sigmoid(np.dot(X, weights)) # Return the sigmoid of the dot product of X and weights
# Confusion matrix calculation
def confusion_matrix_metrics(y_true, y_pred, threshold=0.5): 
    y_pred_binary = np.where(y_pred >= threshold, 1, 0) # Convert probabilities to binary predictions
    TP = np.sum((y_true == 1) & (y_pred_binary == 1)) # True Positives
    TN = np.sum((y_true == 0) & (y_pred_binary == 0)) # True Negatives
    FP = np.sum((y_true == 0) & (y_pred_binary == 1)) # False Positives
    FN = np.sum((y_true == 1) & (y_pred_binary == 0)) # False Negatives
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return accuracy, sensitivity, specificity, y_pred_binary
# ROC curve plot function
def plot_roc_curve(y_true, y_score, title): 
    fpr, tpr, thresholds = [], [], np.linspace(0, 1, 100) 
    
    for thresh in thresholds:
        _, sensitivity, specificity, _ = confusion_matrix_metrics(y_true, y_score, thresh)
        fpr.append(1 - specificity)
        tpr.append(sensitivity)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (Area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend()
    plt.show()
# Area under curve (AUC) calculation
def auc(fpr, tpr):
    return np.trapz(tpr, fpr) # Calculate the area under the curve
# Main function to handle both splits
def main():
    # Step 1: Load the dataset and convert it to CSV
    iris_data = load_iris_data("iris.data")
    convert_to_csv(iris_data, "iris.csv")
    
    # Extract features (X) and labels (y)
    X = iris_data[:, :-1].astype(float)  # Feature columns
    y = np.array([1 if label == 'Iris-versicolor' else 0 for label in iris_data[:, -1]])  # Binary classification: 1 for versicolor, 0 for others
    # Normalize features for logistic regression
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Step 2: Split data into 80%-20% and 70%-30%
    train_80, test_20 = train_test_split(iris_data, 0.2)
    train_70, test_30 = train_test_split(iris_data, 0.3)
    # Train logistic regression model on 80%-20% split
    train_X_80, train_y_80 = train_80[:, :-1].astype(float), np.array([1 if label == 'Iris-versicolor' else 0 for label in train_80[:, -1]])
    test_X_80, test_y_80 = test_20[:, :-1].astype(float), np.array([1 if label == 'Iris-versicolor' else 0 for label in test_20[:, -1]])
    weights_80 = logistic_regression(train_X_80, train_y_80)
    test_pred_80 = predict(test_X_80, weights_80)
    
    # Train logistic regression model on 70%-30% split
    train_X_70, train_y_70 = train_70[:, :-1].astype(float), np.array([1 if label == 'Iris-versicolor' else 0 for label in train_70[:, -1]])
    test_X_70, test_y_70 = test_30[:, :-1].astype(float), np.array([1 if label == 'Iris-versicolor' else 0 for label in test_30[:, -1]])
    weights_70 = logistic_regression(train_X_70, train_y_70)
    test_pred_70 = predict(test_X_70, weights_70)
    # Step 3: Calculate and compare metrics for both splits
    acc_80, sens_80, spec_80, _ = confusion_matrix_metrics(test_y_80, test_pred_80)
    acc_70, sens_70, spec_70, _ = confusion_matrix_metrics(test_y_70, test_pred_70)
    print(f"80%-20% Split -> Accuracy: {acc_80}, Sensitivity: {sens_80}, Specificity: {spec_80}")
    print(f"70%-30% Split -> Accuracy: {acc_70}, Sensitivity: {sens_70}, Specificity: {spec_70}")
    if(acc_80 > acc_70):
        print("80%-20% Split is more accurate")
    elif(acc_80 == acc_70):
        print("Both splits are equally accurate")
    else:
        print("70%-30% Split is more accurate")
    # Step 4: Plot ROC curves
    plot_roc_curve(test_y_80, test_pred_80, '80%-20% Split')
    plot_roc_curve(test_y_70, test_pred_70, '70%-30% Split')
    
if __name__ == "__main__":
    main()