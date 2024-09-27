import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert iris.data to a .csv
with open('./iris.data', 'r') as in_file:
    read = csv.reader(in_file)
    with open('./iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

# Read csv data from iris.csv
iris = pd.read_csv("./iris.csv")

# Plot Style
plt.style.use('_mpl-gallery')

def train_test(train_pct, test_pct):
    total_data = len(iris)
    train_size = int(total_data * (train_pct / 100))

    # Shuffle the data
    shuffled_iris = iris.sample(frac=1, random_state=42).reset_index(drop=True)

    x = shuffled_iris['Petal Length']
    y = shuffled_iris['Petal Width']
    species = shuffled_iris['Species']

    # Train data
    train_x = x[:train_size]
    train_y = y[:train_size]
    train_species = species[:train_size]

    # Test data 
    test_x = x[train_size:]
    test_y = y[train_size:]
    test_species = species[train_size:]

    # Set up a 1x2 subplot layout (one for scatter plot, one for ROC curve)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot scatter plot for train and test data
    ax[0].scatter(train_x, train_y, color='blue', label='Train Data')
    ax[0].scatter(test_x, test_y, color='red', label='Test Data')
    ax[0].set_xlabel('Petal Length')
    ax[0].set_ylabel('Petal Width')
    ax[0].set_title(f'Train-Test Split: {train_pct}% Train, {test_pct}% Test')
    ax[0].legend()

    # Thresholds
    thresholds = np.linspace(0, 7, 100)  # Define 100 threshold levels between 0 and 7

    # Lists to store true positive rates and false positive rates
    tprs = []
    fprs = []

    for threshold in thresholds:
        tp = np.sum((test_x > threshold) & (test_species == 'Iris-virginica'))
        fp = np.sum((test_x > threshold) & (test_species != 'Iris-virginica'))
        tn = np.sum((test_x <= threshold) & (test_species != 'Iris-virginica'))
        fn = np.sum((test_x <= threshold) & (test_species == 'Iris-virginica'))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # Plot ROC Curve
    ax[1].plot(fprs, tprs, color='darkorange', lw=2, label='ROC curve')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal reference line
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC)')
    ax[1].legend(loc="lower right")

    # Calculate AUC
    auc = np.trapz(tprs, fprs)

    # Show plot
    plt.tight_layout()
    plt.show()

    # Calculate metrics for a specific threshold (e.g., 4.5)
    threshold = 4.5
    y_pred = (test_x > threshold).astype(int)
    y_true = (test_species == 'Iris-virginica').astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    print(f"Train: {train_pct}%, Test: {test_pct}%")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Sensitivity: {sensitivity(tp, fn):.2f}")
    print(f"Specificity: {specificity(tn, fp):.2f}")
    print(f"Accuracy: {accuracy(tp, tn, fp, fn):.2f}")
    print(f"AUC: {auc:.2f}")

def sensitivity(tp, fn):   
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def specificity(tn, fp):
    if tn + fp == 0:
        return 0
    return tn / (tn + fp)

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def main():
    train_test(80, 20)  # 80% train, 20% test
    train_test(70, 30)  # 70% train, 30% test

if __name__ == "__main__":
    main()