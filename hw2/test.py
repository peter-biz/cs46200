import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Convert iris.data to .csv
with open('./iris.data', 'r') as in_file:  # Updated for forward slashes
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

    x = iris['Petal Length']
    y = iris['Petal Width']

    # Train data
    train_x = x[:train_size]
    train_y = y[:train_size]

    # Test data 
    test_x = x[train_size:]
    test_y = y[train_size:]

    # Set up a 1x2 subplot layout (one for scatter plot, one for ROC curve)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot scatter plot for train and test data
    ax[0].scatter(train_x, train_y, color='blue', label='Train Data')
    ax[0].scatter(test_x, test_y, color='red', label='Test Data')
    ax[0].set_xlabel('Petal Length')
    ax[0].set_ylabel('Petal Width')
    ax[0].set_title(f'Train-Test Split: {train_pct}% Train, {test_pct}% Test')
    ax[0].legend()

    # Define thresholds to evaluate
    thresholds = np.linspace(0, 5, 100)  # Define 100 threshold levels between 0 and 5

    tprs = []  # True Positive Rate
    fprs = []  # False Positive Rate

    for threshold in thresholds:
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(test_x)):
            # Define classification based on the threshold for petal length and width
            if test_x.iloc[i] > threshold and test_y.iloc[i] > threshold:
                # If classified as positive
                if test_x.iloc[i] > 4.5 and test_y.iloc[i] > 1.5:
                    tp += 1
                else:
                    fp += 1
            else:
                # If classified as negative
                if test_x.iloc[i] < 4.5 and test_y.iloc[i] < 1.5:
                    tn += 1
                else:
                    fn += 1

        # Calculate TPR (Sensitivity) and FPR
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

    # Tight layout for spacing
    plt.tight_layout()
    plt.show()

    print(f"Train: {train_pct}%, Test: {test_pct}%")
    print(f"Thresholds: {thresholds}")
    print(f"TPRs: {tprs}")
    print(f"FPRs: {fprs}")
    print("Tp: ", tp)
    print("Tn: ", tn)
    print("Fp: ", fp)
    print("Fn: ", fn)

    # Calculate Sensitivity, Specificity, and Accuracy
    if(fp + tn == 0):
        spec = 0
    else:
        spec = tn / (fp + tn)
    
    

    print("Sensitivity: ", tpr)
    print("Specificity: ", spec)
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))


def main():
    train_test(80, 20)  # 80% train, 20% test
    train_test(70, 30)  # 70% train, 30% test

if __name__ == "__main__":
    main()
