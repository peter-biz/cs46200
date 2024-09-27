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
with open('./iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
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


    

    # Thresholds
    thresholds = np.linspace(0, 5, 100) # Define 100 threshold levels between 0 and 7

    # Calculate accuracy, specificity, and sensitivity
    tp = 0 # True Positive
    tn = 0 # True Negative
    fp = 0 # False Positive
    fn = 0 # False Negative

    for threshold in thresholds: # Loop through thresholds
        for i in range(len(test_x)): # Loop through test data
            if test_x.iloc[i] > threshold and test_y.iloc[i] > threshold: # If test data is in the top right quadrant
                # pos
                if test_x.iloc[i] > 4.5 and test_y.iloc[i] > 1.5: 
                    tp += 1 
                else:
                    fp += 1
            else: # If test data is not in the top right quadrant
                # neg
                if test_x.iloc[i] < 4.5 and test_y.iloc[i] < 1.5: 
                    tn += 1
                else:
                    fn += 1

    if(fp + tn == 0):
        fprs = 0
    else:
        fprs = fp / (fp + tn) # False Positive Rate

    sens = sensitivity(tp, fn)

    # Plot ROC Curve
    ax[1].plot(fprs, sens, color='darkorange', lw=2, label='ROC curve')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal reference line
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC)')
    ax[1].legend(loc="lower right")

    # Show plot
    plt.tight_layout() # Makes sure the plot stays within the plot window
    plt.show()

    
    
    print("Train: " + str(train_pct) + "%, Test: " + str(test_pct) + "%")
    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("Sensitivity: " + str(sensitivity(tp, fn)))
    print("Specificity: " + str(specificity(tn, fp)))
    print("Accuracy: " + str(accuracy(tp, tn, fp, fn)))



def sensitivity(tp, fn):   
    if tp + fn == 0:
        return "Error: no positive predictions made" 
    return tp / (tp + fn)

def specificity(tn, fp):
    if tn + fp == 0:
        return "Error: no negative predictions made"
    return tn / (tn + fp)

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def main():
    train_test(80, 20) # 80% train, 20% test
    train_test(70, 30) # 70% train, 30% test

if __name__ == "__main__":
    main()