# 2.	For this problem, you need to print the backpropagation weight updates for the activation function and loss. 
# You can design your own neural network with your choice of number of hidden layers, learning rate 
# and activation function so that the loss is minimal and the model gives you the best fit.  
# For this problem, you can use pandas, MLP classifier package from sklearn.neural_network or other python package of your choice. 
# Use the housing.csv data set for your analysis where ocean_proximity is your label.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # Load the data
    data = pd.read_csv('hw4\housing.csv')
    data = data.dropna()
    data = data.drop(['ocean_proximity'], axis=1)

    # Normalize the data
    data = (data - data.mean()) / data.std()

    # Split the data into training and testing
    X = data.drop(['median_house_value'], axis=1)
    y = data['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 80-20 split, random_state for reproducibility

    # Add bias to the input
    X_train['bias'] = 1
    X_test['bias'] = 1

    # Initialize the weights
    np.random.seed(0) # For reproducibility
    weights = np.random.rand(X_train.shape[1])

    # Store the initial, middle and final weights
    init_weights = weights.copy()

    # Train the model
    learning_rate = 0.01
    for i in range(1000):
        # Forward pass
        y_pred = np.dot(X_train, weights) # Linear regression
        loss = np.mean((y_pred - y_train) ** 2) # MSE

        # Backward pass
        gradient = np.dot(X_train.T, y_pred - y_train) / X_train.shape[0]
        weights -= learning_rate * gradient # Update weights

        print(f'Iteration {i}: Loss: {loss}, Weights: {weights}')

        if(i == 500): # Store the middle weights
            mid_weights = weights.copy()

    final_weights = weights 

    # Test the model
    y_pred = np.dot(X_test, weights)
    loss = np.mean((y_pred - y_test) ** 2)
    print(f'Test Loss: {loss}')

    # Print intial, middle and final weights
    print("Backpropagation Weight Updates: ")
    print(f'Initial Weights: {init_weights}')
    print(f'Middle Weights: {mid_weights}')
    print(f'Final Weights: {final_weights}')

if __name__ == "__main__":
    main()