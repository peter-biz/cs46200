Assignment 4 due 10th November. 
This Assignment contains 12 % of the total grade [6 % each]. 
The problems needed to be solved using python.
Mention to the grader how you want her to take the input in the readme file. 
You will be submitting all the files required to run your program. 
Each problem will be evaluated on 20 points. 
Contact: Andrea George [grader] for any questions
1.	Only using NumPy, Matplotlib package solve the following problem:

Obtain the file crash.txt from Brightspace. This file contains measurements from a crash test dummy during an NHSA automobile crash test. Column 0 measures time in milliseconds after the crash event and column 1 represents the acceleration measured by a sensor on the dummy head. The file is formatted according to Numpys default specification for textual data (spaces delineate columns, newlines delineate rows). Thus it can easily be read into a matrix via the following:
 data = numpy.loadtxt(crash.txt)
 Divide this data into a training and test set of equal size even-numbered rows to the training set and odd-numbered rows to test. Write a function to fit the line using Least Square model with the formulas of slope and intercept from your textbook [Page no. 677]. 
Plot the training data using the above model function with the lowest SSE on the training
 data. Repeat the same for the Test Data. Calculate the RMS error for both. 

2.	For this problem, you need to print the backpropagation weight updates for the activation function and loss. You can design your own neural network with your choice of number of hidden layers, learning rate and activation function so that the loss is minimal and the model gives you the best fit.  For this problem, you can use pandas, MLP classifier package from sklearn.neural_network or other python package of your choice. 
Use the housing.csv data set for your analysis where ocean_proximity is your label.



 

