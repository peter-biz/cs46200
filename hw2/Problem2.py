# Problem #2 - Peter Bizoukas, Python 3.8
# 2.	Using the iris data set of assignment 1 [https://archive.ics.uci.edu/dataset/53/iris,
# using python convert .data files into .csv.] split it into 
# a.	80% train and 20% test data
# b.	70% train and 30% test data 
# c.	Compare the accuracy, specificity and sensitivity of a and b and with the ROC curve 
# mention which is a better model. The grader should be able to generate the curve and the stats after running your program. 


import numpy as np
import scipy as sp
import matplotlib as plt


# Conver iris.data to a .csv
with open('.\iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
    read = csv.reader(in_file)
    with open('.\iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

