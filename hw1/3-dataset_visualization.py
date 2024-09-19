# Problem 3
'''You will download iris dataset from this site https://archive.ics.uci.edu/dataset/53/iris
Using python convert .data files into .csv.
Utilize matplotlib to select a graphical representation eg. Scatter plot, 
box and whiskers [of your choice] that best explain the features of the data set.
Write why you think that graphical representation is appropriate. 
Upload your .csv, .py and all other files required to run the program.
 You will be providing the evaluator with clear instructions on how to accept the input in a separate readme file with problem no.
'''

import matplotlib as mp
import numpy as np
import csv

# Conver iris.data to a .csv
with open('iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
    read = csv.reader(in_file)
    with open('iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

# Utilize matplotlib to select a graphical representation eg. scatter plot, box and whsikers, {chose}, that best explain the features of the data set
# write why you think that graphical represntation is approriate
# upload all your .csv, .py and all other files required to run program