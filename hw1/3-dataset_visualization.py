# Problem 3 - Peter Bizouaks - Python version 3.12
'''You will download iris dataset from this site https://archive.ics.uci.edu/dataset/53/iris
Using python convert .data files into .csv.
Utilize matplotlib to select a graphical representation eg. Scatter plot, 
box and whiskers [of your choice] that best explain the features of the data set.
Write why you think that graphical representation is appropriate. 
Upload your .csv, .py and all other files required to run the program.
 You will be providing the evaluator with clear instructions on how to accept the input in a separate readme file with problem no.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# Conver iris.data to a .csv
with open('.\iris.data', 'r') as in_file: # Open data.iris with 'r' parameter(read)
    read = csv.reader(in_file)
    with open('.\iris.csv', 'w') as out_file:
        write = csv.writer(out_file)
        # Set column names
        write.writerow(('Septal Length', 'Septal Width', 'Petal Length', 'Petal Width', 'Species'))
        for row in read:
            write.writerow(row)

# Plot style
plt.style.use('_mpl-gallery')

# Read csv data from iris.csv
iris = pd.read_csv('.\iris.csv')

colors = ['blue', 'red', 'green'] # Colors for species
species = iris['Species'].unique() # Get unique species

# Figure size
plt.figure(figsize=(10,6))

# Loops thru species and plots the respective species with the respective color
for i, sp in enumerate(species):  
    subset = iris[iris['Species'] == sp] # Create a subset of the data for curr species 
    plt.scatter(subset['Petal Length'],  # X-axis
                subset['Petal Width'],  # Y-axis
                marker='o', # Uses circle markers on plot
                color=colors[i % len(colors)],  # Assign color to marker
                label=sp) # Applies species name to legend
    
# Plot titles/labels    
plt.title('Petal Length vs. Petal Width by Species')
plt.xlabel('Petal Length(cm)')
plt.ylabel('Petal Width(cm)') 
plt.gcf().canvas.manager.set_window_title('Iris Petal Scatterplot') # Changes the plot window title cause i didn't like that it just said "Figure 1"


plt.legend() # Species Legend
plt.grid() # Turns the gridlines off
plt.tight_layout() # Makes sure the plot stays within the plot window
plt.show() # Show scatterplot

# Reason why I think this graphical represenation is appropriate: 
# I believe this scatter plot is a good way to show off the petal data of the iris dataset
# because it shows in a clear and readable way the petal length vs petal width by species on the plot. 
