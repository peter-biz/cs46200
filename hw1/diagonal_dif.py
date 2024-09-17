# Problem 1
'''Given a square matrix, calculate the absolute difference between the sums of its diagonals.
For example, the square matrix is shown below:
1 2 3
4 5 6
9 8 9  

The left-to-right diagonal 1 + 5+ 9 = 15 . The right to left diagonal 3 + 5+ 9= 17. 
Their absolute difference is |15 – 17| = 2. 

# Your program’s outcome should be an INTEGER |x| i.e. the absolute diagonal difference.
# The matrix could have both positive and negative numbers. 
# You will be using numpy to solve the problem.
# Array size m x n must be less than 100 x 100. 

The array should be based on user input. 
You will be providing the evaluator with clear instructions on how to accept 
the input in a separate readme file with problem no. 
'''

import numpy as np

def is_square(a):
    m = np.matrix(a)
    return m.shape[0] == m.shape[1]

def diagonal_dif(a):
    if(is_square(a)):
        return(np.trace(a))
    else:
        return("Error, not a square matrix.")

    
#Input must be in this format: [[1,2,3],[4,5,6],[9,8,9]]
a = np.array(input("Enter the matrix: "))
print(a)
print(diagonal_dif(a))
input("Press Enter to exit")