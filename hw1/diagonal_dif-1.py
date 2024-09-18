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

def num_rows():
    rows = input("Input number of rows & columns: ")
    print(rows)
    return int(rows)

def matrix_diag_abs_dif():
    matrix = []
    rows = num_rows()
    print("Enter rows with numbers seperated by spaces")
    for i in range(rows):
        row_input = input("Enter row " + str(i+1) + ": ").split(" ")
        matrix.append([int(element) for element in row_input])

    print("Entered matrix: " + str(matrix))
    left_diag = np.array(matrix)

    # Need to flip the matrix around to get the right diagonal
    flipped_matrix = np.flip(matrix, axis=1)
    right_diag = np.array(flipped_matrix)

    print("Left Diagonal: " + str(diagonal_dif(left_diag)))
    print("Right Diagonal: " + str(diagonal_dif(right_diag)))

    abs_dif = abs(diagonal_dif(left_diag) - diagonal_dif(right_diag))
    return abs_dif

def diagonal_dif(a):
    return np.trace(a)
    
#Input must be in this format: [[1,2,3],[4,5,6],[9,8,9]]
print("Absoulte Difference: " + str(matrix_diag_abs_dif()))
input("Press Enter to exit")