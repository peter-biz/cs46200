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

def matrix_diag_abs_dif():
    matrix = [] # Initialize matrix

    try:
        rows = input("Input number of rows & columns: ") # Number of rows & columns, this also ensures it's square
        rows = int(rows) # Converts string to int
        if rows > 100:
            print("Error: Matrix size should be less than 100x100.")
            matrix_diag_abs_dif() # If matrix size is wrong, calls function again

        print("!!!Enter rows with numbers seperated by spaces. Ex: 'Enter row 1: 1 2 3'!!!") # Input instructions
        for i in range(rows):
            row_input = input("Enter row " + str(i+1) + ": ").split(" ") # Removes the spaces from the string
            if len(row_input) != rows: # Checks if the input is enterered incorrectly
                raise ValueError(f"Error: row does not have correct number of elements or has an invalid input.\n Ex: 'Enter row 1: 1 2 3'.")            
            else:
                matrix.append([int(element) for element in row_input]) # Adds elements of row_input to the matrix

        print("Entered matrix: " + str(matrix))
        left_diag = np.array(matrix) # Top left to bottom right diagonal 

        # Need to flip the matrix around to get the right diagonal
        flipped_matrix = np.flip(matrix, axis=1)
        right_diag = np.array(flipped_matrix) # Top right to bottom left diagonal

        print("Left Diagonal: " + str(diagonal_dif(left_diag)))
        print("Right Diagonal: " + str(diagonal_dif(right_diag)))

        abs_dif = abs(diagonal_dif(left_diag) - diagonal_dif(right_diag)) # Absolute difference
        print("Absoulte Difference: " + str(abs_dif))
        input("Press enter to exit.")
    except ValueError as e: # Error check
        print(e)
        matrix_diag_abs_dif()

def diagonal_dif(a): # Gets the diagonal of a given array
    return np.trace(a)

def main(): # main
    matrix_diag_abs_dif() # First call of function to start the program

if __name__=="__main__":
    main() # Calls main