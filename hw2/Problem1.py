# Problem #1 - Peter Bizoukas, Python 3.8

# 1.	Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, 
# return the k closest points to the origin (0, 0).
# The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).
# You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).                                                                                                                                                                                                                                         
# Example 1:
# Input: points = [[1,3],[-2,2]], k = 1
# Output: [[-2,2]]
# Explanation:
# The distance between (1, 3) and the origin is sqrt(10).
# The distance between (-2, 2) and the origin is sqrt(8).
# Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
# We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
# Example 2:
# Input: points = [[3,3],[5,-1],[-2,4]], k = 2
# Output: [[3,3],[-2,4]]
# Explanation: The answer [[-2,4],[3,3]] would also be accepted.
#  These are examples. The problem is user input based. The grader can test with any input of their choice. 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



def closest_points(pts, k):
    print(pts, k)


def main():
    points = []
    try:
        print("Enter points in the format 'x y', with commas seperating each point. Ex: '1 3, -2 2'")
        pts = input("Input points: ")
        pts = pts.split(", ")
        for i in pts:
            points.append(i.split(" "))
        print(points)
        k = int(input("Input k: "))
        closest_points(points, k)
    except ValueError as e:
        print(e)
        main()

if __name__=="__main__":
    main()