""" 1679. You are given an integer array nums and an integer k.
In one operation, you can pick two numbers from the array whose sum equals k and remove them from the array.
Return the maximum number of operations you can perform on the array.
Input: nums = [3,1,3,4,3], k = 6
Output: 1
"""

def maxOperations(nums, k):
    #We are going to be using 2 pointer approach where you will sorting the nums array first
    nums.sort()
    left, right = 0, len(nums) - 1
    max_operations = 0

    if len(nums) == 1 and nums[0] == k:
        return 1
    
    while left < right:
        total = nums[left] + nums[right]
        if total < k:
            left += 1
        elif total > k:
            right -= 1
        else:
            max_operations += 1
            left += 1
            right -= 1
    return max_operations

nums = [1,2,3,4]
k = 5
print(f"max operations you can perform the array is: {maxOperations(nums, k)}")

"""
2352. Given a 0-indexed n x n integer matrix grid, return the number of pairs (ri, cj) such that row ri and column cj are equal.
A row and column pair is considered equal if they contain the same elements in the same order (i.e., an equal array).

Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
Output: 1
Explanation: There is 1 equal row and column pair:
- (Row 2, Column 1): [2,7,7]
"""
def equalPairs(grid):
    total_count = 0
    rows, cols = len(grid), len(grid[0])

    for c in range(cols):
        temp = []
        for r in range(rows):
            temp.append(grid[r][c])
        if temp in grid:
            total_count += grid.count(temp)
    return total_count

grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
print(f"Number of equal pairs is: {equalPairs(grid)} ")

"""
73. Set Matrix Zeroes
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
"""
def setZeroes(mat):
    row_set = set()
    col_set = set()

    rows, cols = len(mat), len(mat[0])

    #Finding which row and which col has 0
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                row_set.add(r)
                col_set.add(c)

    
    #Converting the entire row to 0
    for z in row_set:
        for c in range(cols):
            mat[z][c] = 0

    #Converting the entire col to 0
    for z in col_set:
        for r in range(rows):
            mat[r][z] = 0

    return mat 

matrix = [[1,1,1],[1,0,1],[1,1,1]]
print(f"After setting to zero {setZeroes(matrix)}")

"""
867. Given a 2D integer array matrix, return the transpose of matrix.

The transpose of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]
"""
def transposeMatrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = []

    for c in range(cols):
        temp = []
        for r in range(rows):
            temp.append(matrix[r][c])
        if temp:
            result.append(temp)
    return result


matrix = [[1,2,3],[4,5,6],[7,8,9]]    
print(f"Transposed matrix is {transposeMatrix(matrix)}")

"""
1975. You are given an n x n integer matrix. You can do the following operation any number of times:

Choose any two adjacent elements of matrix and multiply each of them by -1.
Two elements are considered adjacent if and only if they share a border.

Your goal is to maximize the summation of the matrix's elements. Return the maximum sum of the matrix's elements using the operation mentioned above.
Input: matrix = [[1,-1],[-1,1]]
Output: 4
"""
def maxMatrixSum(matrix):
    n = len(matrix)
    min_val = float('inf')
    res = 0
    count = 0

    for r in range(n):
        for c in range(n):
            res += abs(matrix[r][c])

            if matrix[r][c] < 0:
                count += 1
            
            min_val = min(min_val, abs(matrix[r][c]))
    
    if count % 2 == 0:
        return res
    else:
        return res - 2 * min_val

matrix = [[1,-1],[-1,1]]
print(f"Output of the matrix is {maxMatrixSum(matrix)}")

"""
74. Search in a 2D matrix: You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
"""
def searchMatrix(matrix, target):
    #Binary Search every row

    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        left = 0
        right = cols - 1

        while left <= right:
            mid = left + (right - left) // 2
            if matrix[i][mid] == target:
                return True
            elif matrix[i][mid] < target:
                left = mid + 1
            else:
                right = mid - 1
    return False
            
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target = 20
print(f"Target in 2D matrix : {searchMatrix(matrix, target)}")