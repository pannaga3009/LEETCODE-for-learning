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