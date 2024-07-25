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

"""
1329. A matrix diagonal is a diagonal line of cells starting from some cell in either the topmost row or leftmost column and going in the bottom-right direction until reaching the matrix's end. For example, the matrix diagonal starting from mat[2][0], where mat is a 6 x 3 matrix, includes cells mat[2][0], mat[3][1], and mat[4][2].

Given an m x n matrix mat of integers, sort each matrix diagonal in ascending order and return the resulting matrix.

mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
"""
def diagonalSort(matrix):
    #First understand where does the diagonal start from, looking at this problem take the topmost row and leftmost col
    
    sorted_diagonal = []
    rows, cols = len(matrix), len(matrix[0])

    #Getting the leftmost col
    for r in range(rows):
        sorted_diagonal.append([r, 0])

    #Getting the topmost row
    for c in range(1, cols):
        sorted_diagonal.append([0, c])

    print(sorted_diagonal)
    for start_row, start_col in sorted_diagonal:
        diagonal_elements = []
        row, col = start_row, start_col
        while row < rows and col < cols:
            diagonal_elements.append(matrix[row][col])
            row, col = row + 1, col + 1

            diagonal_elements.sort()

            row, col = start_row, start_col
            for element in diagonal_elements:
                matrix[row][col] = element
                row, col = row + 1, col + 1
    return matrix

mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
#Output: [[1,1,1,1],[1,2,2,2],[1,2,3,3]]
print(f" After sorting diagonally {diagonalSort(mat)}")

"""
1030. You are given four integers row, cols, rCenter, and cCenter. There is a rows x cols matrix and you are on the cell with the coordinates (rCenter, cCenter).

Return the coordinates of all cells in the matrix, sorted by their distance from (rCenter, cCenter) from the smallest distance to the largest distance. You may return the answer in any order that satisfies this condition.

The distance between two cells (r1, c1) and (r2, c2) is |r1 - r2| + |c1 - c2|.
Input: rows = 1, cols = 2, rCenter = 0, cCenter = 0
Output: [[0,0],[0,1]]
Explanation: The distances from (0, 0) to other cells are: [0,1]

"""
def allCellsDistOrder(rows, cols, rCenter, cCenter):
    # mat = [[i, j] for i in range(rows) for j in range(cols)]
    # sorted_dist = sorted(mat, key = lambda x: abs(x[0] - rCenter) + abs(x[1] - cCenter))
    # return sorted_dist
    mat = []
    for i in range(rows):
        for j in range(cols):
            dist = abs(i - rCenter) + abs(j - cCenter)
            mat.append([i, j, dist])
    
    mat.sort(key = lambda x: x[2])

    for cords in mat:
        cords.pop()
    
    return mat

rows = 2
cols = 2
rCenter = 0
cCenter = 1
#Output: [[0,1],[0,0],[1,1],[1,0]]
print(f"All cells distance is {allCellsDistOrder(rows, cols, rCenter, cCenter)}")

"""
2679. Sum in a matrix
You are given a 0-indexed 2D integer array nums. Initially, your score is 0. Perform the following operations until the matrix becomes empty:

From each row in the matrix, select the largest number and remove it. In the case of a tie, it does not matter which number is chosen.
Identify the highest number amongst all those removed in step 1. Add that number to your score.
Return the final score.

Input: nums = [[7,2,1],[6,4,2],[6,5,3],[3,2,1]]
Output: 15
"""
def sumMatrix(nums):
    for r in range(len(nums)):
        nums[r].sort(reverse=True)
   
    rows, cols = len(nums), len(nums[0])
    total = 0
    max_val = 0
    for c in range(cols):
        temp = []
        for r in range(rows):
            temp.append(nums[r][c])
        
        if temp:
            max_val = max(temp)
            total += max_val
    return total


nums = [[7,2,1],[6,4,2],[6,5,3],[3,2,1]]
print(f"Sum in a matrix is {sumMatrix(nums)}")

def mergesort(li):
    """
    Practising merge sort to understand and dive deep into devide and conquer
    where once you devide the problem to subproblem, you recursively solve it
   
    """

    #Divide until the length of array is less than or equal to 1
    if len(li) <= 1:
        return li

    n = len(li)
    mid = n // 2
    left_arr = li[:mid]
    right_arr = li[mid:]

    left_arr = mergesort(left_arr)
    right_arr = mergesort(right_arr)

    left_index, right_index = 0, 0
    data_index = 0
    result = []

    while left_index < len(left_arr) and right_index < len(right_arr):
        if left_arr[left_index] < right_arr[right_index]:
            result.append(left_arr[left_index])
            left_index += 1
            data_index += 1
        else:
            result.append(right_arr[right_index])
            right_index += 1
            data_index += 1
    
    while left_index < len(left_arr):
        result.append(left_arr[left_index])
        left_index += 1
        data_index += 1
    
    while right_index < len(right_arr):
        result.append(right_arr[right_index])
        right_index += 1
        data_index += 1

    return result

#Testing mergesort logic 
print(f"Mergesort = {mergesort([1,3,4,5,2,6])}")   

def TwoSum(nums, target):
    """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
    """
    #Using a HashMap to map indices with values
    numsMap = {}
    output = [] #output array to store all the target indices

    for i in range(len(nums)):
        diff = (target - nums[i])
        if diff in numsMap:
            output.append(numsMap[diff])
            output.append(i)
            return output
        else:
            numsMap[nums[i]] = i


nums = [2,7,11,15]
target = 9
print(f"Output array would be {TwoSum(nums, target)}")

def TwoSumSorted(nums, target):
    """
    Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.
The tests are generated such that there is exactly one solution. You may not use the same element twice.
Your solution must use only constant extra space.
Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

    """
    #For this, we can use a two pointer solution because the array is already sorted
    left = 0
    right = len(nums) - 1
    output = []

    while left < right:
        result = nums[left] + nums[right]
        if result < target:
            left += 1
        elif result > target:
            right -= 1
        else:
            output.append([left + 1, right + 1])
            return output

numbers = [2,7,11,15]
target = 9
print(f"Two sum in a sorted array {TwoSumSorted(numbers, target)}")

def reformatDate(date):
    """
    Given a date string in the form Day Month Year, where:

Day is in the set {"1st", "2nd", "3rd", "4th", ..., "30th", "31st"}.
Month is in the set {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}.
Year is in the range [1900, 2100].
Convert the date string to the format YYYY-MM-DD, where:

YYYY denotes the 4 digit year.
MM denotes the 2 digit month.
DD denotes the 2 digit day.Input: date = "20th Oct 2052"
Output: "2052-10-20"
    """
    #take the date string, split it based on white spaces and addd it to an array
    month_map = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug":"08", "Sep": "09", "Oct":"10", "Nov": "11", "Dec": "12"}
    arr_date = date.split(" ")

    day = arr_date[0][:-2]
    month = month_map[arr_date[1]]
    year = arr_date[2]

    if int(day) < 10:
        day = "0" + day
    
    return f"{year}-{month}-{day}"
    
date = "20th Oct 2052"
print(f" Date is {reformatDate(date)}")

def minimumSwap(s1, s2):
    """
You are given two strings s1 and s2 of equal length consisting of letters "x" and "y" only. Your task is to make these two strings equal to each other. You can swap any two characters that belong to different strings, which means: swap s1[i] and s2[j].
Return the minimum number of swaps required to make s1 and s2 equal, or return -1 if it is impossible to do so.
Example 1:

Input: s1 = "xx", s2 = "yy"
Output: 1
Explanation: Swap s1[0] and s2[1], s1 = "yx", s2 = "yx".
    """
    count_x = 0
    count_y = 0

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if c1 == "x":
                count_x += 1
            else:
                count_y += 1
        
    
    if (count_x + count_y) % 2 != 0:
        return -1 

    return (count_x + 1 // 2) + (count_y + 1 // 2)


print("Minimum swap required ", minimumSwap("xx", "yy")) 

def lengthOfLongestSubstring(s):
    """
    Given a string s, find the length of the longest 
substring
 without repeating characters.
Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:
    """
    char_set = set()
    #Have a set to store all the characters visited
    max_length = 0
    left = right = 0

    while left < len(s) and right < len(s):
        if s[right] not in char_set:
            char_set.add(s[right])
            right += 1
            max_length = max(max_length, right - left)
        else:
            char_set.remove(s[left])
            left += 1
    
    return max_length

print("Length of the longest substring is ", lengthOfLongestSubstring("abcabcbb"))
        
class Solution:
    def isSolvable(self, words:[str], result: str) -> bool:
        # Append the result to the list of words
        words.append(result)
        # Get the number of rows (words) and the maximum length of a word
        R, C = len(words), max(map(len, words))
        # Reverse each word and convert them to lists
        words = list(map(list, map(reversed, words)))

        # Dictionary to store mappings between letters and digits
        letters = {}
        # List to keep track of which digits are assigned to letters
        nums = [None] * 10

        # Backtracking function to solve the equation
        def walk(r, c, carry):
            # If reached end of a column
            if c >= C:
                # Check if carry is zero
                return carry == 0
            # If reached end of all rows
            if r == R:
                # Check if carry is zero and if it's the last column
                if carry % 10 != 0:
                    return False
                return walk(0, c + 1, carry // 10)

            # Get the current word and its length
            word = words[r]
            W = len(word)
            # If reached end of current word
            if c >= W:
                # Move to the next row
                return walk(r + 1, c, carry)

            # Get the current character
            l = word[c]
            # Determine the sign based on whether it's the result word or not
            sign = -1 if r == R - 1 else 1

            # If the character is already mapped
            if l in letters:
                n = letters[l]
                # Check if leading zero is assigned when the word has multiple digits
                if n == 0 and W > 1 and c == W - 1:
                    return False
                return walk(r + 1, c, carry + sign * n)
            else:
                # Try assigning each unused digit to the character
                for n, v in enumerate(nums):
                    if not v:
                        # Check if leading zero is assigned when the word has multiple digits
                        if n > 0 or W == 1 or c != W - 1:
                            # Assign digit to character
                            nums[n] = l
                            letters[l] = n

                            # Recur with updated carry
                            if walk(r + 1, c, carry + sign * n):
                                return True

                            # Undo assignment
                            nums[n] = None
                            del letters[l]
            return False

        # Start backtracking from the first character
        return walk(0, 0, 0)  

def KthSmallestElment(nums, k):
    """
    We can use heap to get the smallest element, by default heapq is min heap
    """
    heap_arr = []
    import heapq
    
    for n in nums:
        heapq.heappush(heap_arr, n)

    for _ in range(1, k):
        heapq.heappop(heap_arr)
    
    x = heapq.heappop(heap_arr)
    return x

nums = [22,33,77,11,90]
print("Kth smallest element",KthSmallestElment(nums, 4))

import heapq
def KthLargestElement(nums, k):
    """
    Brute force: arr.sort()
    index_k = len(arr) - k
    return arr[index_k]
    
    Using heap - n log k
    import heapq
    heap = []

    for n in nums:
        heapq.heappush(heap, n)

        if len(heap) > k:
            heapq.heappop(heap)

    x = heapq.heappop(heap)
    return x

    QuickSelect - average case result = O(n)
    """
    k = len(nums) - k
    def quickSelect(l, r):
        for i in range(l, r):
            p, pivot = l, nums[r]
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
    
        nums[p], nums[r] = nums[r], nums[p]
    
        if p > k: return quickSelect(l, p - 1)
        elif p < k: return quickSelect(p + 1, r)
        else:  return nums[p]

    return quickSelect(0, len(nums)-1)
        
        
nums = [3,2,1,5,6,4]
print("Kth largest element",KthLargestElement(nums, 4))

def leetcode347(nums, k):
    """
Input: nums = [1,1,1,1,2,3,3], k = 2
Output: [1,2]

dict_freq = {1:3, 2:1, 3:2}
can use sorted function - n log n
or can use heapq - n log k

    """
    dict_freq = {}
    for n in nums:
        dict_freq[n] = dict_freq.get(n, 0) + 1

    heap = []
    for key, val in dict_freq.items():
        if len(heap) < k:
            heapq.heappush(heap, (val, key))
        else:
            heapq.heappushpop(heap, (val, key))

    return [h[1] for h in heap]


    # sorted_output = sorted(dict_freq.keys(), key = lambda x: dict_freq[x], reverse=True)
    # return sorted_output[:k]

nums = [1,1,1,1,2,2,2,2,2,3,3]
k = 1
print(leetcode347(nums, k))

def leetcode33(nums, target):
    """
Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
    """
    left, right = 0, len(nums) - 1

    while left<=right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]: #Assuming left is sorted
            if nums[left] <= target < nums[mid]: #assuming target comes in the left
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


nums = [4,5,6,7,0,1,2]
target = 1
print(leetcode33(nums, target))

def leetcode36(board):
    """
    Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
 
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            if board[r][c] == ".":
                continue

            box_index = (r // 3) * 3 + (c // 3)
            if (board[r][c] in rows[r] or board[r][c] in cols[c] 
                or board[r][c] in boxes[box_index]):
                return False
            rows[r].add(board[r][c])
            cols[c].add(board[r][c])
            boxes[box_index].add(board[r][c])
    return True

board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
print("Executed Sudoku", leetcode36(board))

def numIslandsDFS(grid):
    """
    two types of approach -> DFS, stack and BFS, dequeue

    """
    rows = len(grid)
    cols = len(grid[0])
    directions = [(1, 0), (0,1), (-1, 0), (0, -1)]
    numIsland = 0
    visited = set()

    def dfs(r, c):
        stack = [(r, c)]
        visited.add((r, c))

        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if(0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == "1" and (nx, ny) not in visited):
                    stack.append((nx, ny))
                    visited.add((nx, ny))


    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                numIsland += 1
                dfs(r, c)
    return numIsland

from collections import deque
def numIslandsBFS(grid):
    """
    deque since we are going every level
    """
    visited = set()
    rows = len(grid)
    cols = len(grid[0])
    numOfIsland = 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def bfs(r, c):
        q = deque([(r, c)])
        visited.add((r,c))

        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if(0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == "1" and (nx, ny) not in visited):
                    q.append((nx, ny))
                    visited.add((nx, ny))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                numOfIsland += 1
                bfs(r, c)
    return numOfIsland



      
grid =[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
print(f"numIslandsDFS = {numIslandsDFS(grid)}")
print(f"numIslandsBFS = {numIslandsBFS(grid)}")


def partitionString(s):
    """
    Given a string s, partition the string into one or more substrings such that the characters in each substring are unique. That is, no letter appears in a single substring more than once.
    Return the minimum number of substrings in such a partition.
    Note that each character should belong to exactly one substring in a partition.
    Input: s = "abacaba"
Output: 4
Explanation:
Two possible partitions are ("a","ba","cab","a") and ("ab","a","ca","ba")
    """
    unique_char = set()
    ans = 0
    for i in s:
        if i in unique_char:
            unique_char = set()
            ans += 1
        unique_char.add(i)
    return ans + 1


def partitionStringSol2(s):
    unique_char = set()
    result = []
    text = ""

    left, right = 0, 0
    while left >= 0 and right < len(s):
        if s[right] in unique_char:
            unique_char = set()
            result.append(text)
            left = right
            text = ""
        else:
            unique_char.add(s[right])
            text = s[left:right + 1]
            right += 1
    
    if text != "":
        result.append(text)

    return len(result)



        
s = "abacaba"
print("partitionString: ", partitionString(s))
print("partitionString: ", partitionStringSol2(s))


def leet1492(n, k):
    """
    You are given two positive integers n and k. A factor of an integer n is defined as an integer i where n % i == 0.  
    Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.
    """
    result = []
    for i in range(1, n + 1):
        if n % i == 0:
            result.append(i)
            if len(result) == k:
                return result[k-1]
    return -1
        
    
n = 12
k = 3
print("K the factor  ", leet1492(n, k))

def leetcode322(tickets):
    """
    You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.
    All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

    For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
    You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.
    Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
    Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.
    """

    res = ["JFK"]
    adj = { src: [] for src, dst in tickets}

    #Lexical order (alphabets in aseceding order)
    tickets.sort()

    #Adding the src and destination from the tickets to the dict adj
    for src, dst in tickets:
        adj[src].append(dst)

    #Lets do a DFS traversal to go through all the nodes
    def dfs(src):
        if len(res) == len(tickets) + 1:
            return True
        if src not in adj:
            return False
        
        #Intializing a temporary list to check the destination of sources
        temp = list(adj[src])
        for i, v in enumerate(temp):
            adj[src].pop(i)
            res.append(v)
            if dfs(v):
                return True
            adj[src].insert(i, v)
            res.pop()
        return False

    dfs("JFK")
    return res


tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
print("Reconnect itinerary ",leetcode322(tickets))

def leetcode261(n, edges):
    """
    You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.
    Return true if the edges of the given graph make up a valid tree, and false otherwise.
    n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
    """

    if not n:
        return True

    adj = { i: [] for i in range(n)}

    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)

    visited = set()
    def dfs(i, prev):
        if i in visited:
            return False
        
        visited.add(i)
        for j in adj[i]:
            if j == prev:
                continue

            if not dfs(j, i):
                return False
        return True

    return dfs(0, -1) and len(visited) == n

edges = [[0,1],[0,2],[0,3],[1,4]]
n = 5
print("Graph Valid Tree - ", leetcode261(n,edges))


    

    
        



    