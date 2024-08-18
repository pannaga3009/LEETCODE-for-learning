"""


palindrome - > 121
string[::-1]
"""
def palindrome(string):
    return string == string[::-1]

# string_entered = input("Enter the input: ")
# print(palindrome(string_entered))

def twoSum(nums, target):
    """
    nums = [2, 7, 11, 15]
    target = 9
    O(n) 
    hashmap
    """
    dict_index = {
    }
    res = []

    for i in range(len(nums)):
        diff = target - nums[i]
        if diff in dict_index:
            res.append(dict_index[diff])
            res.append(i)
            return res
        else:
            dict_index[nums[i]] = i



print("two sum", twoSum([7, 11, 15, 2], 9))
    

class ListNode:
    def __init__(self, value = 0, next = None):
        self.value = value
        self.next = next
    
def findMiddle(head):
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = slow.next.next

    return slow

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)

middle = findMiddle(head)
print(middle.value)

def findDuplicates(arr):
    seen = set()
    duplicates = []

    for n in arr:
        if n in seen:
            duplicates.append(n)
        else:
            seen.add(n)

    return duplicates

def dropDuplicates(arr):
    """Converting the array to a set has a time complexity of O(n) due to iterating through each element once.
Converting the set back to a list also has a time complexity of O(n) due to iterating through each element once.
    """
    return list(set(arr))


arr = [1, 2, 3, 4, 5, 1,1, 3, 6]
print(f"Finding duplicates {findDuplicates(arr)}")
print(f"After droping the duplicates {dropDuplicates(arr)}")

def longestCommonPrefix(strings):
    """
    1. smallest string
    2. enumerate to go through each index and character and see if they are equal
    Time Complexity: O(n * m)
min_str = min(strings, key = len) has a time complexity of O(n), where n is the number of strings and m is the length of the smallest string.
The nested loops have a time complexity of O(n * m), where n is the number of strings and m is the length of the smallest string.
    """
    min_str = min(strings, key = len)
    for i, char in enumerate(min_str):
        for s in strings:
            if s[i] != char:
                return min_str[:i]



strs = ["flower","flow","flight"]
print(longestCommonPrefix(strs))

def longestSubstring(s):
    """

     s = "abcabcbb"
     1. using set to keep track of unique characters
     2. using two pointers left and right

    """
    char_set = set()
    left, right = 0, 0
    longest_len = 0

    while left < len(s) and right < len(s):
        if s[right] not in char_set:
            char_set.add(s[right])
            longest_len = max(longest_len, (right - left) + 1)
            right += 1
        else:
            char_set.remove(s[left])
            left += 1
    return longest_len
        
def longestPalindrome(string):
    """
    1. expandAroundCentre function
    2. left and right pointer inside it
    3. even palindrome and odd palindrome
    """
    longest = ""

    def expandAroundCentre(s, left, right):
        if left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1: right]
            

    for i in range(len(string)):
        oddPalindrome = expandAroundCentre(string, i, i )

        if len(oddPalindrome) > len(longest):
            longest = oddPalindrome
        
        evenPalindrome = expandAroundCentre(string, i, i + 1)
        
        if len(evenPalindrome) > len(longest):
            longest = evenPalindrome
    
    return longest

def longestPalindromeBruteF(string):

    """
    The space complexity is O(1) because we're not using any additional space that grows with the input size.
    Therefore, the overall time complexity is O(n^3), where n is the length of the input string, and we're iterating through each character and each possible substring, and then checking if each substring is a palindrome.
    """
    
    longest = ""
    def isPalindrome(s):
        return s == s[::-1]
    
    for i in range(len(string)):
        for j in range(i + 1, len(string) + 1):
            substr = string[i:j]
            if isPalindrome(substr) and len(substr) > len(longest):
                longest = substr
    return longest
print(longestSubstring("abcabbcad"))
print(longestPalindrome("cbbd"))
s = "cbbd"
print(f"Brute force approach n^3 : {longestPalindromeBruteF(s)}")

def validParenthesis(s):
    """

    1. stack
    2. using dictionary

    """
    dv = {"(": ")", "[": "]", "{":"}"}
    stack = []

    stack.append(s[0])
    for i in range(1, len(s)):
        if len(stack) > 0 and dv.get(stack[-1]) == s[i]:
            stack.pop()
        else:
            stack.append(s[i])

    return len(stack) == 0
    

print(validParenthesis("()"))

def names_decorator(function):
    def wrapper(arg1, arg2):
        arg1 = arg1.upper()
        arg2 = arg2.upper()
        string_hello = function(arg1, arg2)
        return string_hello
    return wrapper
    
@names_decorator
def sayHello(name1, name2):
    return 'Hello ' + name1 + "! Hello " + name2 + "!"
    
print(sayHello('Pans','Harish'))


#Opening and closing a file
file = open('example.txt', 'r')
content = file.read()
print("From the content",content)


file.close()


with open('example.txt', 'r') as file:
    content = file.read()
    print(content)


with open('example.txt', 'a') as file:
    content = file.write("I am getting through this no matter what")
    print(content)

with open('example.txt', 'w') as file:
    content = file.write("Congrats on your job")
    print(content)

def pickle(data):
    """
    
    Pickling is the process of converting a Python object into a byte stream.
This byte stream can be stored in a file, transmitted over a network, or used in some other way.
    """
    import pickle

    with open('data.pickle', 'wb') as file:
        pickle.dump(data, file)

data = {'name': 'Pannaga', 'age' : 26}
pickle(data)


def unPickle():
    import pickle
    with open('data.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
    
    print(loaded_data)

unPickle()

def NumOfIslandsRecursive(grid):
    """
    dfs approach
    will have visited set
    directions ([-1, 0], [1, 0], [0, 1], [0, -1])
    """
    rows = len(grid)
    cols = len(grid[0])
    numOfIslands = 0
    directions = [(1,0), (0,1), (-1, 0), (0, -1)]

    def dfs(grid, r, c):
        if 0<=r<rows and 0<=c<cols and grid[r][c] == "1":
            grid[r][c] = "0"        
            for dx, dy in directions:
                dfs(grid, r + dx, c + dy)


    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                numOfIslands += 1
                dfs(grid, r, c)
    return numOfIslands


def numOfIslandsDFS(grid):
    """
    DFS - stack
    visited set for rows and cols
    directions as usual 
    """
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    numOfIslands = 0
    directions = [(0,1), (1,0), (-1, 0), (0, -1)]

    def dfs(r, c):
        stack = [(r,c)]
        visited.add((r,c))

        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0<=nx<rows and 0<=ny<cols and grid[nx][ny] == "1" and (nx, ny) not in visited):
                    stack.append((nx, ny))
                    visited.add((nx, ny))
                    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                numOfIslands += 1
                dfs(r, c)
    return numOfIslands

from collections import deque
def numOfIslandsBFS(grid):
    """
    important thing so remember in bfs is it uses dequeue
    and popping the elements breadthwise meaning FIFO - pop.left
    have visited set to track all the visited rows and cols
    deque to add the rows and cols when encountered with one
    """
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    numOfIslands = 0
    directions = [(0,1),(1,0),(-1, 0),(0,-1)]

    def bfs(r, c):
        q = deque([(r, c)])
        visited.add((r,c))

        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0<=nx<rows and 0<=ny<cols and grid[nx][ny] == "1" and (nx,ny) not in visited):
                    q.append((nx,ny))
                    visited.add((nx,ny))


    for r in range(rows):
        for c in range(cols):
            if (grid[r][c] == "1" and (r,c) not in visited):
                numOfIslands += 1
                bfs(r, c)
    return numOfIslands
    

grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

grid1 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

print("NumOfIslandsRecursive : ", NumOfIslandsRecursive(grid))
print("numOfIslandsDFS : ", numOfIslandsDFS(grid1))
print("numOfIslandsBFS : ", numOfIslandsBFS(grid1))

def mergeIntervals(intervals):
    """
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

    merge = [intervals[0]]
    """

    intervals.sort(key = lambda x:x[0])
    merge = []

    for i in intervals:
        if not merge:
            merge.append(i)
        if merge[-1][1] >= i[0]:
            merge[-1][1] = max(merge[-1][1], i[1])
        else:
            merge.append(i)

    return merge

intervals = [[1,3],[2,6],[8,10],[15,18]]
print("mergeIntervals :", mergeIntervals(intervals))

def canJump(nums):
    """
    You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
    Return true if you can reach the last index, or false otherwise.
    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    """
    last_index = len(nums) - 1
    for i in range(len(nums)-1, -1, -1):
        if i + nums[i] >= last_index:
            last_index = i

    return True if last_index == 0 else False
    
nums = [3,2,1,0,4]
print("canJump --- ", canJump(nums))

def buySellStocks(prices):
    min_price = float("inf")
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit

def buySellStocks2(prices):
    """
    You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
    On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
    Find and return the maximum profit you can achieve.
    Approach:
    Whenever there is a increase in stocks value, check out the difference with the previous value and add it profit, keep iterating untill you get another rise and keep adding profit

    """
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

    
prices = [7,6,4,3,1]
print("Buy Sell Stock --- ", buySellStocks(prices))
prices2 = [7,1,5,3,6,4]
print("Buy sell stocks 2 --- ", buySellStocks2(prices2))


def generateParenthesis(n):
    """
    To generate parenthesis
    It has to be valid meaning only can start from open brackets "(" 
    add to stack
    number of close_brackets < open_brackets
    backtrack so once you reach len which is N * 2 then append to the output
    and pop the stack to see other possibilities and backtracking
    """
    output = []
    stack = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            output.append("".join(stack))
            return
        
        if openN < n:
            stack.append("(")
            backtrack(openN+1, closedN)
            stack.pop()
        
        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN + 1)
            stack.pop()
    
    backtrack(0, 0)
    return output


n = 3
print("Generate parenthesis :", generateParenthesis(n))

def uniquePaths(m, n):
    """
    There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
    Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
    approach:
    1. The way to reach the 1st row and col is always 1 
    2. From the second row, you can always go through right or down , but we can calculate the value how it goes by calculating up and left direction
    """

    grid = [[1] * n for _ in range(m)]

    for x in range(1, m):
        for y in range(1, n):
            grid[x][y] = grid[x-1][y] + grid[x][y-1]
    
    return grid[-1][-1]

m = 3
n = 2
print("uniquePaths ", uniquePaths(m,n))

def compressedArray(arr, k):
    """
    Given an array of integers, a, in one operation one can select any two
    adjacent elements and replace them with their product. This
    operation can only be applied if the product of those adjacent
    elements is less than or equal to k.
    The goal is to reduce the length of the array as much as possible by
    performing any number of operations. Return that minimum size.

    Let array a = [2, 3, 3, 7, 3, 5] and k = 20
    This is the list of operations that will give us the smallest array (1-
    based indexing):
    Merge the elements at indices (1, 2), resulting array will be - [6, 3, 7, 3,
    5]
    Merge the elements at indices (1, 2), resulting array will be - [18, 7, 3, 5]
    • Merge the elements at indices (3, 4), resulting array will be - [18, 7, 15]
    Hence, the answer is 3.
    """
    output = [arr[0]]
    if len(arr) == 1:
        return 1
    
    for i in range(1, len(arr)):
        if output[-1] * arr[i] <= k:
            output[-1] = output[-1] * arr[i]
        else:
            output.append(arr[i])
    return len(output)

a = [2, 3, 3, 7, 3, 5]
k = 20
nums = [3,3,3,3]
k2 = 6
print("Minimum length of the compressed array is: ", compressedArray(nums, k2))

def HouseRobber(nums):
    dp = [0] * len(nums)


    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])
    else:
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2])

    return dp[-1]

nums = [1,2,3,1]
print("House Robber -- ", HouseRobber(nums))

def HouseRobber2(nums):
    """
    You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

    Example 1:

    Input: nums = [2,3,2]
    Output: 3
    Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
    """
    if len(nums) == 0:
        return
    elif len(nums) == 1:
        return nums[0]
    elif len(nums) == 2:
        return max(nums[0], nums[1])
    
    def finding(houses):
        if len(houses) == 0:
                return 0
        elif len(houses) == 1:
            return houses[0]
        dp = [0] * len(houses)
        dp[0] = houses[0]
        dp[1] = max(houses[0], houses[1])
        for i in range(2, len(houses)):
            dp[i] = max(houses[i] + dp[i-2], dp[i-1])
        return dp[-1]

    return max(finding(nums[:-1]), finding(nums[1:]))

nums = [2,3,2]
print(" House Robber 2 --- ", HouseRobber2(nums))

def runLengthEncoding(strn):
    """
    input_string = "aaabcca"
    output = a3b1c2a1
    
    """
    compressed_string = []
    prev_char = strn[0]
    count = 0

    for s in strn:
        if (prev_char == s):
            count += 1
        elif prev_char != s:
            compressed_string.append(prev_char)
            compressed_string.append(str(count))
            prev_char = s
            count = 1

    if prev_char:
        compressed_string.append(prev_char)
        compressed_string.append(str(count))

    return "".join(compressed_string)
        

input_string = "aaabcca"
print(" After compression it is: ",runLengthEncoding(input_string))