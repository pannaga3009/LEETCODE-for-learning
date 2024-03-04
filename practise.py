from collections import deque
import re

def maxSlidingWindow(nums, k):
    right = 0
    left = 0
    ans = []
    index = deque([])

    while right < len(nums):

        # Remove indices of elements that are smaller than the current element from the back
        while index and nums[right] >= nums[index[0]]:
            index.popleft()
        
        index.appendleft(right)

        # Check if the window size is equal to k
        if right - left + 1 == k:
            ans.append(nums[index[-1]])
            # Check if the leftmost index in the deque is the index of the leftmost element in the window
            if index[-1] == left:
                index.pop()
            left += 1
        right += 1
    return ans

def minimumPath(grid):
    m, n = len(grid), len(grid[0])

    for i in range(1, m):
        grid[i][0] += grid[i-1][0]

    for i in range(1, n):
        grid[0][i] += grid[0][i-1]

    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])

    return grid[-1][-1]


def trap(height):
    if not height:
        return 0

    left_max, right_max = 0, 0
    left, right = 0, len(height) - 1
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] > left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] > right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water


def fractionToDecimal(numerator, denominator):
    """
    if num == 0 , return 0
    if deno == 0, return "undefined"

    result = ""
    num < 0 or deno < 0
    result = "-"

    result += str(numerator // denominator)
    remainder = numerator % denominator

    if remainder == 0:
        return result
    
    result += "."

    
        seen =  {
        } #Key -> remainder and val -> index position of it
        while rem != 0:
            if seen[remainder] in seen:
                result += result[:seen[remainder]]  + "(" + result[seen[remainder]:] + ")"
            seen[remainder] = len(result)
            remainder *= 10 
            result += str(numerator // denominator)
            remainder %= denominator 
    return result
    """

    if numerator == 0:
        return 0
    
    if denominator == 0:
        return "undefined"

    result = ""
    
    if numerator * denominator < 0:
        result += "-"

    numerator = abs(numerator)
    denominator = abs(denominator)

    #Adding the integer part
    result += str(numerator // denominator)
    remainder = numerator % denominator

    if remainder == 0:
        return result

    result += "."
    seen = {}
    #Store the remainder and its index 
    while remainder != 0:
        if remainder in seen:
            result += result[:seen[remainder]] + "(" + result[seen[remainder]:] + ")"
        seen[remainder] = len(result)
        remainder *= 10
        result += str(remainder // denominator)
        remainder %= denominator
    return result


def reformatDate(date):
    monthDict = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    

    s = date.split()

    day=s[0][:-2]
    month = s[1]
    year = s[2]

    if int(day) < 10:
        day += "0" 
    
    return "".join(f"{year}-{monthDict[month]}-{day}")


def solveNQueens(n):
    col = set()
    posDiag = set()
    negDiag = set()

    res = []
    board = [["."] * n for i in range(n)]

    def solve_back(r):
        if r == n:
            copy = ["".join(row) for row in board]
            res.append(copy)
            return

        for c in range(n):
            if c in col or (r+c) in posDiag or (r-c) in negDiag:
                continue  # Do not do anything move to next iteration

            col.add(c)
            posDiag.add(r+c)
            negDiag.add(r-c)
            board[r][c] = "Q"
            print(board)

            solve_back(r+1)

            col.remove(c)
            posDiag.remove(r+c)
            negDiag.remove(r-c)
            board[r][c] = "."

        solve_back(0)
        return res

def palindromeSubsCount(s):

    def expandAroundCentre(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    palindrome = 0
    for i in range(len(s)):
        
        palindrome += expandAroundCentre(i, i)
        palindrome += expandAroundCentre(i, i+1)
    
    return palindrome

from math import gcd

def minimizeSet(divisor1, divisor2, uniqueCnt1, uniqueCnt2):
    #Calculate LCM of both the divisors
    def calculateLCM(d1, d2):
        return d1*d2//gcd(d1, d2)
    
    left_bound, right_bound, lcm_limt = 0, 10**10, calculateLCM(divisor1, divisor2)

    while left_bound < right_bound:
        mid = ( left_bound + right_bound )// 2

        enough_count_divisor1 = mid - mid//divisor1 >= uniqueCnt1
        enough_count_divisor2 = mid - mid//divisor2 >= uniqueCnt2
        enough_count_combined = mid - mid//lcm_limt >= uniqueCnt1 + uniqueCnt2

        if enough_count_divisor1 and enough_count_divisor2 and enough_count_combined:
            right_bound = mid
        else:
            left_bound = mid + 1
    return left_bound

def reverseStringParanth(s):
    """
    (u(love)i)
    1st iteration:
    stack = [""]
    current_string = ""
    2nd iteration another opening bracket
    current_string = "u"
    stack = ["", "u"]
    current_string =""
    6th iteration, come across ")"
    current_string = love
    reverse of that stack.pop() + evol
    stack = [""]
    u + evol + i
    current_string = "" + iloveyou
    return current_string
    """
    stack = []
    current_string = ""

    for char in s:
        if char == "(":
            stack.append(current_string)
            current_string = ""
        elif char == ")":
            current_string = stack.pop() + current_string[::-1]
        else:
            current_string += char
    return current_string

def NumIslands(grid):
    """
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
    
    """
    num_islands = 0
    directions = [(0,1), (0, -1), (1, 0), (-1, 0)]


    def set_to_zeros(grid, r, c):
        if 0<=r<len(grid) and 0<=c<len(grid[0]) and grid[r][c] == "1":
            grid[r][c] = "0"

            for row_inc, col_inc in directions:
                set_to_zeros(grid, r + row_inc, c + col_inc)

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == "1":
                num_islands += 1
                set_to_zeros(grid, r, c)
            

   
    
    return num_islands

def findTheWinner(n, k):
    """
    [1,2,3,4,5]
 ind[0,1,2,3,4]

    n = 5
    k = 2
    i = 0

    len of nums > 1:
    1st iteration:
        i = (i + k - 1) % len(nums)
        i = (0 + 2 -1 ) % 5
        i = 1
        nums.pop(i)
    2nd iteration
    [1,3,4,5]
    i = (1 + 1) % 4 
    i = 2
    3rd iteration
    [1,3,5]
    2 + 1 % 3
    3 % 3 = 0
    [3,5]
    i = 0
    i = 1 % 2 = 1
    [3]
    return 3
    """
    
    nums = list(range(1, n+1))
    i = 0
    while len(nums) > 1:
        i = (i + k -1)%len(nums)
        nums.pop(i)
    return nums[0]

def canReach(arr, start):
    queue = [start]
    visit = set()

    while queue:
        v = queue.pop(0)
        visit.add(start)

        if v-arr[v] >= 0 and arr[v-arr[v]] == 0:
            return True
        elif v+arr[v] < len(arr) and arr[v+arr[v]] == 0:
            return True
        
        if v-arr[v] >= 0 and v-arr[v] not in visit:
            visit.add(v-arr[v])
            queue.append(v-arr[v])

        if v + arr[v] < len(arr) and v + arr[v] not in visit:
            visit.add(v + arr[v])
            queue.append(v + arr[v])
        
    return False

        
def lengthOfLIS(nums):
    stack = []
    if not nums:
        return 0
    
    for num in nums:
        if not stack or stack[-1] < num:
            stack.append(num)
        else:
            left = 0
            right = len(stack) - 1
            while left < right:
                mid = left + (right - left)//2
                if stack[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            stack[left] = num
    return len(stack)
    
def groupAnagrams(strs):
    strs_sorted = {}
    
    for s in strs:
        sorted_s = "".join(sorted(s))

        if sorted_s not in strs_sorted:
            strs_sorted[sorted_s] = []

        strs_sorted[sorted_s].append(s)

    return list(strs_sorted.values())

def sortByBits(arr):
    return sorted(arr, key = lambda x: (bin(x).count('1'), x))

def KthLargest(nums, k):
    #quick search [3,2,1,5,6,4]
    #[3, 2, 1, 4, 6, 5]
    #[0, 1, 2, 3, 4, 5]
    #           l
    #After swapping
    k = k - 1
    def quickselect(l, r):
        pivot, p = nums[r], l
        for i in range(l, r):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[r] = nums[r], nums[p]


        if k > p: return quickselect(p + 1, r)
        elif k < p: return quickselect(l, p - 1)
        else: return nums[p]
    
    return quickselect(0, len(nums)-1)

import heapq
def KthLargestHeap(nums, k):   
    heap = []
    for n in nums:
        heapq.heappush(heap, n)
        if len(heap) > k:
            heapq.heappop(heap)
    x = heapq.heappop(heap)
    return x

def KthSmallest(nums, k):
    heap = []
    print("k", k)
    for n in nums:
        heapq.heappush(heap, n)
        print(heap)
    
    for _ in range(0, k-1):
        heapq.heappop(heap)
        print("after popping", heap)

    x = heapq.heappop(heap)
    return x

def reverseWords(s):
    #s = "  hello world  "
    #Strip the spaces before and after the text string
    start, end = 0, len(s) - 1
    while start<=end and s[start] == " ":
        start += 1
    while start<=end and s[end] == " ":
        end -= 1
    
    words = []
    current_word = ""

    #Elimimating spaces between words
    for i in range(start, end+1):
        if s[i] == " ":
            words.append(current_word)
            current_word = ""
        else:
            current_word += s[i]

    if current_word:
        words.append(current_word)

    #Now words in an array eliminating all the spaces
    reversed_words = []
    for word in words[::-1]:
        reversed_words.append(word)
    
    return " ".join(reversed_words)


def SearchRotated(nums, target):
    #nums = [4,5,6,7,0,1,2], target = 0
    left, right = 0, len(nums) - 1

    while left<=right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        #Check if the left is sorted
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

          
        # Otherwise, right half is sorted     
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

def Leet151(text):
    left, right = 0, len(text) - 1
    while left<=right and text[left] == " ":   
        left += 1
    while left<=right and text[right] == " ":
        right -= 1

    print(f"{left} and {right}")
    #strip off all the spaces before and after
    split_words  = []
    word = ""
    for i in range(left, right+1):
        if text[i] == " ":
            split_words.append(word)
            word = ""
        else:
            word += text[i]
            

    if word:
        split_words.append(word)

    reversed_words = []
    for word in split_words[::-1]:
        reversed_words.append(word)
    

    return " ".join(reversed_words)

def Leet189(nums, k):
    """
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
    """

    #edge cases are
    if k < 0 or not nums:
        return
    
    i = 0
    while i<k:
        a = nums.pop()
        nums.insert(0, a)
        i += 1
    
    return nums

def Leet238(arr):
    #Input: nums = [1,2,3,4]
    #Output: [24,12,8,6]
    left_arr = [1] * (len(arr))
    right_arr = [1] * (len(arr))
    output = [1] * (len(arr))

    for i in range(1, len(arr)):
        left_arr[i] = arr[i - 1] * left_arr[i - 1] 
    
    for j in range(len(arr)-2, -1, -1):
        right_arr[j] = arr[j + 1] * right_arr[j + 1]

    for k in range(0, len(arr)):
        output[k] = left_arr[k] * right_arr[k]
    
    return output

def Leet139(text, wordDict):
    """
    Input: s = "leetcode", wordDict = ["leet","code"]
    Output: true
    Explanation: Return true because "leetcode" can be segmented as "leet code".
    """
    dp = [False] * (len(text) + 1)
    dp[len(text)] = True

    for i in range(len(text)-1, -1, -1):
        for w in wordDict:
            if i + len(w) <= len(text) and text[i: i + len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    
    return dp[0]


s = "catsandog"
wordDict = ["cats","dog","sand","and","cat"]
print(f"{Leet139(s, wordDict)}")

def Leet274(citations):
    """
    citations = [3,0,6,1,5]

    [0, 1, 2, 3, 4]
    max value in citations 
    arr_h = [0] * max(citations)
    arr_h = [0,0,0,0,0,0,0]
    if c in citations the
    arr_h[c] = 1

    arr_h = [0,1,0,1,0,1,1]
    count = 0
    for len(arr_h), -1, -1
    if arr_h[i] == 1:
    count += 1
    if i<=count:
    return count
    """
    arr_h = [0] * (max(citations)+1)

    for c in citations:
        arr_h[c] += 1
    
    print(arr_h)
    count_sum = 0
    for i in range(len(arr_h)-1, -1, -1):
        count_sum += arr_h[i]
        
        if i<=count_sum:
            return i


citations = [1,2,2]
print(f"{Leet274(citations)}")


    
def TwoSum(nums, target):
    MapValue = {}
    #Input: nums = [2,7,11,15], target = 9
    #Output: [0,1]
    for i in range(len(nums)):
        diff = target - nums[i]
        if diff in MapValue:            
            print(f" This is {diff} and mapval is {MapValue}")
            if diff in MapValue:
               return [MapValue[diff], i]
        else:
            MapValue[nums[i]] = i        
    return 

def ThreeSum(nums):
    #Input: nums = [-1,0,1,2,-1,-4]
    # Output: [[-1,-1,2],[-1,0,1]]
    # Explanation: 
    # nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
    # nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
    # nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
    # The distinct triplets are [-1,0,1] and [-1,-1,2].
    # Notice that the order of the output and the order of the triplets does not matter.

    res= []
    nums.sort()
    
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l = i + 1
        r = len(nums) - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total > 0:
                r -= 1
            elif total < 0:
                l += 1
            else:
                triplets = [nums[i], nums[l], nums[r]]
                res.append(triplets)
                while l < r and nums[l] == triplets[1]:
                    l += 1
                while l < r and nums[r] == triplets[2]:
                    r -= 1
    return res


def TwoSum2(nums, target):
    # Input: numbers = [2,3,4], target = 6
    # Output: [1,3]
    l = 0
    r = len(nums) - 1
    while l < r:
        total = nums[l] + nums[r]
        if total > target:
            r -= 1
        elif total < target:
            l += 1
        else:
            return [l+1, r+1]
    return 

def DuplicateNumbers(arr):
#     Input: nums = [3,1,3,4,2]
# Output: 3
    l = 0
    r = len(arr) - 1

    while l < r:
        mid = (l + r) // 2
        count = 0  
        for n in arr:
            if n <= mid:
                count += 1
        if count > mid:
            r = mid
        else:
            l = mid + 1
        
    return l

def freqKElements(nums, k):
    freq_map = {}
    
    for n in nums:
        freq_map[n] = freq_map.get(n, 0) + 1
    # freq = [[] for _ in range(len(nums) + 1)]
    # for key, val in freq_map.items():
    #     freq[key].append(val)
    
    # res = []
    # for i in range(len(freq) - 1, 0, -1):
    #     for n in freq[i]:
    #         res.append(n)
        
    #     if len(res) == k:
    #         return res
        
    sorted_elements = sorted(freq_map.keys(), key=lambda x: freq_map[x], reverse=True)
    print("sorted elements ", sorted_elements)

    # Return the first k elements
    return sorted_elements[:k]

def sortedSearch2(nums, target):
    # Input: nums = [2,5,6,0,0,1,2]
    # target = 0
    left = 0
    right = len(nums) - 1

    if len(nums) == 1:
        return nums[0] == target
    

    while left <= right:
        mid = (left + right)//2
        if nums[mid] == target:
            return True
        
        if nums[mid] == nums[left]:
            left += 1
            continue
        
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False

def removeDuplicatesSorted(nums):
    # Input: nums = [1,1,1,2,2,3]
    # Output: 5, nums = [1,1,2,2,3,_]
    left = 1
    count = 1
    for right in range(1, len(nums)):
        if nums[right] == nums[right - 1]:
            count += 1
        else:
            count = 1
            # print(f"{nums[left]} and {nums[right]} ")

        if count <= 2:
            nums[left] = nums[right]
            left += 1
            
    print("nums ", nums)   
    return left


def equalPairs(grid) -> int:
    #find the columns 
    count = 0
    for c in len(grid[0]):
        temp = []
        for r in len(grid):
            temp.append(grid[r][c])
        if temp in grid:
            count += grid.count(temp)
    return count


def setZeros(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    z_rows, z_cols = set(), set()

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                z_rows.add(r)
                z_cols.add(c)

    for z in z_rows:
        for c in range(cols):
            matrix[z][c] = 0
    
    for z in z_cols:
        for r in range(rows):
            matrix[r][z] = 0
    
    return matrix

def maxMatrixSum(matrix):
    n = len(matrix)
    count, res = 0, 0
    min_val = float("inf")

    for i in range(n):
        for j in range(n):
            res += abs(matrix[i][j])

            if matrix[i][j] < 0:
                count += 1

            min_val = min(min_val, abs(matrix[i][j]))
    
    if count % 2 == 0:
        return res
    else:
        return res - 2 * min_val
    
def max_sum(nums):
    #nums =[[7,2,1],[6,4,2],[6,5,3],[3,2,1]]
    rows, cols = len(nums), len(nums[0])
    score = 0

    #Sort in descending order
    for row in nums:
        row.sort(reverse=True)

    for col in range(cols):
        col_max = 0
        for row in range(rows):
            if nums[row][col] > col_max:
                col_max = nums[row][col]

        score += col_max
    
    return score

def allCellsDistOrder(rows, cols, rCenter, cCenter):
    #take an array called result where you can store index and distance
    result = []

    for i in range(rows):
        for j in range(cols):
            distance = abs(i - rCenter) + abs(j - cCenter)
            result.append([i, j, distance])
    
    #Lets sort using bubble sort
    
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i][2] > result[j][2]:
                result[i], result[j] = result[j], result[i]
    
    #remove the distance col from the list
    for row in result:
        #For every row pop the distance from the list
        row.pop()
    
    return result

def diagonalSort(mat):
    #mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
    sorted_diagonals = []
    rows = len(mat)
    cols = len(mat[0])

    #Try to take the first row and first column
    #Appending the leftmost column
    for row in range(rows):
        sorted_diagonals.append([row, 0])

    #Appending the topmost row, leave the one before as its shared by leftmost column
    for col in range(1, cols):
        sorted_diagonals.append([0, col])   
    
    #Lets get the diagonal elements and sort them
    for start_row, start_col in sorted_diagonals:
        diagonal_elements = []
        row = start_row
        col = start_col

        while row < rows and col < cols:
           diagonal_elements.append(mat[row][col])
           row, col = row + 1, col + 1
        
        diagonal_elements.sort()

        #Lets place the sorted diagonal elements in the matrx 
        row, col = start_row, start_col
        for ele in diagonal_elements:
            mat[row][col] = ele
            row, col = row + 1, col + 1
    return mat

mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
print(diagonalSort(mat)) 
#Output: [[1,1,1,1],[1,2,2,2],[1,2,3,3]]

# Test cases
rows = 2
cols = 2
rCenter = 0
cCenter = 1
print(allCellsDistOrder(rows,cols, rCenter, cCenter)) 
#[[0,1],[0,0],[1,1],[1,0]]
 # Output: 4
nums = [[7,2,1],[6,4,2],[6,5,3],[3,2,1]]
print(f" Max sum in the matrix is {max_sum(nums)}")


matrix = [[1,2,3],[-1,-2,-3],[1,2,3]]
print(f"Matrix sum is {maxMatrixSum(matrix)}")


matrix = [[1,1,1],[1,0,1],[1,1,1]]
print(f"Set rows and cols to zero {setZeros(matrix)}")

# grid = [[3,2,1],[1,7,6],[2,7,7]]
# print(f"-- equal pairs matrix -- {equalPairs(grid)}")

nums = [1,1,1,2,2,3]
print(f"Dulicates after removal: {removeDuplicatesSorted(nums)}")  



nums = [2,5,6,0,0,1,2]
target = 6
print(f"search in rotated array is: {sortedSearch2(nums, target)}")  

             
nums = [1,1,1,2,2,3]
k = 2
print(f"Frequency num is: {freqKElements(nums, k)}")  

arr = [3,1,3,4,2]
print(f"Duplicate num is: {DuplicateNumbers(arr)}")  

nums = [2,3,4]
target = 6
print(f"Two sum 2 is: {TwoSum2(nums, target)}")

arr = [1,2,3,4]
print(f"{Leet238(arr)}")


s = "  hello world  "
# print(reverseWords(s))
leet151 = Leet151(s)
print(leet151)

nums = [1,2,3,4,5,6,7]
k = 3
print("c'mon", Leet189(nums, k))


func13 = KthLargest([3,6,5,4,1,8], 4)
#kth smallest is [1,3,4,5,6,8]

print("Kth  with quickselect",func13)

func15 = KthSmallest([3,2,1,5,6,4], 2)
print("Kth Smallest", func15)

func14 = KthLargestHeap([3,2,3,1,2,4,5,5,6], 4)
print("Heappp", func14)

func10 = canReach([4,2,3,0,3,1,2], 5)
print("Can reach -- ",func10)



func4 = solveNQueens(3)
print(func4)


nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
func1 = maxSlidingWindow(nums, k)
print(func1)

grid = [[1, 2, 3], [4, 5, 6]]
minimumpath = minimumPath(grid)
print(minimumpath)

height = [0,1,0,2,1,0,1,3,2,1,2,1]
func2 = trap(height)
print(func2) 

result = fractionToDecimal(-50, 8)
print("fraction to decimal" ,result)

date = "20th Oct 2052"
func3 = reformatDate(date)
print(func3)

func5 = palindromeSubsCount("aaa")
print("Palindrome ---> ", func5)

func6 = minimizeSet(3, 15, 2, 10)
print("minimize set",func6)

w = " (u(love)i)"
func7 = reverseStringParanth(w)
print(func7)

grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]

func8 = NumIslands(grid)
print(func8)

func9 = findTheWinner(5, 2)
print("Winner is -",func9)

func10 = lengthOfLIS([10,9,2,5,3,7,101,18])
print(func10)

strs = ["eat","tea","tan","ate","nat","bat"]
func11 = groupAnagrams(strs)
print("Anagrams -- ", func11)

func12 = sortByBits([1024,512,256,128,64,32,16,8,4,2,1])
print("sort by bits", func12)