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