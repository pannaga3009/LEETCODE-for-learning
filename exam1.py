# print('Heloo all')

# array = [1,2,8,9,3,5,7,10]
# target = 11

# def linearsearch(array):
#     for i in range(len(array)):
#         if array[i] == target:
#             return i
#     return -1

# print('Calling the function', linearsearch(array))
def twoSum(array):
    """
    sorted 

    dict_map to store the index
    """
    dict_index = {}
    output = []
    for i in range(len(array)):
        diff = target - array[i]
        if diff in dict_index:
            output.append([i, dict_index[diff]])
            return output
        else:
            dict_index[array[i]] = i

    return -1 

def mergeSorted(nums1, nums2, m, n):
    """
    merge the sorted arrays 
    1. In place merge
    2. Sorted in ascending order
    3. Have another element k = m + n

    example
    Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
    Output: [1,2,2,3,5,6]
    Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
    The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
    """

    k = len(nums1) - 1
    i = m - 1
    j = n - 1

    while i>=0 and j>=0:
        if nums1[i] < nums2[j]:
            nums1[k] = nums2[j]
            k -= 1
            j -= 1
        else:
            nums1[k] = nums1[i]
            k -= 1
            i -= 1
    
    while j>=0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
    
    return nums1

        
# print('Two sum', twoSum(array))        

""""
select employee_name FROM employee
where salary > 70000;
"""


if __name__ == "__main__":
    # array = [3,2,7,11,15]
    # target = 101
    # array = list(map(int, input("Enter the array list").split()))
    # target = int(input("What is the target ?"))
    # print('Two sum', twoSum(array))
    nums1 = [1,2,3,0,0,0]
    m = 3
    nums2 =[2,5,6]
    n = 3
    print("Merge two array === ", mergeSorted(nums1, nums2, m, n))





