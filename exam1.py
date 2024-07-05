# print('Heloo all')

# array = [1,2,8,9,3,5,7,10]
# target = 11

# def linearsearch(array):
#     for i in range(len(array)):
#         if array[i] == target:
#             return i
#     return -1

# print('Calling the function', linearsearch(array))

array = [3,2,7,11,15]
target = 101

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
        
print('Two sum', twoSum(array))        

    