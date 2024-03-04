import stack


string = "gninrael nIdeknil htiw tol a nrael"
reversed_string = ""
s = stack.Stack()



for i in string:
    s.push(i)

for i in range(0, s.size()):
    reversed_string += s.pop()



print(reversed_string)