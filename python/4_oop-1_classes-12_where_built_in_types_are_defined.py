# We can create list, tuple, dictionary, set, string, functions, generators etc. But the class of which these are instance of are present in different modules.
# Majority built-in types we use are present in a module called types.
# types module is present in built-in standard python library

from types import FunctionType, MethodType, LambdaType, GeneratorType

# list class - No need to import
nums = [1,2,3,4,5]

# FunctionType class - Need to import
def fun():
    pass

# Generator class - Need to import
squares = ( num**2 for num in nums )

print(type(nums))
print(type(fun))
print(type(squares))

# Need More Analysis to complete
