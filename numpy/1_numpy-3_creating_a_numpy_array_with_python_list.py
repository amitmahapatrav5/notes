# "import numpy as np" <= This is the accepted aliasing convention
# numpy arrays are called ndarray
# That means, if we create a numpy array, then type(arr) is numpy.ndarray
# syntax to remember "nums = np.array([1,2,3], dtype=np.uint8)"
# numpy array elements use ctypes
# To know the type, we can use arr.dtype.(This .dtype is also present in pandas)
# If we do not specify, numpy assigns the default dtype
    # for integers, default is np.int64
    # for floats, default is np.float64
# However, we can override that if we specify "dtype" param while creating the array.
# Why we want to specify dtype of lower capacity?
# Basically to reduce space
# But be careful here
# If you have floats in the array and you specify dtype as int64, then floats will be truncated.

# Also numpy n dimensional matrix cannot be ragged.
# ragged list is something like this
# l = [
#     [1,2,3].
#     [1,2],
#     [1,2,3,4]
# ]

# So, of a numpy array, if you want to know the shape, just get the arr.shape attribute
# It always returns a tuple

import numpy as np
arr = np.array([1,2,3])
print(type(arr)) # <class 'numpy.ndarray'>
print(arr.dtype) # int64

nums = np.array([1,2,3], dtype=np.uint8)
print(nums.dtype) # uint8
print(nums.shape) # (3,)

l = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
]
matrix = np.array(l, dtype=np.uint8)
print(matrix.shape) # (3, 3)

# Be careful when you are specifying dtype explicitly
# uint8 range is [0,255]
import numpy as np
try:
    arr = np.array([1,2,3,300], dtype=np.uint8)
except OverflowError as ex:
    print(ex) # Python integer 300 out of bounds for uint8

# But in case of float to integer, it will just truncate
arr = np.array([1.20, 2.035, 3.125], dtype=np.int8)
print(arr) # [1 2 3]

# Just for an FYI, numpy has a dtype for strings "<U21", that is , but it's not really used
a = np.array([1,2,3,'x'])
print(a) # ['1' '2' '3' 'x']
print(a.dtype) # <U21