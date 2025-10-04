# Reshaping is pretty simple. We only need to keep track of 3 things
    # 1. Example: (6, ) can only be reshaped into (3,2) or (2,3).
    # 2. After reshaping, we get a new array object.
    # 3. Tricky part is, the array objects are different, but the elements present in the array are same. So change will replicate everywhere.
    # 4. If you want to break this link, use copy method as arr = original.reshape(3,2).copy()

import numpy as np

arr = np.array([1,2,3,4,5,6])
print(arr)
# [1 2 3 4 5 6]

arr_cpy_1 = arr.reshape(3,2)
print(arr_cpy_1)
# [[1 2]
#  [3 4]
#  [5 6]]
print(arr_cpy_1 is arr) # False

arr_cpy_2 = arr.reshape(2,3)
print(arr_cpy_2)
# [[1 2 3]
#  [4 5 6]]

arr[2], arr[4] = 30, 50
print(arr)
# [ 1  2 30  4 50  6]

print(arr_cpy_1)
# [[ 1  2]
#  [30  4]
#  [50  6]]

print(arr_cpy_2)
# [[ 1  2 30]
#  [ 4 50  6]]

# How it is mostly used?
arr = np.arange(1,11).reshape(2,5)
print(arr)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]


# Stacking is also very simple. There are n points to keep in mind
# 1. 2 types of stacking are there. hstack and vstack
# 2. For stacking the arrays must be compatible.
    # for vstack, the columns should be of same shape
    # for hstack, the rows should be of same shape
# 3. np.v/hstack((a1,a2,a3)) => a1, a2, a3 should be passed as a tuple or also can be a list
# 4. The new array after stacking has no connection to the original arrays
# 5. The new array dtype will be decided by pandas. We don't have control on it
    # However, we can control in some extent, but casting the arrays to a certain dtype before stacking
    # This casting is done using astype(np.int64) function
# 6. np.concatenate() is a more generic version of hstack and vstack introduced in numpy 1.2
    # In np.concatenate() we can however control the resulting array dtype.


import numpy as np
from numpy import random as npr

a1 = np.arange(1,11).reshape(2,5)
a2 = npr.random(15).reshape(3,5)
a3 = np.linspace(1,10,5)

a = np.vstack((a1,a2,a3))

print(a1, a1.dtype) # int64
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]
print(a2, a2.dtype) # float64
# [[0.18919165 0.52660099 0.92025013 0.63069162 0.08731057]
#  [0.91583613 0.50204843 0.54321538 0.50263901 0.48517779]
#  [0.85487135 0.90056292 0.05776739 0.20241095 0.04295447]]
print(a3, a3.dtype) # float64
# [ 1.    3.25  5.5   7.75 10.  ]
print(a, a.dtype)
# [[ 1.          2.          3.          4.          5.        ]
#  [ 6.          7.          8.          9.         10.        ]
#  [ 0.18919165  0.52660099  0.92025013  0.63069162  0.08731057]
#  [ 0.91583613  0.50204843  0.54321538  0.50263901  0.48517779]
#  [ 0.85487135  0.90056292  0.05776739  0.20241095  0.04295447]
#  [ 1.          3.25        5.5         7.75       10.        ]]

a1 = np.arange(1,13).reshape(3,4)
a2 = npr.random(15).reshape(3,5)
a3 = np.linspace(1,3,3).reshape(3,1)

# a = np.hstack((
#     a1.astype(np.int64),
#     a2.astype(np.int64),
#     a3.astype(np.int64)
# ))


# print(a, a.dtype) # int64
# # [[ 1  2  3  4  0  0  0  0  0  1]
# #  [ 5  6  7  8  0  0  0  0  0  2]
# #  [ 9 10 11 12  0  0  0  0  0  3]]


# np.concatenate need further studying
# import numpy as np
# from numpy import random as npr

# a1 = np.arange(1,13).reshape(3,4)
# a2 = npr.random(15).reshape(3,5)
# a3 = np.linspace(1,3,3).reshape(3,1)

# a = np.concatenate(
#     (a1,a2,a3),
#     axis=1,
#     dtype=np.float64
# )

# print(a)