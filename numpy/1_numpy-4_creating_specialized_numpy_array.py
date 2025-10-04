# np.zeros(shape, dtype)
    # shape (mandatory) => integer if 1D else a tuple
    # dtype => default is np.float64
# np.ones(shape, dtype)
    # same as np.zeros
# np.full(shape, fill_value, dtype)
    # same as zeros
    # fill_value => fills the the given number
    # This is the more generalized version of np.zeros and np.ones
    # default dtype however in this case is None
# np.eye(N, M, k, dtype)
    # Generates Identity Matrix [Well not very strict though]
    # N (mandatory) => number of rows
    # M => number of columns (This is weird)
    # k => diagonal index
        # 0 => default => main diagonal
        # +ve => upper diagonal
        # -ve => lower diagonal
    # dtype => default is float64
# np.arange(start, stop, step, dtype)
    # works exactly like python range function
    # only difference is, we can provide step as a non int number like 0.1
    # But if we need the step to be non integer, then better to use linspace
# np.linspace(start, stop, num, endpoint=True)
    # basically used when you need n numbers between a range
    # ex: gimme 10 equi-distance points between 1 and 5
    # there is an option if you want to exclude the stop from the series(endpoint=False)
# np.random.random()
    # Numpy has the random module built in it
    # In that random module, a random function is present
    # Also, like python random module, it also has a seed to set
# np.random.randint()
    # PERFORM R&D
    # PERFORM R&D
    # PERFORM R&D

import numpy as np

zeros = np.zeros(shape=5, dtype=np.uint8)
print(zeros) # [0 0 0 0 0]

zeros = np.zeros(shape=(5,), dtype=np.uint8)
print(zeros) # [0 0 0 0 0]

zeros = np.zeros(shape=(2,5))
print(zeros)
# [[0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]

ones = np.ones(shape=(2,5))
print(ones)
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]

full = np.full(shape=(2,5), fill_value=15)
print(full)
# [[15 15 15 15 15]
#  [15 15 15 15 15]]


# most used identity matrix
eye = np.eye(5)
print(eye)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]

eye = np.eye(3,3,0,np.uint8)
print(eye)
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]

eye = np.eye(3,3,1,np.uint8)
print(eye)
# [[0 1 0]
#  [0 0 1]
#  [0 0 0]]

eye = np.eye(3,3,-1,np.uint8)
print(eye)
# [[0 0 0]
#  [1 0 0]
#  [0 1 0]]

eye = np.eye(5,3)
print(eye)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

arr = np.arange(1,20,2)
print(arr) # [ 1  3  5  7  9 11 13 15 17 19]

# But this is not very frequently used
arr = np.arange(1,20,1.5)
print(arr) #[ 1.   2.5  4.   5.5  7.   8.5 10.  11.5 13.  14.5 16.  17.5 19. ]

# What we generally need is 20 equi-spaced points between a range
# linspace is used very much in that
# we do not have to do all the calc, what should be the step value
arr = np.linspace(1,5, num=6)
print(arr) # [1.  1.8 2.6 3.4 4.2 5. ]

arr = np.linspace(1,5,num=10, endpoint=False)
print(arr) # [1.  1.4 1.8 2.2 2.6 3.  3.4 3.8 4.2 4.6]

# Most used example
x = np.linspace(-2*np.pi, 2*np.pi, 50)
print(x)
# [-6.28318531 -6.02672876 -5.77027222 -5.51381568 -5.25735913 -5.00090259
#  -4.74444605 -4.48798951 -4.23153296 -3.97507642 -3.71861988 -3.46216333
#  -3.20570679 -2.94925025 -2.6927937  -2.43633716 -2.17988062 -1.92342407
#  -1.66696753 -1.41051099 -1.15405444 -0.8975979  -0.64114136 -0.38468481
#  -0.12822827  0.12822827  0.38468481  0.64114136  0.8975979   1.15405444
#   1.41051099  1.66696753  1.92342407  2.17988062  2.43633716  2.6927937
#   2.94925025  3.20570679  3.46216333  3.71861988  3.97507642  4.23153296
#   4.48798951  4.74444605  5.00090259  5.25735913  5.51381568  5.77027222
#   6.02672876  6.28318531]