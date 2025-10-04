# Array is a conceptual thing in the world of programming.
# Python List is one type of implementation of array.
# Similarly the NumPy array(also called ndarray) is also another implementation of arrays.
# With some differences of courses
# Facts
    # NumPy arrays are fixed in size, cannot add or remove, BUT Replace, Yes
    # They are "homogeneous", yes there is air quotes
    # Elements have to be one of the specialized datatypes. (This is interesting)
    # ndarray has only implicit index, that is positional. We cannot add explicit index like we can to Pandas Series.
    # ndarray supports slicing, like python List, arr[start:stop:step]
    # ndarray Special
        # reshaping => 4x6 can be 2x12 or 3x8 or 24x1 etc
        # masking => arr[(arr>2) & (arr<10)]
        # fancy indexing => arr[[0,3,4]]
    # ndarray benefits
        # space efficient than python Lists
        # array manipulation and calculations are way faster than python List
            # This is possible because of something called "Vectorization"
    # ndarray used datatypes of underlying C Language
        # this is to make it memory efficient and fast

# History to understand what datatypes NumPy team has decided to make
    # Numbers are basically stored using bit
    # Say we have 4 bits
    # We can have 0000 to 1111, that is, [0,15]
    # But if it is signed, then 1 bit is required to store the sign
    # So -7,-6,...,-0,+0,...+6,+7
    # But we do't need -0 and +0, only one 0 is needed, so we save one.
    # So the range is now, [-8, 7]. Why not [-7, 8] => Well thats how it was initially decided.
    # This is how the encoding is basically defined for computers
    # Now we have various sizes
    # Signed Integers
        # 8 bits
        # 16 bits
        # 32 bits
        # 64 bits
    # Unsigned Integers
        # 8 bits
        # 16 bits
        # 32 bits
        # 64 bits
    # Now all the above is basically fixed size encoding
    # But python used something called variable size encoding
    # Hence thats a overhead, and also for python everything is an object
    # But python uses fixed size encoding for floats, 64 bits
# NumPy team, basically used the types which we discussed above
    # Signed Integers => np.int8, np.int16, np.int32, np.int64
    # Unsigned Integers => np.uint8, np.uint16, np.uint32, np.uint64
    # floats => np.float32, np.float64(np.float64 is compatible with python float)
    # complex => np.complex64, np.complex128(np.complex128 is compatible with python complex)

# What is this vectorization, and how it speeds things up?
# When we perform an arithmetic operation, C is faster than python. WHY?
# Well python is dynamically typed
# So every time we perform an operation, python has to do the following
    # determine the type of each operand, say a*b=> type(a) and type(b)
    # perform the operation => if a*b is not possible, check if b*a is possible? => __mul__, __rmul__
    # And if we are using a loop, like,
        # a = [1,2,3,4,5]
        # b = [10,20,30,40,50]
        # ab = [ i*j for i, j in zip(a,b) ]
        # Then it has to perform the overhead work for every objects
        # This slows things down
    # But in case of C, its already known in advance
        # Type of every operand
        # How to perform the operation
    # Hence C is faster
# As NumPy used the C supported types, it pushes these operation down to the C level
# Hence the calculation is much faster
# Given that, a and b are NumPy arrays
    # NumPy implements many functions to support several operations
        # a+b => add(a,b)
        # a*b => mul(a,b)
        # a/b => divide(a,b)
        # sin(a)/sin(b) => divide(sin(a), sin(b))
# NumPy pushes these loops and the calculation to the c level. This is called "Vectorization"
# Also these add, mul, divide, sin functions are called universal functions(ufunc), by NumPy team