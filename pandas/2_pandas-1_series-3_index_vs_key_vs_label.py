# Index is a way to look up one or more values in a sequence type, such as array or dictionary.
# Yes, index can contain duplicates, which we will explore afterwards.
# Index can be Explicit or Implicit
# Implicit index is mostly Positional Index, like we have in sequence types.
# For example every python list element, has a position.
# We can access any element in that list using the positional index.
# Explicit index is something that we provide.
# Something like the key which we provide in a dictionary.
# We can access an element in the dictionary using that key.


# pandas.Index is the most generic Index Class
# They contain elements
# They are based on NumPy arrays
# They themselves have an implicit positional index
# Basically any python list, tuple or np array etc we can pass to the constructor
# idx = pd.Index([10,20,30,40,50])
# On this index object perform various operation
# We can access the element using the implicit positional index like idx[0]
# We can use the slicing like idx[1:4]
# We can do fancy slicing like idx[[0,2]]
# We can do Boolean masking like idx[idx % 4 == 0]
# In all the above case we get another new index object back
# pandas.Index has few child classes like
# Int64 index that contains integer indices
# Float64 index that contains float indices
# Range index that contains integer sequences defined via range
# The difference between pandas.RangeIndex and pandas.Index is similar to the difference between list and range object in pandas.
# RangeIndex is efficient, hence if we don't provide any label to pandas.Series, then by default it creates a RangeIndex object.

import pandas as pd

idx = pd.Index([1,2,3])
print(idx) # Index([1, 2, 3], dtype='int64')

idx = pd.Index([0.1, 0.2, 0.3])
print(idx) # Index([0.1, 0.2, 0.3], dtype='float64')


idx = pd.Index(range(1,4))
print(idx) # RangeIndex(start=1, stop=4, step=1)
# we can also do as below
ridx = pd.RangeIndex(start=1, stop=4, step=1)
print(ridx) # RangeIndex(start=1, stop=4, step=1)

# pandas indexes have set like properties
# we can perform union(|), intersection(&) elementof(in) operation