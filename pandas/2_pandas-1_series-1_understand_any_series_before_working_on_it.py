# First thing that you need to know about pandas is Series class
# You can think of this as a single column in an excel sheet
# Also Series can be comparable to a 1D numpy array with an explicit index, called label
# Series is a class. Instantiation way is pd.Series()
    # Important params during instantiation are 
    # data = it needs to be an iterable
    # index = optional, if not provided, then pandas will provide 0,1,2.., basically a RangeIndex object.
    # dtype  = optional, basically the type of data
# This is basically a combination of Python List and Python Dictionary.
# List because it holds a list of values
# Dictionary because it kinda requires a key to access a value or values.
# If you pass a dictionary, it automatically fetched the keys from the dictionary, this is the index param in Series() constructor.
# Series can contain duplicate values.

import pandas as pd

s = pd.Series()
print(s) # Series([], dtype: object)

categories = pd.Series( ["food", "household", "communication", "entertainment"] )
print(categories)
# 0             food
# 1        household
# 2    communication
# 3    entertainment
# dtype: object

categories = pd.Series({
    'food': 'food.xlsx',
    'household': 'household.xlsx',
    'communication': 'communication.xlsx',
    'entertainment': 'entertainment.xlsx'
})
print(categories)
# food                      food.xlsx
# household            household.xlsx
# communication    communication.xlsx
# entertainment    entertainment.xlsx
# dtype: object

categories = pd.Series(data=["food", "household", "communication", "entertainment"], 
                        index=(i for i in range(1,5)))
print(categories)
# 1             food
# 2        household
# 3    communication
# 4    entertainment
# dtype: object

# All the below properties are also present in dataframes
print(categories.size) # 4

print(categories.shape) # (4,)

print(categories.dtype) # object

# Return the transpose, which is by definition self.
print(categories.T)
# 1             food
# 2        household
# 3    communication
# 4    entertainment
# dtype: object

print(categories.values) # ['food' 'household' 'communication' 'entertainment']

# If not provided, pandas provide a pandas.RangeIndex Object
# If provided, pandas create a pandas.Index object from the given value
print(categories.index) # Index([1, 2, 3, 4], dtype='int64')

nums = pd.Series( (i for i in range(51)) )
# In both head() and tail() - important parameter n=number of rows you want to display
print(nums.head())
# 0    0
# 1    1
# 2    2
# 3    3
# 4    4
# dtype: int64
print(nums.tail())
# 46    46
# 47    47
# 48    48
# 49    49
# 50    50
# dtype: int64