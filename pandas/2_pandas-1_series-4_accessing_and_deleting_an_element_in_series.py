# In case of Pandas Series, we have an implicit positional index.
# But it also allows us to provide an explicit index which often referred as Label.
# So in case of Pandas Series, we can access an element using either the positional index which is implicit or by the label which is explicit.

# Note that this explicit index is not something mandatory for us to provide.
# If we provide an arraylike object while creating the series it will create the index object
# Otherwise it will just assign a RangeIndex object.
import pandas as pd

categories = pd.Series({
    'food': 'food.xlsx',
    'household': 'household.xlsx',
    'communication': 'communication.xlsx',
    'entertainment': 'entertainment.xlsx'
})
print(categories)

# ============SAME AS============
import pandas as pd

data=[
    'food.xlsx',
    'household.xlsx',
    'communication.xlsx',
    'entertainment.xlsx'    
]
idx = pd.Index(['food', 'household', 'communication', 'entertainment'])

categories = pd.Series(data=data, index=idx)
print(categories)

# About slicing, it is natural to understand that we can do slicing using the positional index.
# But we can also do the slicing using the explicit index or Label
# Note that when we do slicing with positional index(implicit), that time, it works same as python list or sting slicing
# but in case of slicing using the labels, the last index is also considered as below

import pandas as pd

categories = pd.Series({
    'Administrative': 'Administrative.xlsx',
    'Clothing': 'Clothing.xlsx',
    'Communication': 'Communication.xlsx',
    'Electronics': 'Electronics.xlsx',
    'Entertainment': 'Entertainment.xlsx',
    'Food': 'Food.xlsx',
    'Household': 'Household.xlsx',
    'Hygiene': 'Hygiene.xlsx',
    'Medical': 'Medical.xlsx',
    'Nutrition': 'Nutrition.xlsx',
    'Travel': 'Travel.xlsx'
})

print(categories['Clothing':'Food'])
# Clothing              Clothing.xlsx
# Communication    Communication.xlsx
# Electronics        Electronics.xlsx
# Entertainment    Entertainment.xlsx
# Food                      Food.xlsx # This is not excluded
# dtype: object

# A very basic question can be raised here, can index have duplicates?
# If we think about it, if it is positional, then it must not. This must be obvious to understand.
# But if it is explicit then, yes. But why?
# Well, if explicit index can have duplicates, then if we search for such duplicate index, what do we get?
# Yes, we get multiple elements from the Series.
# To however update a specific value, we should use the implicit index
# If we use the label to update the value, the all the values having that label will get updated.
import pandas as pd

data = [
    'Administrative.xlsx',
    'Clothing.xlsx',
    'Communication.xlsx',
    'Electronics.xlsx',
    'Food1.xlsx',
    'Food2.xlsx',
    'Food3.xlsx',
    'Household.xlsx'
]
idx=[
    'Administrative',
    'Clothing',
    'Communication',
    'Electronics',
    'Food',
    'Food',
    'Food',
    'Household'
]
categories = pd.Series(data=data, index=idx)
print(categories['Food'])
# Food    Food1.xlsx
# Food    Food2.xlsx
# Food    Food3.xlsx
# dtype: object

# Note that the implicit index is based on position and it is like 0,1,2,..
# But we can also give the explicit index as integers.
# In such case where both implicit and explicit index are integers, when we access say series[n] then what index is considered?
# To resolve this conflict, Series objects have 2 attributes which returns an array like object
# Those 2 attributes are Series.loc and Series.iloc
# Series.loc refers to the location basically to the label
# Series.iloc refers to the indexed location basically to the implicit index
# This is the preferable way to access data in pandas.Series
# Both loc and iloc returns an arraylike data structure
# Hence we can use the [] notation

import pandas as pd

data = [i for i in range(1,11)]
idx = [i for i in range(3,13)]
nums = pd.Series(data=data, index=idx)

print(nums)
# 3      1
# 4      2
# 5      3
# 6      4
# 7      5
# 8      6
# 9      7
# 10     8
# 11     9
# 12    10
# dtype: int64

print(nums[3]) # 1 not 4 so it is considering labels not implicit index

print(nums[:3]) # Now it is considering implicit index
# 3    1
# 4    2
# 5    3
# dtype: int64

# This is confusing when both implicit and explicit indices are integers
# To avoid this confusion, pandas exposed 2 attributes
# loc and iloc
print(nums.loc[3], nums.iloc[3]) # 1 4


# Note that Index objects are immutable
# Hence, deleting an element from a series object is not possible directly
# Hence we use the drop([label1, label2, ..]) method
# That basically creates another series object and the new series object will not contain the items we dropped.
# Note that we cannot dorp by implicit index
# But, we can get the label from the positional index itself
# Because, Index object is nothing but a series itself.
import pandas as pd

data=[2, 4, 3, 9, 6, 6, 4, 6, 9]
idx = ['two', 'two', 'three', 'three', 'three', 'two', 'four', 'six', 'nine']

divisible = pd.Series(data=data, index=idx)
print(divisible)
# two      2
# two      4
# three    3
# three    9
# three    6
# two      6
# four     4
# six      6
# nine     9
# dtype: int64

dropped_by_key = divisible.drop(['four', 'six', 'nine'])
print(dropped_by_key)
# two      2
# two      4
# three    3
# three    9
# three    6
# two      6
# dtype: int64

dropped_by_position = divisible.drop([divisible.index[-1], divisible.index[-2], divisible.index[-3]])
print(dropped_by_position)
# two      2
# two      4
# three    3
# three    9
# three    6
# two      6
# dtype: int64

# Simply
dropped_by_position = divisible.drop(divisible.index[[-1,-2,-3]])
print(dropped_by_position)
# two      2
# two      4
# three    3
# three    9
# three    6
# two      6
# dtype: int64