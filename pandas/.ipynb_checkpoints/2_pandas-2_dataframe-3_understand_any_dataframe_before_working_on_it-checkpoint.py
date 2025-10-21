import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

data = {
    'Category':     categories,
    'Sub Category': sub_categories,
    'Amount':       amounts
}

df = pd.DataFrame(data)
# By default head() prints the first 5 records in the df
# By default tail() prints the last 5 records in the df
# Most used parameter df.head(n=7) or df.tail(n=6)
print(df.head())
#          Category      Sub Category  Amount
# 0  Administrative           Banking    1000
# 1        Clothing           Top Wear    2000
# 2   Communication          Internet   12000
# 3     Electronics           Handset  200000
# 4   Entertainment  OTT Subscription    2000
print(df.tail())
#      Category Sub Category  Amount
# 10    Hygiene     Skincare     500
# 11    Medical    Insurance   51560
# 12  Nutrition   Supplement   15000
# 13     Travel          Bus    2000
# 14     Travel        Train    1500


# Dataframe is kinda like an excel sheet
# In excel, we have headers and row numbers which are metadata
# And data is stored in the cell
# From a dataframe, if we just want to get the data, we can use .value
# type of df.values is a numpy.ndarray type object
# so we can see it's shape by df.values.shape
# we can check it's dtype by df.values. dtype
print(df.values)
# [['Administrative' 'Banking' 1000]
#  ['Clothing' 'Top Wear' 2000]
#  ['Communication' 'Internet' 12000]
#  ['Electronics' 'Handset' 200000]
#  ['Entertainment' 'OTT Subscription' 2000]
#  ['Food' 'Vegetable' 10000]
#  ['Food' 'Fruits' 2000]
#  ['Food' 'Dairy' 3000]
#  ['Household' 'House Rent' 96000]
#  ['Hygiene' 'Haircare' 2500]
#  ['Hygiene' 'Skincare' 500]
#  ['Medical' 'Insurance' 51560]
#  ['Nutrition' 'Supplement' 15000]
#  ['Travel' 'Bus' 2000]
#  ['Travel' 'Train' 1500]]


# basically shape and size of the dataframe
print(df.shape) # (15, 3) => representing number of (rows, cols)
print(df.size) # 45 => representing number of cells in the sheet


# Understand that df is a collection of series
# Every series is somewhat like a numpy array
# Every numpy array must be homogeneous
# Hence the property name is dtypes not dtype
# Basically gives the overview of dtype of every column
print(df.dtypes)
# Category        object
# Sub Category    object
# Amount           int64
# dtype: object


# At this point pandas dataframe cannot go beyond 2D
# So basically it will have rows and columns
# Rows by default is a pandas.RangeIndex object
# Cols by default is a pandas.Index object
# df.axes provides a list holding those 2 information
print(df.axes)


# Basically provides all the above information in a concise format
# One thing you can notice from this info() is missing data
# RangeIndex basically tells how many rows are there in the df
# And for every column, we get Non-Null Counts
# So if RangeIndex Count > Non-Null Counts, then there are some missing values
# Also the Dtype of every column is also useful
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 15 entries, 0 to 14
# Data columns (total 3 columns):
#  #   Column        Non-Null Count  Dtype 
# ---  ------        --------------  ----- 
#  0   Category      15 non-null     object
#  1   Sub Category  15 non-null     object
#  2   Amount        15 non-null     int64 
# dtypes: int64(1), object(2)
# memory usage: 492.0+ bytes
