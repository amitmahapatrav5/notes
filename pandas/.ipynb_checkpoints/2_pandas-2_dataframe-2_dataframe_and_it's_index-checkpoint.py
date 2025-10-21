# There are 4 indices in the dataframe
# 2 in row (Implicit and Explicit)
# 2 in column (Implicit and Explicit)

# Whatever we see while printing the dataframe, the row name and the column name are explicit index
    # Row Index, print(df.index)
    # Column Index, print(df.index)
import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

idx = pd.Index(range(100,115))

categories = pd.Series(data= categories, index= idx, name= 'Category')
sub_categories = pd.Series(data= sub_categories, index= idx, name= 'Sub Category')
amount = pd.Series(data= amounts, index= idx, name= 'Amount')

df = pd.DataFrame([categories, sub_categories, amount]).T

print(df)
#           Category      Sub Category  Amount
# 100  Administrative           Banking    1000
# 101        Clothing           Top Wear    2000
# 102   Communication          Internet   12000
# 103     Electronics           Handset  200000
# 104   Entertainment  OTT Subscription    2000
# 105            Food         Vegetable   10000
# 106            Food            Fruits    2000
# 107            Food             Dairy    3000
# 108       Household         House Rent   96000
# 109         Hygiene          Haircare    2500
# 110         Hygiene          Skincare     500
# 111         Medical         Insurance   51560
# 112       Nutrition        Supplement   15000
# 113          Travel               Bus    2000
# 114          Travel             Train    1500

print(df.index) # RangeIndex(start=100, stop=115, step=1)
print(df.columns) # Index(['Category', 'Sub Category', 'Amount'], dtype='object')

# We can even change the both row and column index using rename method on df object
df = df.rename(index={ idx: idx-99 for idx in df.index })
print(df)
#           Category      Sub Category  Amount
# 1   Administrative           Banking    1000
# 2         Clothing           Top Wear    2000
# 3    Communication          Internet   12000
# 4      Electronics           Handset  200000
# 5    Entertainment  OTT Subscription    2000
# 6             Food         Vegetable   10000
# 7             Food            Fruits    2000
# 8             Food             Dairy    3000
# 9        Household         House Rent   96000
# 10         Hygiene          Haircare    2500
# 11         Hygiene          Skincare     500
# 12         Medical         Insurance   51560
# 13       Nutrition        Supplement   15000
# 14          Travel               Bus    2000
# 15          Travel             Train    1500

# Similarly if we want to change the column explicit index, we can use the columns param
# Also, it is not necessary to change all the index using rename
# We may change only the selected ones
df = df.rename(columns={ 'Category': 'Category Name', 'Sub Category': 'Sub Category Name' })
print(df)
#       Category Name Sub Category Name  Amount
# 100  Administrative           Banking    1000
# 101        Clothing           Top Wear    2000
# 102   Communication          Internet   12000
# 103     Electronics           Handset  200000
# 104   Entertainment  OTT Subscription    2000
# 105            Food         Vegetable   10000
# 106            Food            Fruits    2000
# 107            Food             Dairy    3000
# 108       Household         House Rent   96000
# 109         Hygiene          Haircare    2500
# 110         Hygiene          Skincare     500
# 111         Medical         Insurance   51560
# 112       Nutrition        Supplement   15000
# 113          Travel               Bus    2000
# 114          Travel             Train    1500


# Interestingly, we can set a particular column as an index
# Why we need that?
# Well basically for filtering purpose
# We can do that using the df.set_index() method
import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

idx = pd.Index(range(100,115))

categories = pd.Series(data= categories, index= idx, name= 'Category')
sub_categories = pd.Series(data= sub_categories, index= idx, name= 'Sub Category')
amount = pd.Series(data= amounts, index= idx, name= 'Amount')

df = pd.DataFrame([categories, sub_categories, amount]).T

print(df)
#           Category      Sub Category  Amount
# 100  Administrative           Banking    1000
# 101        Clothing           Top Wear    2000
# 102   Communication          Internet   12000
# 103     Electronics           Handset  200000
# 104   Entertainment  OTT Subscription    2000
# 105            Food         Vegetable   10000
# 106            Food            Fruits    2000
# 107            Food             Dairy    3000
# 108       Household         House Rent   96000
# 109         Hygiene          Haircare    2500
# 110         Hygiene          Skincare     500
# 111         Medical         Insurance   51560
# 112       Nutrition        Supplement   15000
# 113          Travel               Bus    2000
# 114          Travel             Train    1500
df = df.set_index('Category')
print(df)
#                     Sub Category  Amount
# Category                                
# Administrative           Banking    1000
# Clothing                 Top Wear    2000
# Communication           Internet   12000
# Electronics              Handset  200000
# Entertainment   OTT Subscription    2000
# Food                   Vegetable   10000
# Food                      Fruits    2000
# Food                       Dairy    3000
# Household              House Rent   96000
# Hygiene                 Haircare    2500
# Hygiene                 Skincare     500
# Medical                Insurance   51560
# Nutrition             Supplement   15000
# Travel                       Bus    2000
# Travel                     Train    1500
print(df.loc['Food'])
#          Sub Category Amount
# Category                    
# Food        Vegetable  10000
# Food           Fruits   2000
# Food            Dairy   3000


# Dropping column or row, almost works the same way as Series
# We need to use the drop method, which basically returns a new df
# We can drop one or more, row or column at a time, similar syntax as series
# If we want to delete specific columns(s), we can use column param, df.drop(columns=[])
# If we want to delete specific row(s), we can use index param, df.drop(index=[])
import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

idx = pd.Index(range(100,115))

categories = pd.Series(data= categories, index= idx, name= 'Category')
sub_categories = pd.Series(data= sub_categories, index= idx, name= 'Sub Category')
amount = pd.Series(data= amounts, index= idx, name= 'Amount')

df = pd.DataFrame([categories, sub_categories, amount]).T

print(df)
#           Category      Sub Category  Amount
# 100  Administrative           Banking    1000
# 101        Clothing           Top Wear    2000
# 102   Communication          Internet   12000
# 103     Electronics           Handset  200000
# 104   Entertainment  OTT Subscription    2000
# 105            Food         Vegetable   10000
# 106            Food            Fruits    2000
# 107            Food             Dairy    3000
# 108       Household         House Rent   96000
# 109         Hygiene          Haircare    2500
# 110         Hygiene          Skincare     500
# 111         Medical         Insurance   51560
# 112       Nutrition        Supplement   15000
# 113          Travel               Bus    2000
# 114          Travel             Train    1500

df = df.drop(columns='Sub Category')
print(df)
#           Category  Amount
# 100  Administrative    1000
# 101        Clothing    2000
# 102   Communication   12000
# 103     Electronics  200000
# 104   Entertainment    2000
# 105            Food   10000
# 106            Food    2000
# 107            Food    3000
# 108       Household   96000
# 109         Hygiene    2500
# 110         Hygiene     500
# 111         Medical   51560
# 112       Nutrition   15000
# 113          Travel    2000
# 114          Travel    1500
df = df.drop(index=[100,105, 110])
print(df)
#           Category  Amount
# 101       Clothing    2000
# 102  Communication   12000
# 103    Electronics  200000
# 104  Entertainment    2000
# 106           Food    2000
# 107           Food    3000
# 108      Household   96000
# 109        Hygiene    2500
# 111        Medical   51560
# 112      Nutrition   15000
# 113         Travel    2000
# 114         Travel    1500