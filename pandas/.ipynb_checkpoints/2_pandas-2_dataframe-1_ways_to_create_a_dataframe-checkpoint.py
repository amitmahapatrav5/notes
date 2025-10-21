# Series is like a column of an excel sheet [1D]
# Dataframe is like the sheet in an excel sheet [2D]

# Series has 2 types of index(implicit which is positional and explicit, also called label)
# Similarly Dataframe has 2 types of index in each dimension
#       0   1   2   3   4  <= Implicit Index
#       C1  C2  C3  C4  C5 <= Explicit Index
# 0 R1  11  12  13  14  15
# 1 R2  21  22  23  24  25
# 2 R3  31  32  33  34  35
# 3 R4  41  42  43  44  45
# 4 R5  51  52  53  54  55
# ^ ^ Explicit Index
# | Implicit Index

# We can create explicit index for rows and columns of the dataframe
# If we don't provide explicit index, then pandas is going to create RangeIndex object by taking positional index as reference.

# Dataframe can be created from
    # List of Lists
    # List of Series
    # Dict of Lists
    # Dict of Series
    # List of Dicts
    # Dict of Dicts
    # Other sources like csv, excel file etc

# 2 Common Observation form the above
    # If you are giving everything encapsuled in a LIST
    # Pandas will create the dataframe as Vertical Stack
        # [List Item 1]
        # [List Item 2]
        # [List Item 3]
    # If you are giving everything encapsuled in a DICTIONARY
    # Pandas will create the dataframe as Horizontal Stack
        # [List Item 1][List Item 2][List Item 3]

# In few of the above forms, 4 cases are possible
# And the above observation will help in identifying in which case we are providing which index
    # ROW Index YES | COLUMN Index YES [List of Series(name, index), Dict of Series(index)]
    # ROW Index YES | COLUMN Index NO  [List of Series(name)]
    # ROW Index NO  | COLUMN Index YES [Dict of Lists, List of Series(index)]
    # ROW Index NO  | COLUMN Index NO  [List of Lists, List of Series()]
# Turns out that if we are using Dict ot List of Series object, we have a way to provide both ROW and COLUMN Index
# Below I have shown 3 scenarios, but a lot more is also possible

# From List of Lists
# Explicit Row Index NOT Provided
# Explicit Column Index NOT Provided
# So pandas is going to create an Index object like 0,1,2... for both ROW and COLUMN
import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

df = pd.DataFrame([categories, sub_categories, amounts])

print(df)
#               0         1              2   ...          12      13      14
# 0  Administrative  Clothing  Communication  ...   Nutrition  Travel  Travel
# 1         Banking   Top Wear       Internet  ...  Supplement     Bus   Train
# 2            1000      2000          12000  ...       15000    2000    1500

print(df.T)
#                  0                 1       2
# 0   Administrative           Banking    1000
# 1         Clothing           Top Wear    2000
# 2    Communication          Internet   12000
# 3      Electronics           Handset  200000
# 4    Entertainment  OTT Subscription    2000
# 5             Food         Vegetable   10000
# 6             Food            Fruits    2000
# 7             Food             Dairy    3000
# 8        Household         House Rent   96000
# 9          Hygiene          Haircare    2500
# 10         Hygiene          Skincare     500
# 11         Medical         Insurance   51560
# 12       Nutrition        Supplement   15000
# 13          Travel               Bus    2000
# 14          Travel             Train    1500


# From List of Series
# Explicit Row Index IS Provided [index param of Series Object]
# Explicit Column Index NOT Provided [name param of Series Object]
# So pandas will use the given as index only
import pandas as pd

categories = ['Administrative', 'Clothing', 'Communication', 'Electronics', 'Entertainment', 'Food', 'Food', 'Food', 'Household', 'Hygiene', 'Hygiene', 'Medical', 'Nutrition', 'Travel', 'Travel']
sub_categories = ['Banking', 'Top Wear', 'Internet', 'Handset', 'OTT Subscription', 'Vegetable', 'Fruits', 'Dairy', 'House Rent', 'Haircare', 'Skincare', 'Insurance', 'Supplement', 'Bus', 'Train']
amounts = [1000, 2000, 12000, 200000, 2000, 10000, 2000, 3000, 96000, 2500, 500, 51560, 15000, 2000, 1500]

idx = pd.Index(range(100,115))

categories = pd.Series(
    data= categories,
    index= idx,
    name= 'Category'
)

sub_categories = pd.Series(
    data= sub_categories,
    index= idx,
    name= 'Sub Category'
)

amount = pd.Series(
    data= amounts,
    index= idx,
    name= 'Amount'
)

df = pd.DataFrame([categories, sub_categories, amount])
print(df)
#                          100       101  ...     113     114
# Category      Administrative  Clothing  ...  Travel  Travel
# Sub Category         Banking   Top Wear  ...     Bus   Train
# Amount                  1000      2000  ...    2000    1500

print(df.T)
#            Category      Sub Category  Amount
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


# Using Dictionary of Lists
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

print(df)
#           Category      Sub Category  Amount
# 0   Administrative           Banking    1000
# 1         Clothing           Top Wear    2000
# 2    Communication          Internet   12000
# 3      Electronics           Handset  200000
# 4    Entertainment  OTT Subscription    2000
# 5             Food         Vegetable   10000
# 6             Food            Fruits    2000
# 7             Food             Dairy    3000
# 8        Household         House Rent   96000
# 9          Hygiene          Haircare    2500
# 10         Hygiene          Skincare     500
# 11         Medical         Insurance   51560
# 12       Nutrition        Supplement   15000
# 13          Travel               Bus    2000
# 14          Travel             Train    1500


# Dataframe can also be created from other external ways
# Among them csv and excel is mostly used.

# Params for reading data from csv and excel is pretty much same
# Important parameters
    # io, sheet_name => positional param => only for read_excel
    # filepath_or_buffer => positional param => only for read_csv
    # sep => only for read_csv
    # COMMONS
    # usecols => the columns which will be used to create dataframe
    # names => we can rename the columns
    # index_col => which column needs to be used as index
    # converters => converting the data in some way

from datetime import datetime
import pandas as pd

df = pd.read_csv(
    'crape_diem.csv',
    sep=',',
    header=0,
    usecols=[0,1,3],
    names={0: 'Date', 1:'Time', 3:'Is Valid'},
    index_col=0,
    converters={
        1: lambda dt: datetime.strptime(dt, format='').time(),
        3: lambda entry: True if entry=='Filled' else False
    }
)

df = pd.read_excel(
    'crape_diem.excel', 0,
    header=0,
    usecols=[0,1,3],
    names={0: 'Date', 1:'Time', 3:'Is Valid'},
    index_col=0,
    converters={
        1: lambda dt: datetime.strptime(dt, format='').time(),
        3: lambda entry: True if entry=='Filled' else False
    }
)