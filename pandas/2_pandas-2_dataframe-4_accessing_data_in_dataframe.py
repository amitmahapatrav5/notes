# It is a must to understand different ways to create dataframe
# Also the observation that is deduced when we give "Everything in a List" vs "Everything in a Dict"

# Dataframe is a 2D Object
# So common sense days, to access a cell, we need syntax like df[][]
# This is fixed by pandas team => df[column][row] => ALWAYS
# Question is whether column and row should be explicit index or implicit
# This is also fixed by pandas team => EXPLICIT INDEX
# But this is kinda something that we are remembering
# And code is not clean
# HENCE, Pandas team came up with loc and iloc, YES, This is also present in Series
# loc is Location => Explicit Index
# iloc is Implicit Location => Implicit Index
# Using loc and iloc is preferable and cleaner way to write code
# Only one thing pandas team did in a reverse way => This is what I think
# They have fixed => df[column][row]
# BUT, WHEN IT COMES TO loc and iloc, the syntax is
# df.loc[row, column] and df.iloc[row, col] , NOT COLUMN and ROW
# One way this is straight is because this ROW, COL way, we access a matrix
# But it is opposite to the convention they fixed initially

# Often in pandas dataframe, we pick a single column.
# In that case, it is very common to use df[column label] syntax.
# Just know that if df[something(single)] => returns a series and something is explicit