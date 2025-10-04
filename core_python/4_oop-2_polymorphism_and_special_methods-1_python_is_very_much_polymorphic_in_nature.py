# Polymorphism: Ability to define a generic type of behavior which will behave differently when applied to different types.
# Example 1+2=3 but 'Sheldon' + ' Cooper' = 'Sheldon Cooper'

# Python is very much Polymorphic in nature
# Example: As long as a class implements an iterator protocol, we can iterate over it. It does not matter what the object is. It can be a list or tuple or set or even any custom class.
# Similarly other operators(+, -, *, / etc) in python are polymorphic. Because these operate over different types of operands like integer, float, decimal, list, tuples etc.
# We can support the same functionality by implementing some dunder methods. Example to support + operator, we need to implement __add__ method.

# for initializing a class
# __init__

# For Implementing Context Manager
# __start__
# __end__

# For Sequence Types
# __getitem__
# __setitem__
# __delitem__

# For Iterable and Iterator
# __iter__
# __next__

# To implement len()
# __len__

# To implement `in` functionality
# __contains__