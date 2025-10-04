# We can make an object of a class callable in python.
# Why we need this? Not very much clear to me, though I have few use case which is mostly for readability purpose.
# To make an object callable, we need to implement __call__ method in the class.
# This is a good example which supports the fact why, in python, we talk about callable in general rather than functions only.

class Character:
    def __init__(self, name):
        self.name = name
    
    def __call__(self):
        return f'My name is {self.name}'
    
sheldon = Character('Sheldon Lee Cooper')
print(sheldon())


# callable() is a builtin function available in python which takes an object in argument and returns whether the object is a callable or not
print(callable(print))
print(callable(Character))
print(callable(sheldon))
print(callable(True))


# Use case 1
# partial class present in functools module builtin python has __call__ method implemented, so the objects of partial class is callable.
from functools import partial

def add_abc(a, b, c):
    return a+b+c

add_abc = partial(add_abc, 10, 20) # Looks somewhat like a decorator, but not exactly
print(add_abc(30))

# Use case 2
# I want to track how many times a function is called
# And the way I can access that number is using the function itself

# Functional Programming Approach
from functools import wraps

def track(fn):
    _counter = 0
    
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal _counter
        _counter+=1
        return fn(*args, **kwargs)
    
    def counter():
        return _counter
    
    inner.counter = counter # This is very crucial to understand that this is a closure
    return inner

@track
def add_nums(*args):
    return sum(args)

add_nums(1,2,3)
add_nums(2,3)
print(add_nums.counter()) # Syntax looks a bit weird and the code setup is also a bit complected

# OOP Approach
class Track:
    def __init__(self, fn):
        self._counter = 0
        self._fn = fn
    
    def __call__(self, *args, **kwargs):
        self._counter+=1
        return self._fn(*args, **kwargs)
    
    counter = property(fget= lambda self: self._counter)

@Track
def add_nums(*args):
    return sum(args)

# The code here looks clean because the object of Track class is callable.
# Although this seems like add_num is a function, but it is actually an instance of Track call
add_nums(1,2,3)
add_nums(2,3)
print(add_nums.counter)
print(type(add_nums)) # <class '__main__.Track'>