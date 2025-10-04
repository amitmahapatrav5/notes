# Instance attributes are normally stored in a dictionary
# When we create lots of instances, lots of dictionary gets created
# In python 3.3, this process got slightly optimized by introduction of something called Key Sharing dictionary
# But we can optimize it further by using something called __slots__.
# On an average use of __slots__ can make the program 30% faster but the main usage of slots is getting the memory advantage
# When we use __slots__, python does not create instance dictionary to store the attributes
# Hence obj.__dict__ or vars(obj) throws error
# Attributes defined in slots can be created, reassign or deleted, without issue
# The downside of slots is, you cannot monkeypatch attributes to the object at runtime by assignment operator or by setattr()
# Also the use of __slots__ complicates the multiple inheritance and other aspects
# So only use slots when you know you will benefit substantially.

class Character:
    def __init__(self, *, name, age):
        self.name = name # This is allowed because we defined the same in __slots__
        self.age = age # This is allowed because we defined the same in __slots__

sheldon = Character(name='Sheldon', age=32)
print(sheldon.__dict__)
print(vars(sheldon))

class Character:
    __slots__ = ('name', 'age') # This can be any iterable
    def __init__(self, *, name, age):
        self.name = name # This is allowed because we defined the same in __slots__
        self.age = age # This is allowed because we defined the same in __slots__

sheldon = Character(name='Sheldon', age=32)
try:
    sheldon.__dict__
except AttributeError as ex:
    print(ex) # 'Character' object has no attribute '__dict__'

try:
    vars(sheldon)
except TypeError as ex:
    print(ex) # vars() argument must have __dict__ attribute

sheldon.name = 'Sheldon Lee Cooper'
print(sheldon.name) # Sheldon Lee Cooper
del sheldon.name
try:
    sheldon.name
except AttributeError as ex:
    print(ex) # 'Character' object has no attribute 'name'