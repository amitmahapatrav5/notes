# To make the instance of a class hashable we need to implement __hash__() in the class
# By default __hash__() is inherited from object class which is parent of all the classes in python
# This default __hash__() use the id of the object to find the hash
# But if we define __eq__() then python sets the __hash__ to None
# Why do we want to make our objects hashable? Because very often it happens to be the case that we need to add the objects in a dictionary or set.

class Character:
    def __init__(self, name):
        self.name = name
    
sheldon = Character('Sheldon Lee Cooper')
print(hash(sheldon))

class Character:
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return isinstance(other, Character) and self.name == other.name
    
sheldon = Character('Sheldon Lee Cooper')
cooper = Character('Sheldon Lee Cooper')
print(sheldon==cooper, sheldon is cooper) # True False

try:
    hash(sheldon)
except TypeError as ex:
    print(ex) # unhashable type: 'Character'

# The below code is inefficient because anyone can modify the name attribute of the object.
class Character:
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return isinstance(other, Character) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

# Correct way of doing this is
class Character:
    def __init__(self, name):
        self._name = name
    
    def __eq__(self, other):
        return isinstance(other, Character) and self.name == other.name
    
    # Only getter is defined to ensure this is a readonly attribute
    @property
    def name(self):
        return self._name
    
    def __hash__(self):
        return hash(self.name)

sheldon = Character('Sheldon Lee Cooper')
cooper = Character('Sheldon Lee Cooper')
print(sheldon==cooper, sheldon is cooper) # True False
print(hash(sheldon)==hash(cooper)) # True