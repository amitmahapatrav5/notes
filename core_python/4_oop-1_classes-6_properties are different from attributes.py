# What we define in the __init__ method are basically bare attributes.
# In many programming languages(e.g. Java), direct access to the attribute is highly discouraged.
# They basically make the attribute private and create getter and setter to access the attribute value.
# But  in python there is no concept of private attribute.
# But there is a generally accepted convention.
# If we have a attribute started with _, then that attribute is considered(but programmatically public only) as private. 
# Then we define getter and setter method for that attribute.
# But adding getter and setter method would change the class interface which is not a good.
# So to retain the interface and add getter and setter method, we can use a class called property
# You might notice that we did not have to change the code which is written to consume the class
# Hence in python we always start with bare attribute until we need such getter and setter

# Initially
class Character:
    def __init__(self, name: str):
        self.name: str = name

sheldon = Character(name='Sheldon Cooper')
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'


# Adding Attribute specific getter and setter methods
class Character:
    def __init__(self, *, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, *, new_name):
        self._name = new_name

# Class Interface got changed  
sheldon = Character(name='Sheldon Cooper')
print(sheldon.get_name())
sheldon.set_name(new_name='Sheldon Lee Cooper')


# Using property class
class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    name = property(fget=get_name, fset=set_name)

sheldon = Character(name='Sheldon Cooper')
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'



# Note: In above example name is a bare class attribute, not instance attribute
# It happens to be a special property object
# So name is a property but _name is an attribute
# We can still access the underlying variable _name

from pprint import pprint

class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    name = property(fget=get_name, fset=set_name)

sheldon = Character(name='Sheldon Cooper')
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'

print(sheldon._name)

print(sheldon.__dict__) # {'_name': 'Sheldon Lee Cooper'}

pprint(Character.__dict__)