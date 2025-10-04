# property is a class
# property(fget=getter_func, fset=setter_func, fdel='deleter_func')
# the docstring is only collected from getter function

class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    def del_name(self):
        del self._name

    name = property(fget=get_name, fset=set_name, fdel=del_name, doc= 'Doc string for name')

sheldon = Character(name='Sheldon Cooper')

print(help(Character.name))
print(sheldon.__dict__)
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'
print(sheldon.__dict__)
del sheldon.name
print(sheldon.__dict__)
try:
    print(sheldon.name)
except AttributeError as ex:
    print(ex)


# property class has getter, setter and deleter function attributes
# getter, setter and deleter returns a property object when called.
class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    def del_name(self):
        del self._name
    
    # name = property(fget=get_name, fset=set_name, fdel=del_name)

    # Interesting Part
    name = property(doc='Doc String for name')
    name = name.getter(get_name)
    name = name.setter(set_name)
    name = name.deleter(del_name)

sheldon = Character(name='Sheldon Cooper')

print(help(Character.name))
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'
print(sheldon.name)
del sheldon.name
print(sheldon.__dict__)
try:
    print(sheldon.name)
except AttributeError as ex:
    print(ex)


# 1st parameter of property() takes the getter function.
class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    def del_name(self):
        del self._name
    
    # Changes here
    name = property(get_name, doc='Doc String for getter')
    name = name.setter(set_name)
    name = name.deleter(del_name)

sheldon = Character(name='Sheldon Cooper')

print(help(Character.name))
print(sheldon.name)
print(sheldon.name)
del sheldon.name
print(sheldon.__dict__)

# The above follow the decorator syntax
# 1. Takes a function
# 2. Decorates the function
# 3. returns the function with the same name as given function
# This is why the name of getter, setter and deleter function is same, in this case - name
class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    @property
    def name(self):
        'Doc String for name'
        return self._name
    
    @name.setter
    def name(self, new_name):
        self._name = new_name
    
    @name.deleter
    def name(self):
        del self._name

sheldon = Character(name='Sheldon Cooper')

print(help(Character.name))
print(sheldon.name)
print(sheldon.name)
del sheldon.name
print(sheldon.__dict__)

# When we call property.getter(fn) method, it creates a new attribute called fget in the property object and that attribute points to fn.
# Same thing happens in case of property.setter(fn) and property.deleter(fn)

def get_prop(arg):
    print('Property Getter')
    
def set_prop(arg):
    print('Property Setter')
    
def del_prop(arg):
    print('Property Deleter')

prop = property()

print(type(prop.fget)) # NoneType
print(type(prop.fset)) # NoneType
print(type(prop.fdel)) # NoneType

prop_get = prop.getter(get_prop)

print(type(prop_get.fget)) # Function
print(type(prop_get.fset)) # NoneType
print(type(prop_get.fdel)) # NoneType

prop_set = prop_get.setter(set_prop)

print(type(prop_set.fget)) # Function
print(type(prop_set.fset)) # Function
print(type(prop_set.fdel)) # NoneType

prop_del = prop_set.deleter(del_prop)

print(type(prop_del.fget)) # Function
print(type(prop_del.fset)) # Function
print(type(prop_del.fdel)) # Function