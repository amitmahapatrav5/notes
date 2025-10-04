# __str__ and __repr__ methods are defined for providing a string representation of the instance of your class
# By convention, __str__ is more of a Application User Specific String format like logging, wheres __repr__ is mostly used developer centric operation like debugging.
# When we pass the instance to print() or str(), python calls the functions in order (custom __str__, custom __repr__, default __repr__)
# default __repr__ is present to every class because every class in python inherits from object class and object class has that __repr__ method define.
# default __repr__ provides the name of the class and the location of the object in memory.

class Character:
    def __init__(self, *, first_name, middle_name, last_name):
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
    
    def __str__(self):
        return f'Character name is {self.first_name} {self.middle_name} {self.last_name}'
    
    def __repr__(self):
        return f"Character(first_name='{self.first_name}', middle_name='{self.middle_name}', last_name='{self.last_name}')"

sheldon = Character(first_name='Sheldon', middle_name='Lee', last_name='Cooper')

print(sheldon.__str__())
print(str(sheldon))
print(sheldon)
print(sheldon.__repr__())

class Series:
    pass

tbbt = Series()
print(tbbt)

# `object` class has an attribute called __class__ which returns the CLASS Object.
# Every CLASS Object has an attribute called __name__ which returns the name of the class as a string
# So the __repr__ method can be generalized as follow.
class Character:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'
    
sheldon = Character('Sheldon Lee Cooper')
print(sheldon)