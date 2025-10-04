# When we inherit from another class, we inherit it's attributes including all the callables
# We choose to redefine these existing callables in the subclass. This is called overriding.
# We can override any callable in the parent class, including the callables which the parent class inherited from it's parents

class Actor:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.__repr__() if self.__repr__ else f'Actor name is {self.name}'

class Character(Actor):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

sheldon = Character('Sheldon Lee Cooper')
print(sheldon) # Character(name=Sheldon Lee Cooper)


# How inheritance work in case of fallback function
# Example __str__ is defined in parent class and __repr__ is defined in child class.
class Actor:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return f'Actor name is {self.name}'

class Character(Actor):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'
    
sheldon = Character('Sheldon Lee Cooper')
print(sheldon) # Actor name is Sheldon Lee Cooper