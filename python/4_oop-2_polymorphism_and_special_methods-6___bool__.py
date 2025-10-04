# Every object has an associated boolean value
# In python every non 0 value is considered a Truthy value
# By default custom objects has Truthy value
# If we want to control that truthiness, then we need to implement __bool__()
# __bool__() must return a bool type value
# If __bool__() is not defined then python looks for __len__()
# If __len__() is defined and returns 0 then truthiness of the object is False
# If __len__() is not defined or __len__() returns a nonzero value, then truthiness of the object is True

num = []
print(bool(num)) # False
num.append(1)
print(bool(num)) # True

class Character:
    def __init__(self, name):
        self.name = name

class Casts:
    def __init__(self, *, series:str, characters: list[Character]):
        self._characters = characters
        self._series = series
    
    @property
    def series(self):
        return self._series
    
    @property
    def characters(self):
        return self._characters
    
    def __len__(self):
        return len(self.characters)

sheldon = Character('Sheldon Lee Cooper')
leonard = Character(' Leonard Hofstadter')

print(bool(sheldon), bool(leonard)) # True True

tbbt_casts = Casts(series='The Big Bang Theory', characters=[sheldon, leonard])
print(bool(tbbt_casts)) # True

# Just to demo that first __bool__() is called if exists, if not, then __len__() if not then True
class Casts:
    def __init__(self, *, series:str, characters: list[Character]):
        self._characters = characters
        self._series = series
    
    @property
    def series(self):
        return self._series
    
    @property
    def characters(self):
        return self._characters
    
    def __len__(self):
        return len(self.characters)
    
    def __bool__(self):
        return False if len(self) < 2 else True
        
sheldon = Character('Sheldon Lee Cooper')
tbbt_casts = Casts(series='The Big Bang Theory', characters=[sheldon])
print(bool(tbbt_casts)) # False
leonard = Character(' Leonard Hofstadter')
tbbt_casts = Casts(series='The Big Bang Theory', characters=[sheldon, leonard])
print(bool(tbbt_casts)) # True