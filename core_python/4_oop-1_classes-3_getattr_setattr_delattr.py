# getattr(), setattr() and delattr() are built-in functions

# In getattr(), if attribute does not exists, then Python raise AttributeError.
# To get away from AttributeError, we can specify default Parameter
# Shorthand way - dot notation - object.prop. But we cannot specify default in this notation.

# In setattr - Shorthand way - dot notation - object.prop = value
# If attribute prop does not exists, it will create, else will update

# In delattr - If attribute does not exists, then Python raise AttributeError
# Shorthand way - del keyword - del object.prop

# adding attributes to built-in class(e.g. list or str) or the instance of the class in not possible

from datetime import date

class TBBTCharacter:
    series: str = 'The Big Bang Theory'

    def __init__(self, first_name, middle_name, last_name, dob):
        self.first_name: str = first_name
        self.middle_name: str = middle_name
        self.last_name: str = last_name
        self.dob: date = dob



# getattr
print(getattr(TBBTCharacter, 'series'))

try:
    print(getattr(TBBTCharacter, 'season'))
except AttributeError as ex:
    print(ex)

print(getattr(TBBTCharacter, 'season', None))

print(TBBTCharacter.series)

try:
    print(TBBTCharacter.season)
except AttributeError as ex:
    print(ex)



# setattr
setattr(TBBTCharacter, 'season_count', 12)
print(getattr(TBBTCharacter, 'season_count'))

TBBTCharacter.main_character_count = 7
print(TBBTCharacter.main_character_count)



# delattr
setattr(TBBTCharacter, 'dummy_attr', 'dum_val')
print(getattr(TBBTCharacter, 'dummy_attr'))
delattr(TBBTCharacter, 'dummy_attr')
print(getattr(TBBTCharacter, 'dummy_attr', None))

try:
    delattr(TBBTCharacter, 'dummy_attr')
except AttributeError as ex:
    print(ex)

setattr(TBBTCharacter, 'dummy_attr', 'dum_val')
print(getattr(TBBTCharacter, 'dummy_attr'))
del TBBTCharacter.dummy_attr
print(getattr(TBBTCharacter, 'dummy_attr', None))



# built-in
seasons = [season for season in range(1,13)]
sheldon = 'Sheldon Cooper'

print(type(seasons))
try:
    print(seasons.__dict__)
except AttributeError as ex:
    print(ex)
    
print(type(sheldon))
try:
    print(sheldon.__dict__)
except AttributeError as ex:
    print(ex)

print(getattr(seasons, '__class__'))