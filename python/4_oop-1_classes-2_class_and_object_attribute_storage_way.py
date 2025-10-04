# As Class is also an object, it also has attributes and behavior.
# CLASS Object attribute can be a Data Attribute or Callable Type Attribute just like custom class instance attribute.
# Only difference is the behavior of CLASS Object is defined in type class, not by us. This is meta programming.

# Attributes of a CLASS Object is stored in a dictionary like data structure called mappingproxy.
# mappingproxy is kind of a readonly dictionary. So we cannot directly mutate it.
# However as mappingproxy is a dict like object, we can access the values using it's key just like dict object.
# The attributes of mappingproxy are string, hence hashable, hence, all the attributes of a class in python has to be a valid string.
# Even though mappingproxy is a immutable dict like data structure, still we can mutate it indirectly.
# We can access mappingproxy using the __dict__ property of the CLASS Object.
# But not everything is stored in mappingproxy, e.g. `__name__`

# Unlike CLASS Object attributes, the attributes of the instance of any class is stored in a plane dictionary.
# Hence we can mutate the dictionary directly or indirectly


from datetime import date
from pprint import pprint

class TBBTCharacter:
    series_name = 'The Big Bang Theory' # Data Attribute
    # Callable Type Attribute
    def get_season_count():
        return 12

    def __init__(self, first_name, middle_name, last_name, dob):
        self.first_name: str = first_name
        self.middle_name: str = middle_name
        self.last_name: str = last_name
        self.dob: date = dob

pprint(TBBTCharacter.__dict__, type(TBBTCharacter.__dict__))

TBBTCharacter.__dict__['series_name']

TBBTCharacter.get_season_count(), TBBTCharacter.__dict__['get_season_count'](), getattr(TBBTCharacter, 'get_season_count')()

setattr(TBBTCharacter, 'side_character_count', 18)
pprint(TBBTCharacter.__dict__)
del TBBTCharacter.side_character_count
pprint(TBBTCharacter.__dict__)


print(TBBTCharacter.__name__)
print(getattr(TBBTCharacter, '__name__'))
try:
    print(TBBTCharacter.__dict__['__name__'])
except KeyError as ex:
    print(f'Key {ex} does not exists.')



sheldon = TBBTCharacter(first_name='Sheldon', 
                        middle_name= 'Lee', 
                        last_name='Cooper', 
                        dob=date(year=1980, month=2, day=18))

pprint(sheldon.__dict__)
print(type(sheldon.__dict__))
sheldon.__dict__['nick_name'] = 'Moon Pie'
pprint(sheldon.__dict__)
print(sheldon.nick_name)
del sheldon.__dict__['nick_name']
try:
    print(sheldon.nick_name)
except AttributeError as ex:
    print(ex)

print(sheldon.__class__)
print(getattr(sheldon, '__class__'))
try:
    print(sheldon.__dict__['__class__'])
except KeyError as ex:
    print(f'Key {ex} does not exists.')
