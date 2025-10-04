# Each instance of any class in python has it's own namespace. __dict__ attribute of the instance points to it.
# dot notation and getattr both look for the attributes in the same place.
# If its CLASS Object, searching happens in mappingproxy pointed by class_obj.__dict__
# If its instance of a class, searching happens in the dict pointed by instance.__dict__
# In case any attribute is missing in the instance.__dict__, Python looks for the same attribute in corresponding class_obj.__dict__. There are some other places also where it looks. More on this in inheritance and Meta Programming.

from pprint import pprint

class TBBTCharacter:
    series = 'The Big Bang Theory'
    debut_season = 1
    
    def __init__(self, *, first_name, last_name):
        self.first_name: str = first_name
        self.last_name: str = last_name

sheldon = TBBTCharacter(first_name = 'Sheldon',
                    last_name = 'Cooper')

bernadette = TBBTCharacter(first_name = 'Bernadette',
                        last_name = 'Rostenkowski-Wolowitz')

pprint(TBBTCharacter.__dict__)

pprint(sheldon.__dict__)
pprint(bernadette.__dict__)
print(sheldon.debut_season)
print(bernadette.debut_season)

bernadette.debut_season = 3

pprint(sheldon.__dict__)
pprint(bernadette.__dict__)
print(sheldon.debut_season)
print(bernadette.debut_season)