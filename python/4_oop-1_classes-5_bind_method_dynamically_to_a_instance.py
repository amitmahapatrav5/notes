# If we define a function attribute in a class then all instances created from that class will have that as a bound method.
# If we try to add a function as an attribute to an instance, it still will be a function not a method.
# But we can create a method object and bind that to a specific instance using MethodType class from type module.
# The bound method will also reflect in the instance.__dict__

from types import MethodType
from pprint import pprint

class Character:
    def __init__(self, *, name, field_of_study):
        self.name = name
        self.field_of_study = field_of_study
    
    def work(self):
        return f'Work on {self.field_of_study}'

sheldon = Character(name ='Sheldon Lee Cooper', field_of_study= 'string theory')
amy = Character(name= 'Amy Farrah Fowler', field_of_study= 'neuroscience')

print(sheldon.work(), amy.work())

sheldon.designation = 'Apartment Leader'
sheldon.update_roommate_agreement = MethodType(lambda self: f'Add clause: Sheldon as {self.designation}', sheldon)
print(type(sheldon.update_roommate_agreement), sheldon.update_roommate_agreement())

pprint(sheldon.__dict__)