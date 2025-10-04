# Sequence in which Python search for accessing an attribute
class Character:
    def __init__(self, name: str):
        self._name: str = name
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        self._name = new_name
    
    name = property(fget=get_name, fset=set_name)

sheldon = Character(name='Sheldon Cooper')

print(sheldon.__dict__) # {'_name': 'Sheldon Cooper'}

sheldon.__dict__['name'] = 'Sheldon Lee Cooper'

print(sheldon.__dict__) # {'_name': 'Sheldon Cooper', 'name': 'Sheldon Lee Cooper'}

print(sheldon.name) # still getter is called even though the instance __dict__ has the key name present in it.