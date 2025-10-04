# Main reason of using getter and setter is, to ensure valid values are getting assigned to the attributes.
class Character:
    def __init__(self, name: str):
        self.set_name(name)
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name):
        if isinstance(new_name, str) and len(new_name.strip()) > 0:
            self._name = new_name
        else:
            raise ValueError('Name should be a non empty string.')
    
    name = property(fget=get_name, fset=set_name)

sheldon = Character(name='Sheldon Cooper')
print(sheldon.name)
sheldon.name = 'Sheldon Lee Cooper'
print(sheldon.name)

# May be its something that we have to derive. 
# e.g. Age can be derived from dob.
# e.g. Area and Perimeter can be derived from the radius of a circle.
