# Basically to support format(obj, your_format) call, we can include __format__()
# Fallback sequence, __format__() > __str__() > __repr__()
# Method Signature __format__(self, format)
# If __format__() is not defined, then call like format(obj), not format(obj, your_format)

from datetime import datetime

class Character:
    def __init__(self, name, dob: datetime):
        self.name = name
        self.dob = dob
    
    def __repr__(self):
        return f'Character(name={self.name}, dob={format(self.dob, "%Y-%m-%d")})'
    
    def __str__(self):
        return f'My Name is {self.name}, born on {format(self.dob, "%d %B %Y")}'

    def __format__(self, dob_format):
        return f'Character(name={self.name}, dob={format(self.dob, dob_format)})'


sheldon = Character('Sheldon Lee Cooper', datetime(year=1926, month=2, day=15))

print(format(sheldon, '%Y-%m-%d %H-%M-%S'))
print(format(sheldon))