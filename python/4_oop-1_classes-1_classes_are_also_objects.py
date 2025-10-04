# Object in Python can be considered as a Container while Class is like a Template used to create objects.
# Every Class in Python is also an Object. So every class also has a type.
# type of a CLASS Object is type
# type of type CLASS Object is type itself.
# This is related to Meta Programming in Python
# type() returns the CLASS Object, not a string

from datetime import date

class Character:
    def __init__(self, *, first_name, last_name, dob):
        self.first_name: str = first_name
        self.last_name: str = last_name
        self.dob: date = dob

    def work(self):
        pass

    def sleep(self):
        pass

sheldon = Character(first_name='Sheldon', last_name='Cooper', dob=date(year=1980, month=2, day=18))
print(type(sheldon))
print(type(sheldon) is Character)

print(type(Character), type(Character) is type, type(type))