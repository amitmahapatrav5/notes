# There could be below 4 possible combo (without considering the overlapping fact)
# Scenario 1: Parent has __slots__ Child does't have
# Scenario 2: Parent has __slots__ Child has __slots__ too
# Scenario 3: Parent doesn't have __slots__ Child does't have [Normal Inheritance]
# Scenario 4: Parent doesn't have __slots__ Child has __slots__ though


# Scenario 1: Parent has __slots__ Child does't have
# When we use __slots__, we don't get any instance dictionary
# But the inherited class will have an instance dictionary unless, it declares a __slot__
class Actor:
    __slots__ = 'realname',
    def __init__(self, *, realname):
        self.realname = realname

class Character(Actor):
    def __init__(self, *, charname):
        self.charname = charname

jim = Actor(realname='Jim Parson')
sheldon = Character(charname='Sheldon Cooper')

try:
    jim.__dict__ # we don't have __dict__ in Actor, because we have __slots__ defined in there
except AttributeError as ex:
    print(ex)
print(sheldon.__dict__) # we have __dict__, because we don't have __slots__ directly defined in the class

# Scenario 2: Parent has __slots__ Child has __slots__ too
class Actor:
    __slots__ = 'realname',
    def __init__(self, *, realname):
        self.realname = realname

class Character(Actor):
    __slots__ = 'charname',
    def __init__(self, *, charname):
        self.charname = charname

jim = Actor(realname='Jim Parson')
sheldon = Character(charname='Sheldon Cooper')

try:
    jim.__dict__
except AttributeError as ex:
    print(ex)

try:
    sheldon.__dict__
except AttributeError as ex:
    print(ex)

# Scenario 4: Parent doesn't have __slots__ Child has __slots__ though
# Both Parent and Child instance will have instance dictionary
class Actor:
    def __init__(self, *, realname):
        self.realname = realname

class Character(Actor):
    __slots__ = 'charname',
    def __init__(self, *, charname):
        self.charname = charname

jim = Actor(realname='Jim Parson')
sheldon = Character(charname='Sheldon Cooper')

print(jim.__dict__)
print(sheldon.__dict__)


# Overlapping in Scenario 2: Parent has __slots__ Child has __slots__ too


# Overlapping in Scenario 4: Parent has __slots__ Child has __slots__ too
# Technically you can, it will not fail, but in future a check may be added in python to prevent this. So this is not really recommended.
class Character:
    def __init__(self, name):
        self.name = name
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        if name and name.strip():
            self._name = name.strip().title()
        else:
            raise ValueError('Name should be a non empty string')
    
class TBBTCharacter(Character):
    __slot__ = 'name',
    def __init__(self, name):
        self._name = name
        
sheldon = Character('  sheldon lee cooper  ')
cooper = TBBTCharacter('  sheldon    ')

print(sheldon.name)
print(cooper.name)


# __slots__ are efficient from time and memory point of view
# instance __dict__ are providing more dynamic feature
# we can bring the best of both the worlds by adding __dict__ to __slots__
class Family:
    __slots__ = '_sigil', '__dict__'
    def __init__(self, sigil):
        self.sigil = sigil
    
    @property
    def sigil(self):
        return self._sigil
    
    @sigil.setter
    def sigil(self, new_sigil):
        if new_sigil and new_sigil.strip():
            self._sigil = new_sigil.strip().title()
        else:
            raise ValueError('Sigil must be a non empty string')
    
class Stark(Family):
    def __init__(self, *, sigil, name):
        super().__init__(sigil)
        self.name = name
    
ned = Stark(sigil='direwolf', name='Edard Stark')

print(ned.sigil)
print(ned.name)
print(ned.__dict__)
ned.honor = 'Hand of the king'
print(ned.__dict__)

# How __slots__ attributes are different than properties
# __slots__ are not stored in the instance dictionary
# properties are stored in the instance dictionary
# then how they are different.
# more on this in descriptors