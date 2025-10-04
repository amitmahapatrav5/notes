# It is preferable to call super().__init__() first and then do the assignment. Otherwise overriding issue may come.
# generally super() is used for calling the __init__() of the parent class
# But it can be used for anything that is defined in parent class
# When a function defined in parent class is called using super() from child, the same object is passed.
class Actor:
    def __init__(self, name):
        print(id(self)) # 133345895666880 [EXACT SAME]
        self.real_name = name

class Character(Actor):
    def __init__(self, *, real_name, character_name):
        print(id(self)) # 133345895666880 [EXACT SAME]
        super().__init__(real_name)
        self.character_name = character_name

sheldon = Character(real_name='Jim Parson', character_name='Sheldon Lee Cooper')

# Whenever we are using any calling any callable in the parent using super() in child, the callable that is bound to the instance is run.
class Actor:
    def __init__(self, realname):
        self.realname = realname
    
    def eat(self):
        print('Eat Healthy')

    def work(self):
        print('Go to set')
    
    def sleep(self):
        print('Sleep')

    def routine(self):
        self.eat()
        self.work()
        self.sleep()

class TBBTCharacter(Actor):
    def __init__(self, *, realname, character_name):
        super().__init__(realname)
        self.character_name = character_name

    def eat(self):
        print('Go to Cheese Cake Factory')

    def work(self):
        print('Go to University')
    
    def routine(self):
        super().routine()

sheldon = TBBTCharacter(realname='Jim Parson', character_name='Sheldon Cooper')
sheldon.routine()
# Go to Cheese Cake Factory [This is binded to Character Object]
# Go to University [This is binded to Character Object]
# Sleep [This is not present in Character class so, inherited from Actor and binded to Character object and ran]

# super() looks for an attribute(data or callable) up the inheritance hierarchy chain
class Actor:
    def get_skills(self):
        print('Skill 1, Skill 2, Skill 3')

class Character(Actor):
    pass

class TBBTCharacter(Character):
    def get_skills(self):
        super().get_skills()

sheldon = TBBTCharacter()
sheldon.get_skills() # Skill 1, Skill 2, Skill 3

# No need to use super() if there is no ambiguity

class Actor:
    def breakfast(self):
        print('Have Breakfast')
    
class Character(Actor):
    def lunch(self):
        print('Have Lunch')
    
class TBBTCharacter(Character):
    def dinner(self):
        print('Have dinner')

    def eat(self):
        self.breakfast() # super().breakfast() would also work exactly same way
        self.lunch() # super().lunch() would also work exactly same way
        self.dinner()

sheldon = TBBTCharacter()
sheldon.eat()

# IMPORTANT EXAMPLE
from numbers import Real
from math import pi

class Circle:
    def __init__(self, r):
        self.radius = r
        self._area = None
        self._perimeter = None

    @property
    def radius(self):
        return self._r

    @radius.setter
    def radius(self, r):
        if isinstance(r, Real) and r > 0:
            self._r = r
            self._area = None
            self._perimeter = None
        else:
            raise ValueError('Radius must a positive real number.')

    @property
    def area(self):
        if self._area is None:
            self._area = pi * self.radius ** 2
        return self._area

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = 2 * pi * self.radius
        return self._perimeter


class UnitCircle(Circle):
    def __init__(self):
        super().__init__(1)
        
    @property
    def radius(self):
        super().radius

u = UnitCircle() # this will raise Attribute error. VERY IMPORTANT TO UNDERSTAND

# SOLUTION
from numbers import Real
from math import pi

class Circle:
    def __init__(self, r):
        self.radius = r
        self._area = None
        self._perimeter = None

    def get_radius(self):
        return self._r

    def set_radius(self, r):
        if isinstance(r, Real) and r > 0:
            self._r = r
            self._area = None
            self._perimeter = None
        else:
            raise ValueError('Radius must a positive real number.')

    radius = property() \
            .getter(get_radius) \
            .setter(set_radius)
    
    @property
    def area(self):
        if self._area is None:
            self._area = pi * self.radius ** 2
        return self._area

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = 2 * pi * self.radius
        return self._perimeter

c = Circle(10)
print(c.radius)
c.radius = 100
print(c.radius)

class UnitCircle(Circle):
    def __init__(self):
        super().__init__(1)

    def get_radius(self):
        return super().radius
    
    def set_radius(self, r):
        if not getattr(self, '_r', None):
            self._r = r
        else:
            raise ValueError('Cannot change the radius of UnitCircle')
    
    radius = property() \
            .getter(get_radius) \
            .setter(set_radius)


u = UnitCircle()
print(u.radius)
try:
    u.radius = 100
except ValueError as ex:
    print(ex)
print(u.radius)