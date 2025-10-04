# Special Methods for Arithmetic Operator Support
# __add__ for +
# __sub__ for -
# __mul__ for *
# __truediv__ for /
# __floordiv__ for //
# __mod__ for %
# __pow__ for **
# __matmul__ for @ => This is introduced in Python 3.5 for better numpy support which they use for matrix multiplication

# When python executes a+b it basically calls the __add__ method binded to instance a, a.__add__(b)
# Iff a.__add__(b) returns NotImplemented AND operands are not of same type, then python calls b.__radd__(a)

# Corresponding Reflected Operators
# __radd__
# __rsub__
# __rmul__
# __rtruediv__
# __rfloordiv__
# __rmod__
# __rpow__
# __rmatmul__

# Special Methods for Inplace Operator Support
# Inplace Operators, typically mutate the object. But it depends on the object
l = [1,2,3]
print(id(l))
l += [4,5]
print(id(l))

t = (1,2,3)
print(id(t))
t += (4,5)
print(id(t))
# __iadd__ for +=
# __isub__ for -=
# __imul__ for *=
# __itruediv__ for /=
# __ifloordiv__ for //=
# __imod__ for %=
# __ipow__ for **=

# Special Methods for Unary Operator Support
# __neg__ for -a
# __pos__ for +a
# __abs__ for abs(a)

from numbers import Real
from math import sqrt

class Vector:
    def __init__(self, *components):
        if len(components)<1:
            raise ValueError('Vector must have at least one component.')
        for component in components:
            if not isinstance(component, Real):
                raise TypeError('Each component must be a real number.')
        self._components = tuple(components)
    
    def __len__(self):
        return len(self._components)
    
    @property
    def components(self):
        return self._components

    def __repr__(self):
        return f'Vector{self.components}'

    def _validate_type_and_dim(self, other):
        return isinstance(v, Vector) and len(v) == len(self)

    def __add__(self, other):
        if self._validate_type_and_dim(other):
            components=[c1+c2  for c1, c2 in zip(self.components, other.components)]
            return Vector(*components)
        else:
            return NotImplemented

    def __sub__(self, other):
        if self._validate_type_and_dim(other):
            components=[c1-c2  for c1, c2 in zip(self.components, other.components)]
            return Vector(*components)
        else:
            return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, Real):
            # scalar product
            components = [component*other for component in self.components]
            return Vector(*components)
        elif self._validate_type_and_dim(other):
            # dot product
            return sum([c1*c2  for c1, c2 in zip(self.components, other.components)])
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        return self * other
    
    def __iadd__(self, other):
        # return self + other # But this did not perform the operation inplace
        if self._validate_type_and_dim(other):
            components=[c1+c2  for c1, c2 in zip(self.components, other.components)]
            self._components=tuple(components)
            return self
        return NotImplemented
    
    def __neg__(self):
        components = [-component for component in self._components]
        return Vector(*components)

    def __abs__(self):
        components = [component**2 for component in self._components]
        return sqrt(sum(components))

try:
    Vector()
except ValueError as ex:
    print(ex)

try:
    Vector(1,2,3,'abc')
except TypeError as ex:
    print(ex)


v = Vector(1,2,3,4,5,6)
s = Vector(2,4,6,8,10,12)
t = Vector(3,6,9,12,15,18)

print(v)

print(v.components)
print(len(v))

print(v+s)
print(v+s+t)

print(t-s)

print(v*10) # v.__mul__(10)
print(10*v) # 10.__mul__(v) => v.__rmul__(10)

print(v*s)

print(v, id(v))
v+=s
print(v, id(v))

print(-v)

print(abs(v))