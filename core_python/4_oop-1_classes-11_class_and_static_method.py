# In python class method and static method are not complected
# End of the day, all of it boils down to the fact what the function is bound to.
# There are basically 3 combinations.
# 1. Function can be bound to instance of the class - Instance Method
# 2. Function can be bound to CLASS Object itself - Class Method
# 3. Function that will not be bound to neither the CLASS Object nor the instance object - Static Method

# There are not much of a use case of a static method. Because the definition says, it does not depend on the class or the instance. So it can easily be placed outside the class.

# Thing to note here is is which object can access which method.

class Character:
    
    def greet_instance(self):
        print(f'{self} says Hello!!')
    
    @classmethod
    def greet_class(cls):
        print(f'{cls} says Hello!!')
    
    @staticmethod
    def greet():
        print(f'Someone says Hello!!')
    
sheldon = Character()


print(type(sheldon.greet_instance)) # <class 'method'>
print(type(Character.greet_instance)) # <class 'function'>
print(type(sheldon.greet)) # <class 'function'>
print(type(Character.greet)) # <class 'function'>

# Instance method can be called by instance directly
# If we want to call instance method using CLASS Object, we need to pass the instance because it expects and instance
sheldon.greet_instance() # <__main__.Character object at 0x7f65af915430> says Hello!!
try:
    Character.greet_instance()
except TypeError as ex:
    print(ex)
Character.greet_instance(sheldon) # <__main__.Character object at 0x7f65af915430> says Hello!!

# Class method can be called using both instance and the CLASS Object.
# Even though we are calling the class method using instance, still the method is bound to class
# Python will automatically pass the CLASS Object to the class method.
sheldon.greet_class() # <class '__main__.Character'> says Hello!!
Character.greet_class() # <class '__main__.Character'> says Hello!!

# Static method can be called using both instance and the CLASS Object
sheldon.greet() # Someone says Hello!!
Character.greet() # Someone says Hello!!