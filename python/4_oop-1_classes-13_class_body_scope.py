# Class Body has it's own namespace
# But the functions which are defined in the class body are not nested within the class body namespace.
# So below 2 are exactly same
class Character:
    def greet(self):
        print(f'{self} says Hello!!')

sheldon = Character()
sheldon.greet()

def say_hello(arg):
    print(f'{arg} says Hello!!')
class Character:
    greet = say_hello

sheldon = Character()
sheldon.greet()

# When we are writing like self.something or cls.something or ClassName.something, we are explicitly telling python in which namespace must it look for.
series = 'The Office'
class Character:
    series = 'The Big Bang Theory'
    
    def works_in(self):
        return self.series
    
    @classmethod
    def get_series(cls):
        return cls.series
    
    @staticmethod
    def find_series():
        return Character.series
    
    def fetch_series(args=None):
        return series
    
sheldon = Character()
print(sheldon.works_in()) # It searched in instance namespace, did not find, then searched in class namespace
print(Character.get_series()) # It searched in class namespace
print(sheldon.find_series(), Character.find_series()) # It searched in class namespace
print(sheldon.fetch_series(), Character.fetch_series()) # It searched the symbol series in the module namespace


# Example
name = 'Sheldon'
age = 30

def fun():
    name = 'Leonard'
    
    class Character:
        name = 'Howard'
        
        @staticmethod
        def get_values():
            return name, age
    
    return Character

cls = fun()
name, age = cls.get_values()
print(name, age)

name = 'HIMYM'
class Character:
    name = 'TBBT'
    words_1 = [name] * 3
    words_2 = [name for _ in range(3)]
    

char = Character()
print(char.name)
print(char.words_1) # ['TBBT', 'TBBT', 'TBBT']
print(char.words_2) # ['HIMYM', 'HIMYM', 'HIMYM'] Because List Comprehension is internally a function.