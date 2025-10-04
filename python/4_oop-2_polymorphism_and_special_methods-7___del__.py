# Python Garbage Collector destroys an object when the object is not referenced anywhere
# So when an object gets destroyed, that is not in our control.
# One way to hook into that is using __del__()
# Before the object destruction, python runs the __del__(), if defined in the class.
# If any exception gets generated that is silenced and python destroys the object.
# So it is not a good idea to perform any clean up, like db commit in this method. Preferred way is to use context managers.
# The exception description is sent to stderr though.
# __del__() sometimes called class finalizer.
# When we say `del obj`, the __del__() method is not called at that time.
# Because `del obj` basically deletes the binding between the reference and the actual object but not deletes the object unless that is the only reference to the object.

# How can we know how many references are there for an object
import ctypes

def ref_count(address):
    return ctypes.c_long.from_address(address).value


nums = [1,2,4]
print(ref_count(id(nums))) # 1
nums_copy = nums
print(ref_count(id(nums))) # 2


class Character:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'Character(name={self.name})'

    def __del__(self):
        print(f'{self} is deleted.')

sheldon = Character('Sheldon Lee Cooper')
cooper = sheldon
address = id(sheldon)
print(ref_count(address)) # 2

del sheldon
print(ref_count(address)) # 1
del cooper # Person(name=Sheldon Lee Cooper) is deleted.
print(ref_count(address)) # 0


class Character:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'Character(name={self.name})'

    def __del__(self):
        raise ValueError('Something Went Wrong.')

sheldon = Character('Sheldon Lee Cooper')
del sheldon # Exception will be ignored and will be printed to the stdout(in my case terminal)
print('Post Deletion') # This will surely be printed

# How to change the stdout to a file from terminal
# I will create a class to make a custom context manager for practice

class FileErrorWriter:
    def __init__(self, file_name):
        self._file_name = file_name

    def __enter__(self):
        import sys
        self.file_obj = open(self._file_name, 'w')
        sys.stderr = self.file_obj # sys.stderr assigned to the file
        return self.file_obj
    
    def __exit__(self, ex_type, ex_value, traceback):
        if self.file_obj:
            self.file_obj.close()

with FileErrorWriter('file.txt') as file:
    sheldon = Character('Sheldon Lee Cooper')
    del sheldon # Exception will be ignored and will be printed to file
    print('Post Deletion')