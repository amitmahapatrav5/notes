# class attribute can be a function object.
# These function object when called using the CLASS Object, behaves the way we expected.
# But when the function objects are called by the instance of the class python creates a method object.
# That method object, like any other object has attributes.
# That method object is bound to the instance using which the function is called. [see the example]
# The method object has a data attribute called __self__ which points to the instance
# The method object has a function attribute called __func__ which points to the function
# When instance.method() is called python creates the method object, binds it with the instance and pass the instance itself to the function which the method object holds. By convention this first argument is called self.
# type(instance.method) is method and this object is not same as function object.
# Summery: functions that are defined in the class are transformed into methods when they're called from instances of the class.


class TBBTCharacter:
    series_name = 'The Big Bang Theory'
    def get_season_count():
        return 12

sheldon = TBBTCharacter()

print(TBBTCharacter.series_name)
print(sheldon.series_name)

print(TBBTCharacter.get_season_count())

try:
    print(sheldon.get_season_count())
except TypeError as ex:
    print(ex)

print(getattr(TBBTCharacter, 'get_season_count')) # function object
print(getattr(sheldon, 'get_season_count')) # method object

# =========================================================================

class TBBTCharacter:
    series_name = 'The Big Bang Theory'
    def get_season_count(obj):
        return 12

sheldon = TBBTCharacter()

print(sheldon.get_season_count.__func__ is TBBTCharacter.get_season_count)
print(sheldon.get_season_count.__name__)
print(sheldon.get_season_count.__self__ is sheldon)

sheldon_get_season_count_method = sheldon.get_season_count

print(sheldon_get_season_count_method())
# or
print(sheldon_get_season_count_method.__func__(sheldon))
# or
print(TBBTCharacter.get_season_count('anything in this case'))
# or
print(sheldon.get_season_count())


# ==========================================================================
# Some weird examples

class TBBTCharacter:
    def __init__(self, field_of_study, favorite_movie):
        self.field_of_study = field_of_study
        self.favorite_movie = favorite_movie

    def work(self):
        return f'Do {self.field_of_study} work.'

sheldon = TBBTCharacter(field_of_study='Science', favorite_movie='Raiders of the Lost Ark')
print(sheldon.work())

# monkey patching
TBBTCharacter.watch_movie = lambda self: f'Play {self.favorite_movie}'
sheldon.attend_quarterly_roommate_meeting = lambda *args :f'Attend Meeting with {args}'

print(type(sheldon.watch_movie), sheldon.watch_movie()) # Method, ..
print(type(sheldon.attend_quarterly_roommate_meeting), sheldon.attend_quarterly_roommate_meeting()) # Function, ..