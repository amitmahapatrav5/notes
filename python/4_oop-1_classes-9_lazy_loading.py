# There is nothing called read-only or computed property in python
# Its just a convention that is followed and useful in scenarios like getting the area of a circle when we already know the radius
# Also this is called lazy loading, because the area is calculated when being called.

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, new_radius):
        if(new_radius<0):
            raise ValueError('Radius cannot be negative')
        self._radius = new_radius
    
    @property
    def area(self):
        import math
        return math.pi * self.radius**2
    
unit_circle = Circle(1)
print(unit_circle.area)

# One thing we can improve here by using a caching mechanism.
# This is not a python related concept, but a better programming practice.
# Because the area is being calculated again and again, even though the radius is not modified.
# So we can cache the result and reset it when the radius is modified.
# Note that in the __init__ itself we can calculate the area but if say someone is creating lots of circle, it is useless to hold the values unless really required.

class Circle:
    def __init__(self, radius):
        self._radius = radius
        self._area = None
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, new_radius):
        if(new_radius<0):
            raise ValueError('Radius cannot be negative')
        self._radius = new_radius
        self._area = None
    
    @property
    def area(self):
        if not self._area:
            import math
            self._area = math.pi * self.radius**2
        return self._area
    
unit_circle = Circle(1)
print(unit_circle.area)
unit_circle.radius = 2
print(unit_circle.area)





# Where this lazy loading can be used
# Here we have used the urllib module of python which comes built-in, but it is recommended to use request module for dealing with http requests

from urllib import request
from time import perf_counter

class Webpage:
    def __init__(self, url):
        self._url = url
        self._page = None
        self._load_time = None
        self._size = None
    
    @property
    def url(self):
        return self._url
    
    @url.setter
    def url(self, new_url):
        self._url = new_url
        self._page = None
        self._load_time = None
        self._size = None

    @property
    def size(self):
        if not self._page:
            self.download_page()
        return self._size
    
    @property
    def page(self):
        if not self._page:
            self.download_page()
        return self._page       
    
    @property
    def load_time(self):
        if not self._page:
            self.download_page()
        return self._load_time
    
    def download_page(self):
        with request.urlopen(self.url) as page:
            start = perf_counter()
            self._page = page.read()
            end = perf_counter()
        self._load_time = end - start
        self._size = len(self._page)

    def __str__(self):
        return f'Page {self.url} Load Time: {self.load_time} Size: {self.size}'

urls = [
    'https://www.google.com',
    'https://www.yahoo.com',
    'https://www.python.org',
]

webpages = [Webpage(url) for url in urls]

for webpage in webpages:
    print(webpage)


# We can define property without property class - Yes!!
# But why we wanna do that
# Sometimes we need to use same property and similar functionality(validation) in many different classes and attributes. In those cases if we use property, we would be repeating a lot of codes.
# This will be explained in Data Descriptor