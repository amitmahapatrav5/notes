# When we initialize a class python does 2 things.
# 1. Create an instance of the class.
# 2. Initialize the namespace of the newly created instance.
# We can override the initialization behavior[step 2] by adding __init__()
# step 1 can be overridden by defining __new__ function in the class. BUT do this only and only you understand WHY this must be done.