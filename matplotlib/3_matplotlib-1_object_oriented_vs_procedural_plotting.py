# Object Oriented Plotting
# ========================
import matplotlib.pyplot as plt

x = range(0,10_000)
y = [ n**2 for n in x ]

fig, axes = plt.subplots() # Difference 1, 2
axes.plot(x, y)
# We are calling subplot()/subplots() method here to get fig and axes object
# In procedural, we just the plot() function and add the data points there directly


axes.set_title('Quadratic Equation Object Oriented') # Difference 3
# We will be adding all the properties to the axes object
# In case of procedural, we will be passing the value to .title() method

# fig.show()
# Figure will show and vanish in blink
# But this is proper object oriented approach

plt.show() 
# If you want the image to stay until you close
# Usually this is used when you are running the code in VS Code like editor
# In Jupyter Notebook, this is not required


# Procedural Plotting
# ===================
import matplotlib.pyplot as plt

x = range(0,10_000)
y = [ n**2 for n in x ]

plt.plot(x, y)
plt.title('Quadratic Equation Procedural')

plt.show()