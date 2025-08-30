

# Reference

Section: 6 \
Lecture: 34 \
Title: Gradient descent in 2D \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842092 \
Lecture: 35 \
Title: CodeChallenge: 2D gradient ascent
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842094
Udemy Reference Link: \
Pre-Requisite:`

# Gradient descent in 2D

$$
f(x, y) = 3(1 - x)^2e^{-x^2-(y+1)^2} - 10\left(\frac{x}{5}-x^3-y^5\right)e^{-x^2-y^2} - \frac{1}{3}e^{-(x+1)^2-y^2}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y)
def f(x, y):
    term1 = 3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
    term2 = 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
    term3 = (1/3) * np.exp(-(x + 1)**2 - y**2)
    return term1 - term2 - term3

# Define the range for x and y
x_min, x_max = -3, 3
y_min, y_max = -3, 3

# Create a grid of x and y values
x = np.linspace(x_min, x_max, 500)
y = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(x, y)

# Calculate the Z values (function values) for each point in the grid
Z = f(X, Y)

# Create the heatmap
plt.figure(figsize=(5, 4))
plt.imshow(Z, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
plt.colorbar(label='Function Value f(x, y)')
plt.title('Heatmap of f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

```

![png](6_gradient_descent_34_gradient_descent_in_2d_files/6_gradient_descent_34_gradient_descent_in_2d_3_0.png)

**In a 2D space (such as the XY plane), we have one variable (let's say X). Therefore, when we calculate the derivative, we obtain a single value.** \
**In a 3D space (XYZ) or higher dimensions, we deal with multiple variables (for example, X and Y). Consequently, when we compute the derivative, we perform a partial derivative with respect to X and another partial derivative with respect to Y. In even higher dimensions, we would compute as many partial derivatives as there are variables.**

### Steps for Gradient Descent(Exactly similar step as in 1D, Only difference is we take partial derivatives explicitly):

**STEP 1:** Evaluate the function at the given points => fx(-3, -3) to fx(3, 3)  
**STEP 2:** Calculate the partial derivatives => dz/dx(-3, -3) to dz/dx(3, 3) and dz/dy(-3, -3) to dz/dy(3, 3)  
**STEP 3:** Set an initial random value => localmin(x, y)  
**STEP 4:** Define constants => epoch, learning rate (lr)  
**STEP 5:** Update the coordinates of the local minimum => localmin*x = localmin_x - dz/dx * lr, localmin*y = localmin_y - dz/dy * lr

## Thinking Challenge: Modify the Steps in sucj a way that we get the global maximum(Gradiant Ascent)

### Way 1 => Change in STEP 5

**STEP 1:** Evaluate the function at the given points => fx(-3, -3) to fx(3, 3)  
**STEP 2:** Calculate the partial derivatives => dz/dx(-3, -3) to dz/dx(3, 3) and dz/dy(-3, -3) to dz/dy(3, 3)  
**STEP 3:** Set an initial random value => localmin(x, y)  
**STEP 4:** Define constants => epoch, learning rate (lr)  
**STEP 5:** Update the coordinates of the local minimum => localmin*x = localmin_x **+** dz/dx * lr, localmin*y = localmin_y **+** dz/dy * lr

### WAY 2 => Change in STEP 2

**STEP 1:** Evaluate the function at the given points => fx(-3, -3) to fx(3, 3)  
**STEP 2:** Calculate the partial derivatives => **-1 X** dz/dx(-3, -3) to **-1 X** dz/dx(3, 3) and **-1 X** dz/dy(-3, -3) to **-1 X** dz/dy(3, 3)  
**STEP 3:** Set an initial random value => localmin(x, y)  
**STEP 4:** Define constants => epoch, learning rate (lr)  
**STEP 5:** Update the coordinates of the local minimum => localmin*x = localmin_x - dz/dx * lr, localmin*y = localmin_y - dz/dy * lr

```python

```
    