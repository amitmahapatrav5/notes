# Reference

Section: 6 \
Lecture: 32 \
Title: Gradient descent in 1D \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842084 \
Udemy Reference Link: \
Pre-Requisite:

# Gradient Descent in 1D

### Things to remember

- Gradient Descent basically helps in finding out the approximate x, where y value is minimum.
- It does that by the help of slope.
- To visualize in 2D, when slope is positive, X is decreased and when slope is negetive X is increased.
- It works in any dimension.
- Drawback is, sometimes, it might get stuck in a local mimima. However this is not a big problem, because in higher dimension, the probability of getting stuck in a local mimima is very low.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Functions having single local minima

$f(x)=x^2 - 12x +36$

$dy/dx = 2x - 12$

```python
def fx(x):
    return x**2 - 12*x - 36

def dydx(x):
    return 2*x - 12
```

```python
domain = (0, 12)
X = np.arange(*domain, 0.005)
Y = fx(X)
```

```python
plt.plot(X, Y)
plt.show()
```

![png](6_gradient_descent_32_gradient_descent_in_1d_files/6_gradient_descent_32_gradient_descent_in_1d_8_0.png)

```python
learning_rate = 0.1
epochs = 100
```

```python
localmin = np.random.choice(X, 1)
learning = np.zeros((epochs, 2))
for epoch in range(epochs):
    grad = dydx(localmin)
    learning[epoch][0] = localmin[0]
    learning[epoch][1] = fx(localmin)[0]
    localmin -= learning_rate*grad # ROOT OF GRADIENT DESCENT
localmin
```

    array([6.])

```python
x, y= learning[:,0], learning[:,1]
```

```python
plt.plot(X, Y)
plt.scatter(x, y, c='g', marker='o')
plt.show()
```

![png](6_gradient_descent_32_gradient_descent_in_1d_files/6_gradient_descent_32_gradient_descent_in_1d_12_0.png)

## What happens when we make the learning parameter too large or too small

```python
def gradient_descent(localmin, learning_rate, epochs):
    learning = np.zeros((epochs, 2))
    for epoch in range(epochs):
        grad = dydx(localmin)
        learning[epoch][0] = localmin[0]
        learning[epoch][1] = fx(localmin)[0]
        localmin -= learning_rate*grad # ROOT OF GRADIENT DESCENT
    return localmin, learning

def plot(X, Y, x, y):
    plt.plot(X, Y)
    plt.scatter(x, y, c='g', marker='o')
    plt.show()

def fx(x):
    return x**2 - 12*x - 36

def dydx(x):
    return 2*x - 12
```

```python
X = np.arange(0, 12, 0.005)
Y = fx(X)
plt.plot(X, Y)
plt.plot()
```

    []

![png](6_gradient_descent_32_gradient_descent_in_1d_files/6_gradient_descent_32_gradient_descent_in_1d_15_1.png)

```python
# This will diverge
localmin, learning = gradient_descent(np.array([0.]), learning_rate=1, epochs=100)
plot(X, Y, learning[:, 0], learning[:, 1])
localmin
```

![png](6_gradient_descent_32_gradient_descent_in_1d_files/6_gradient_descent_32_gradient_descent_in_1d_16_0.png)

    array([0.])

```python
# This will converge
localmin, learning = gradient_descent(np.array([0.]), learning_rate=0.1, epochs=100)
plot(X, Y, learning[:, 0], learning[:, 1])
localmin
```

![png](6_gradient_descent_32_gradient_descent_in_1d_files/6_gradient_descent_32_gradient_descent_in_1d_17_0.png)

    array([6.])