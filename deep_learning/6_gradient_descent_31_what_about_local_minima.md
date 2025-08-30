# Reference

Section: 6 \
Lecture: 31 \
Title: What about local minima? \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842082 \
Udemy Reference Link: \
Pre-Requisite:

# What About Local Minima

## Functions having multiple local minimas

Even though, the model can get stuck in a local minimum, still it does not seem to be a big problem. WHY ??

The incredible success of deep learning, in spite of the "problems" with gradient descent, remains a mystery.

It is possible that there are many good solutions(many equally good local minima). This interpretation is consistent with the huge diversity of weight configurations that produce similar model performance. That means, models with very different configuration perform very similar in accuracy of the same problem. So that gives a hint that there are many good solutions or good local minima.

Another possibility is that there are extremely few local minima in high-dimensional space. This interpretation is consistent with the complexity and absurd dimensionality of DL models.

Also in higher dimensions, few points are there where that point is a minima for one dimension, but not a minima for the other dimension. These points are called saddle points. So Gradient descent will not get stuck here. So say, we have a 4000 dimension space, gradient descent will get stuck at a local minimum which will be a local minimum of all 4000 dimension. Then only there is a problem. But the probability of existence of such local minima in such high dimensional space is very low. And usually our deep learning models work on very high dimensional space.

**What to do about it?**

When model performance is good, don't worry about local minima.

One possible solution: Re-train the model many times using different random weights (different starting locations on the loss landscape) and pick the model that does best.

Another possible solution: Increase the dimensionality (complexity) of the model to have fewer local minima.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

$f(x) = sin(x) + 0.5sin(3x)$

$dy/dx = cos(x) + 1.5cos(3x)$

```python
def fx(x):
    return np.sin(x) + 0.5*np.sin(3*x)

def dydx(x):
    return np.cos(x) + 1.5*np.cos(3*x)
```

```python
X = np.linspace(-5, 5, 100)
Y = fx(X)
```

```python
plt.plot(X, Y)
plt.show()
```

![png](6_gradient_descent_31_what_about_local_minima_files/6_gradient_descent_31_what_about_local_minima_8_0.png)

```python
def gradient_descent(localmin, learning_rate, epochs):
    learning = np.zeros((epochs, 2))
    for epoch in range(epochs):
        grad = dydx(localmin)
        learning[epoch][0] = localmin[0]
        learning[epoch][1] = fx(localmin)[0]
        localmin -= learning_rate*grad # ROOT OF GRADIENT DESCENT
    return localmin, learning

def plot(x, y):
    plt.plot(X, Y)
    plt.scatter(x, y, c='g', marker='o')
    plt.show()
```

```python
localmin, learning = gradient_descent(np.array([1.]), 0.1, 100)
plot(learning[:,0], learning[:,1])
localmin
```

![png](6_gradient_descent_31_what_about_local_minima_files/6_gradient_descent_31_what_about_local_minima_10_0.png)

    array([1.57079633])

```python
localmin, learning = gradient_descent(np.array([0.5]), 0.1, 100)
plot(learning[:,0], learning[:,1])
localmin
```

![png](6_gradient_descent_31_what_about_local_minima_files/6_gradient_descent_31_what_about_local_minima_11_0.png)

    array([-0.70167412])

```python

```