# Reference

Section: 6 \
Lecture: 33 \
Title: CodeChallenge: unfortunate starting value \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842090 \
Udemy Reference Link: \
Pre-Requisite:

# Code Challenge Unfortunate Starting Value

## Vanishing Gradiant Problem

When gradients are small, the updates to the weights during training become negligible, causing the learning process to slow down significantly or even stall.

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
def fn(x):
    return np.cos(2*np.pi*x) + x**2

def grad(x):
    return -2*np.pi*np.sin(2*np.pi*x) + 2*x
```

```python
x = np.linspace(-2, 2, 2001)
plt.plot(x, fn(x), label='f(x)')
plt.plot(x, grad(x), label='dfdx')
plt.legend()
plt.show()
```

![png](6_gradient_descent_33_code_challenge_unfortunate_starting_value_files/6_gradient_descent_33_code_challenge_unfortunate_starting_value_5_0.png)

```python
lr=0.01
epochs=100
localmin = np.random.choice(x)
print(localmin)
for epoch in range(epochs):
    localmin -= lr*grad(localmin)
localmin # -1.42, -0.47, 0.47, 1.42 => These are possible values
```

    -1.76





    np.float64(-1.4250674148261986)

```python
x = np.linspace(-2, 2, 2001)
plt.plot(x, fn(x), label='f(x)')
plt.plot(x, grad(x), label='dfdx')
plt.plot(localmin, fn(localmin), 'ro', label='Local Min')
plt.legend()
plt.show()
```

![png](6_gradient_descent_33_code_challenge_unfortunate_starting_value_files/6_gradient_descent_33_code_challenge_unfortunate_starting_value_7_0.png)

### Unfortunate Starting Point - Vanishing Gradiant Problem

```python
lr=0.01
epochs=100
localmin = 0
localmins = np.zeros(epochs)
print(localmin)
for epoch in range(epochs):
    localmins[epoch] = localmin
    localmin -= lr*grad(localmin)

localmin # 0 => It does not matter howmany epochs we use, still the model is not learning
```

    0





    np.float64(0.0)

```python
x = np.linspace(-2, 2, 2001)
plt.plot(x, fn(x), label='f(x)')
plt.plot(x, grad(x), label='dfdx')
plt.plot(localmin, fn(localmin), 'ro', label='Local Min')
plt.legend()
plt.show()
```

![png](6_gradient_descent_33_code_challenge_unfortunate_starting_value_files/6_gradient_descent_33_code_challenge_unfortunate_starting_value_10_0.png)

```python

```