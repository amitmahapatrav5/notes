# Reference

Section: 5 \
Lecture: 20 \
Title: Logarithms \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841938 \
Udemy Reference Link: \
Pre-Requisite:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
```

### Exponent and Logarithm cancel eachother

```python
X = np.random.random(size=100)
y1 = np.exp(X)
y2 = np.log(np.exp(X))

_, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X, y1, marker='o', color='gray', facecolor='w')
axes[0].set_title('$e^x$')

axes[1].scatter(X, y2, marker='o', color='green', facecolor='w')
axes[1].set_title('$log_e(e^x) = x$')
plt.show()
```

### Logarithm is a monotonus function

That means, when X foes up, the value of log(X) also goes up \
Log with different bases have the same properties \
But natural logarithm is mostly used because of it's relation with $e$

```python
X = np.arange(1, 100)
ye = np.log(X)
y2 = np.log2(X)
y10 = np.log10(X)

plt.scatter(X, ye, marker='o', color='gray', facecolor='w', label='$log_e$')
plt.scatter(X, y2, marker='o', color='red', facecolor='w', label='$log_2$')
plt.scatter(X, y10, marker='o', color='green', facecolor='w', label='$log_{10}$')
plt.xlabel('X')
plt.ylabel('$log(X)$')
plt.legend()
plt.show()
```