# Things To Know

## NumPy/Torch

### Transposing a matrix

### Dot Product of 1D and 2D Vectors

### 1. stack(), vstack(), hstack() in numpy and torch

#### Transpose works for matrices having more than 1D. This is valid for both numpy and torch

```python
import numpy as np
import torch

n = np.arange(24)
t = torch.arange(24)

# n.shape, n.T.shape # ((24,), (24,))
# t.shape, t.T.shape # (torch.Size([24]), torch.Size([24])) # x.T.shape throws an warning
```

#### numpy axis is same as torch dim. Both starts with 1, not 0

```python
import numpy as np
import torch

x = np.arange(4).reshape((4, 1)).shape # (4, 1) => axis1= 4, axis2= 1
y = torch.arange(4).reshape((4, 1)).shape # torch.Size([4, 1]) => dim1 = 4, dim2= 1

x, y
```

#### When using stack((x, y, z, ..)), all input arrays must have the same shape

```python
import numpy as np

x = np.arange(4).reshape((4,1))
y = np.arange(5).reshape((5,1))
try:
    x0 = np.stack((x, y), axis=0)
except ValueError as ex:
    print(ex)
# x1 = np.stack((x, x), axis=1)
# x2 = np.stack((x, x), axis=2)
```

#### stack() is NOT a general method for hstack()/vstack()

#### stack() method add one more axis/dim to the input arrays/tensors

#### hstack()/vstack() keep the number of axis/dim same as source, **except 1 case**.

#### vstack() works on axis/dim=1, in other words, it increases number of rows

#### hstack() works on axis/dim=2, in other words, it increases number of cols

```python
import numpy as np

x = np.arange(24) # 1D => stacking axis = 0, 1
xx = x.reshape(3, 8) # 2D => stacking axis = 0, 1, 2
xxx = x.reshape(2,3,4) # 3D => stacking axis = 0, 1, 2, 3

# print(x) # (24,)
# print(np.stack((x,x), axis=0)) # (2, 24)
# print(np.stack((x,x), axis=1)) # (24, 2)
# print(np.hstack((x, x))) # (48,)
# print(np.vstack((x, x))) # (2, 24)

# print(xx) # (3, 8)
# print(np.stack((xx, xx), axis=0)) # (2, 3, 8)
# print(np.stack((xx, xx), axis=1)) # (3, 2, 8)
# print(np.stack((xx, xx), axis=2)) # (3, 8, 2)
# print(np.hstack((xx, xx))) # (3, 16)
# print(np.vstack((xx, xx))) # (6, 8)

# print(xxx) # (2, 3, 4)
# print(np.stack((xxx, xxx), axis=0)) # (2, 2, 3, 4)
# print(np.stack((xxx, xxx), axis=1)) # (2, 2, 3, 4)
# print(np.stack((xxx, xxx), axis=2)) # (2, 3, 2, 4)
# print(np.stack((xxx, xxx), axis=3)) # (2, 3, 4, 2)
# print(np.hstack((xxx, xxx))) # (2, 6, 4)
# print(np.vstack((xxx, xxx))) # (4, 3, 4)


# import torch

# x = torch.arange(24) # 1D => stacking dim = 0, 1
# xx = x.reshape(3, 8) # 2D => stacking dim = 0, 1, 2
# xxx = x.reshape(2,3,4) # 3D => stacking dim = 0, 1, 2, 3

# print(x) # torch.Size([24])
# print(torch.stack((x,x), dim=0)) # torch.Size([2, 24])
# print(torch.stack((x,x), dim=1)) # torch.Size([24, 2])
# print(torch.hstack((x, x))) # torch.Size([48])
# print(torch.vstack((x, x))) # torch.Size([2, 24])

# print(xx) # torch.Size([3, 8])
# print(torch.stack((xx, xx), dim=0)) # torch.Size([2, 3, 8])
# print(torch.stack((xx, xx), dim=1)) # torch.Size([3, 2, 8])
# print(torch.stack((xx, xx), dim=2)) # torch.Size([3, 8, 2])
# print(torch.hstack((xx, xx))) # torch.Size([3, 16])
# print(torch.vstack((xx, xx))) # torch.Size([6, 8])

# print(xxx) # torch.Size([2, 3, 4])
# print(torch.stack((xxx, xxx), dim=0)) # torch.Size([2, 2, 3, 4])
# print(torch.stack((xxx, xxx), dim=1)) # torch.Size([2, 2, 3, 4])
# print(torch.stack((xxx, xxx), dim=2)) # torch.Size([2, 3, 2, 4])
# print(torch.stack((xxx, xxx), dim=3)) # torch.Size([2, 3, 4, 2])
# print(torch.hstack((xxx, xxx))) # torch.Size([2, 6, 4])
# print(torch.vstack((xxx, xxx))) # torch.Size([4, 3, 4])
```

#### hstack()/vstack() keep the number of axis/dim same as source, **except 1 case, where the sourse axis/dim is 1 and we are performing vstack**

```python
# numpy example
import numpy as np

x = np.arange(12)
y = np.arange(11, -1, -1)
print(x.shape, y.shape)

print(np.stack((x, y), axis=0).shape)
print(np.vstack((x, y)).shape)


# torch example
import torch

x = torch.arange(12)
y = torch.arange(11, -1, -1)
print(x.shape, y.shape)

print(torch.stack((x, y), dim=0).shape)
print(torch.vstack((x, y)).shape)
```

### 2. squeeze() and unsqueeze() operation in numpy and torch

### 3. mean() in numpy/torch along various axis

### 4. where() in numpy/torch

### 5. round() in numpy/torch

### 6. How to create the distributions like below and understand what exactly normal distribution means

```python
import torch
import matplotlib.pyplot as plt

nPerClust = 100
blur = 1

A = [1, 1]
B = [6, 6]

# Generate data
a = torch.stack((A[0] + torch.randn(nPerClust) * blur, A[1] + torch.randn(nPerClust) * blur))
b = torch.stack((B[0] + torch.randn(nPerClust) * blur, B[1] + torch.randn(nPerClust) * blur))

# True labels
labels = torch.cat((torch.zeros(nPerClust, 1), torch.ones(nPerClust, 1)))

# Concatenate into a matrix
data = torch.cat((a, b), dim=1).T

# Show the data
fig = plt.figure(figsize=(5, 5))
plt.plot(data[labels.squeeze() == 0, 0], data[labels.squeeze() == 0, 1], 'bs')
plt.plot(data[labels.squeeze() == 1, 0], data[labels.squeeze() == 1, 1], 'ko')
plt.show()

```

### 7. How model performance is calculated using co-relation coefficient

### 8. What is np.meshgrid and if the same is present in torch

```python
import torch

x = torch.arange(1, 11, dtype=torch.float32).reshape(5, 2)
x, torch.mean(x, dim=1)
```

## Matplotlib

### plt.subplots(row, col, figsize)

### axis.plot(x, y, color, marker, linestyle, markerfacecolor, markersize, label)

### axis.scatter(x, y, color, marker, label)

```python
from matplotlib import pyplot as plt

N = 30
x1 = torch.randn(N)
y1 = x + torch.randn(N) + torch.randn(N)/2

fig, axis = plt.subplots(1, 1)
axis.scatter(x, y, marker='s', color='r', facecolor='w')

plt.show()
```

### How to create a 3D plot

### How to create heatmap 3D plot

## Pandas

## Seaborn

### sns.load_dataset()

### sns.pairplot()

## Experiment

```python
import torch
from torch import nn

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()

# # Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()
```

```python
torch.empty(4)
```

```python
bcell = nn.BCEWithLogitsLoss()
cel = nn.CrossEntropyLoss()
bcell, cel
```

---