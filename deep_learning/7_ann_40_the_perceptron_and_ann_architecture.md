# Reference

Section: 7 \
Lecture: 40 \
Title: The perceptron and ANN architecture \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842114 \
Udemy Reference Link: \
Pre-Requisite:

# Linear and Non Linear ANN Model

**Linear models only solve linearly separable problems** \
**Model is considered Linear, if the operation it is performing is only addition or multiplication** \
**If the Model is performing anything else, then it is Non-Linear Model** \
**I think the only component that can be non Linear is activation function**

**We should not use Linear Models for Non Linear Problems** \
**Similarly, we should not use Non Linear Models for Linear Problems**

```python
import torch
from torch import nn
```

```python
# Example of Linear Model
class  LinearANN(nn.Module):
    def __init__(self):
        super(LinearANN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(3, 1)
        )
LinearANN()
```

    LinearANN(
      (stack): Sequential(
        (0): Linear(in_features=3, out_features=1, bias=True)
      )
    )

```python
# Example of Non-Linear Model
class NonLinearANN(nn.Module):
    def __init__(self):
        super(NonLinearANN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid() # This is the non Linear Component
        )
NonLinearANN()
```

    NonLinearANN(
      (stack): Sequential(
        (0): Linear(in_features=3, out_features=1, bias=True)
        (1): Sigmoid()
      )
    )

**Linearly Separable problem: Where the separation can be performed by a line or plane or hyperplane**

```python
# torch.randn()
```

**Majority math that happens in any ANN is $\sigma(X^{T}W)$**

**Why do we need bias?**

Equations like $aX + bY + ... + cZ = 0$ always passes through origin. \
But equations like $aX + bY + ... + cZ + bias = 0$ does not, and hence the advantage as shown in below figure. \
However, it is possible to mean center all the points so that we will always have the separating hyper-plane pass through origin. But this is not something we always go for. I am not exactly clear on the scenario of non-linear equation

```python
import matplotlib.pyplot as plt

nPerClust = 100
blur = 1

# First diagram coordinates
A1 = [3, 8]
B1 = [8, 3]

# Second diagram coordinates (existing)
A2 = [1, 1]
B2 = [6, 6]

# Generate data for the first diagram
a1 = torch.stack((A1[0] + torch.randn(nPerClust) * blur, A1[1] + torch.randn(nPerClust) * blur))
b1 = torch.stack((B1[0] + torch.randn(nPerClust) * blur, B1[1] + torch.randn(nPerClust) * blur))

# Generate data for the second diagram
a2 = torch.stack((A2[0] + torch.randn(nPerClust) * blur, A2[1] + torch.randn(nPerClust) * blur))
b2 = torch.stack((B2[0] + torch.randn(nPerClust) * blur, B2[1] + torch.randn(nPerClust) * blur))

# True labels for both diagrams
labels1 = torch.cat((torch.zeros(nPerClust, 1), torch.ones(nPerClust, 1)))
labels2 = torch.cat((torch.zeros(nPerClust, 1), torch.ones(nPerClust, 1)))

# Concatenate into matrices
data1 = torch.cat((a1, b1), dim=1).T
data2 = torch.cat((a2, b2), dim=1).T

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# No Need of Bias for this
axs[0].plot(data1[labels1.squeeze() == 0, 0], data1[labels1.squeeze() == 0, 1], 'bs', label='Cluster A')
axs[0].plot(data1[labels1.squeeze() == 1, 0], data1[labels1.squeeze() == 1, 1], 'ko', label='Cluster B')
axs[0].set_title('Separating Line Passes through Origin')
axs[0].legend()

# Need Bias for this
axs[1].plot(data2[labels2.squeeze() == 0, 0], data2[labels2.squeeze() == 0, 1], 'bs', label='Cluster A')
axs[1].plot(data2[labels2.squeeze() == 1, 0], data2[labels2.squeeze() == 1, 1], 'ko', label='Cluster B')
axs[1].set_title('Separating Line cannot pass through Origin')
axs[1].legend()

# Show the plots
plt.tight_layout()
plt.show()
```

![png](7_ann_40_the_perceptron_and_ann_architecture_files/7_ann_40_the_perceptron_and_ann_architecture_10_0.png)

```python

```