# Reference

Section: 7 \
Lecture: 48 \
Title: Learning rates comparison \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842132 \
Udemy Reference Link: \
Pre-Requisite:

# Learning Rate Comparisons

## Why Setting the correct learning rate is important?

If we select a very large learning rate, then we may fall in the issue of divergence or just keep bouncing between few points and never reach the minima.

However if we select a very small learning rate, the model might take very long time to learn.

```python
import torch
from matplotlib import pyplot as plt
```

```python
x = torch.linspace(-1, 1, 20)
y = x**2

plt.plot(x, y)
plt.show()
```

![png](7_ann_48_learning_rate_comparisons_files/7_ann_48_learning_rate_comparisons_4_0.png)

```python
fx = lambda x : x**2
grad = lambda x: 2*x
epochs = 50

def learn(lr):
    localmins = torch.zeros(epochs)

    localmin = torch.tensor(0.75) # initial min
    for epoch in range(epochs):
        localmins[epoch] = localmin
        localmin = localmin - lr*grad(localmin)
    return localmins

diverge_localmins = learn(1)
converge_localmins = learn(0.01)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].set_title('High Learning Rate - Keep bouncing between 2 points')
axes[0].plot(x, y)
axes[0].scatter(diverge_localmins, fx(diverge_localmins), marker='x', color='r')

axes[1].set_title('Lower Learning Rate - Slow Learning')
axes[1].plot(x, y)
axes[1].scatter(converge_localmins, fx(converge_localmins), marker='x', color='g')

plt.show()
```

![png](7_ann_48_learning_rate_comparisons_files/7_ann_48_learning_rate_comparisons_5_0.png)

## Parametric Experiment - Learning Rate

```python
import torch
from torch import nn

from matplotlib import pyplot as plt
```

```python
A = [ 1, 1 ]
B = [ 5, 1 ]
N = 100

a = torch.stack( (A[0] + torch.randn(N), A[1] + torch.randn(N)), dim=1 )
b = torch.stack( (B[0] + torch.randn(N), B[1] + torch.randn(N)), dim=1 )

data = torch.vstack((a, b))
labels = torch.vstack( (torch.zeros(N, 1), torch.ones(N, 1)) )

data.shape, labels.shape

plt.scatter(data[ torch.where(labels == 0)[0], 0], data[ torch.where(labels == 0)[0], 1], marker='s', color='b', facecolor='w')
plt.scatter(data[ torch.where(labels == 1)[0], 0], data[ torch.where(labels == 1)[0], 1], marker='s', color='g', facecolor='w')
plt.show()
```

![png](7_ann_48_learning_rate_comparisons_files/7_ann_48_learning_rate_comparisons_8_0.png)

```python
class ANNClassify(nn.Module):
    def __init__(self):
        super(ANNClassify, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)

model = ANNClassify()
model
```

    ANNClassify(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=1, bias=True)
        (1): ReLU()
        (2): Linear(in_features=1, out_features=1, bias=True)
        (3): Sigmoid()
      )
    )

```python
epochs = 1000

def train(lr):
    model = ANNClassify()

    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = torch.zeros(epochs)

    for epoch in range(epochs):
        yHat = model(data)

        loss = loss_func(yHat, labels)
        losses[epoch] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prediction = model(data)
    accuracy = torch.mean(((prediction>0.5) == labels.bool()).float())*100

    return losses, accuracy
```

## Experiment Learning Rate vs Accuracy and Epochs vs Losses

```python
lrs = torch.linspace(0.001, 0.1, 40)

allLosses = torch.zeros((len(lrs), epochs))
accuracies = torch.zeros(len(lrs))

for i in range(len(lrs)):
    losses, accuracy = train(lrs[i])

    allLosses[i, :] = losses
    accuracies[i] = accuracy
```

```python
# Plot the Experiment Findings
_, axes = plt.subplots(1, 2, figsize=(12, 5))

# Learning Rate vs Accuracy
axes[0].plot(lrs.detach(), accuracies.detach(), marker='s', linestyle='-', markerfacecolor='w')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Accuracy')
axes[0].set_title(f'Accuracy(MAX={accuracies.max()}%, MIN={accuracies.min()}%) by Learning Rate')

# Epochs vs Losses
for i, lr in enumerate(lrs):
    axes[1].plot(range(len(allLosses[i])), allLosses[i].detach(), linestyle='-')
axes[1].set_xlabel('Loss')
axes[1].set_ylabel('Epochs')
axes[1].set_title('Losses by Learning Rate')

plt.show()
```

![png](7_ann_48_learning_rate_comparisons_files/7_ann_48_learning_rate_comparisons_13_0.png)

### Why model is either behaving very good or very poor?

It's really very difficult to say. There could be various possibility like

1. May be the model is not complex enough
2. May be the problem statement is a linear one and we used a non-linear model etc

Also it is very difficult to visualize whether the model is getting stuck in a local minima in case of poor performance. Because we are trying to minimize the loss function in 6(3 weights + 2 biases) + 1 (output feature) dimensions and it si not possible to visualize.
