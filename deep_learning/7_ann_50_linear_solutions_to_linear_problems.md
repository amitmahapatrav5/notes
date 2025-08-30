# Reference

Section: 7 \
Lecture: 50 \
Title: Linear solutions to linear problems \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842140 \
Udemy Reference Link: \
Pre-Requisite:

# Linear Solution to Linear Problems

## Demystifying the uncertainity to Qwerty's problem

The problem we faced earlier was with one perceptron or multilayer perceptron, for the exact same metaparameters, the model was either performing poorly or very good. Why is this the case?

It turns out that, Qwerty is a very simple binary classification problem. However to solve that, we are using non-linear model. So the model is trying to find a non-linear solution, whearase a simple linear line can solve the problem. So we can just remove the non-linearity(nn.ReLU activation fuctions) from the model and should give us consistent result.

```python
import torch
from torch import nn
from matplotlib import pyplot as plt
```

```python
# Prepare Data
A = [ 1, 1 ]
B = [ 1, 5 ]
N = 100

a = torch.stack((A[0]+torch.randn(N), A[1]+torch.randn(N)), dim=1)
b = torch.stack((B[0]+torch.randn(N), B[1]+torch.randn(N)), dim=1)

data = torch.vstack((a, b))
labels = torch.vstack((torch.zeros(N, 1), torch.ones(N, 1)))
data.shape, labels.shape
```

    (torch.Size([200, 2]), torch.Size([200, 1]))

```python
# Visualize data
plt.scatter(data [torch.where(labels==0)[0], 0], data [torch.where(labels==0)[0], 1], marker='s', color='b', facecolor='w')
plt.scatter(data [torch.where(labels==1)[0], 0], data [torch.where(labels==1)[0], 1], marker='s', color='g', facecolor='w')
plt.show()
```

![png](7_ann_50_linear_solutions_to_linear_problems_files/7_ann_50_linear_solutions_to_linear_problems_5_0.png)

```python
# Build the model
class ANNMultiLayerBinaryClassifier(nn.Module):
    def __init__(self):
        super(ANNMultiLayerBinaryClassifier, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 16),
            # nn.ReLU(), # removed non-linearity
            nn.Linear(16, 1),
            # nn.ReLU(), # removed non-linearity
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    def forward(self, X):
        return self.stack(X)

ANNMultiLayerBinaryClassifier()
```

    ANNMultiLayerBinaryClassifier(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=1, bias=True)
        (2): Linear(in_features=1, out_features=1, bias=True)
        (3): Sigmoid()
      )
    )

```python
# Train the model
epochs = 1000

def train(lr):
    model = ANNMultiLayerBinaryClassifier()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = torch.zeros(epochs)
    for epoch in range(epochs):
        # forward
        yHat = model(data)

        # compute loss
        loss = loss_func(yHat, labels)
        losses[epoch] = loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prediction = model(data)
    accuracy = torch.mean(((prediction>0.5)==labels).float()) * 100

    return accuracy, losses
```

```python
# Test the code ones
accuracy, losses = train(lr=0.01)

plt.plot(range(epochs), losses.detach(), marker='o')
plt.title(f'Epoch vs Loss, accuracy={accuracy}')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

![png](7_ann_50_linear_solutions_to_linear_problems_files/7_ann_50_linear_solutions_to_linear_problems_8_0.png)

## Experiment Learning Rate vs Accuracy and Epochs vs Losses

```python
lrs = torch.linspace(0.001, 0.1, 40)

allLosses = torch.zeros((len(lrs), epochs))
accuracies = torch.zeros(len(lrs))

for i in range(len(lrs)):
    accuracy, losses = train(lrs[i])

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
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].set_title('Losses by Learning Rate')

plt.show()
```

![png](7_ann_50_linear_solutions_to_linear_problems_files/7_ann_50_linear_solutions_to_linear_problems_11_0.png)

```python

```