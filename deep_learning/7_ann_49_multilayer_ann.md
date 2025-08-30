# Reference

Section: 7 \
Lecture: 49 \
Title: Multilayer ANN \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842138 \
Udemy Reference Link: \
Pre-Requisite:

# Multilayer ANN

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

![png](7_ann_49_multilayer_ann_files/7_ann_49_multilayer_ann_4_0.png)

```python
# Build the model
class ANNMultiLayerBinaryClassifier(nn.Module):
    def __init__(self):
        super(ANNMultiLayerBinaryClassifier, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
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
        (1): ReLU()
        (2): Linear(in_features=16, out_features=1, bias=True)
        (3): ReLU()
        (4): Linear(in_features=1, out_features=1, bias=True)
        (5): Sigmoid()
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

![png](7_ann_49_multilayer_ann_files/7_ann_49_multilayer_ann_7_0.png)

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
axes[1].set_xlabel('Loss')
axes[1].set_ylabel('Epochs')
axes[1].set_title('Losses by Learning Rate')

plt.show()
```

![png](7_ann_49_multilayer_ann_files/7_ann_49_multilayer_ann_10_0.png)

### Why it is woking very good for certain lr and very poor for others?

I have observed that, for the same learning rate also, model sometimes performs good and sometimes performs bad. However, if I am increasing the number of epochs(1000, then 2000, then 5000), the frequency of getting bad accuracy is reduced. And the experiment is showing that in verity of learning rates. So, if we perform the experiment, once again, it is likely that the learning rate for which the accuracy was bad on nth experiment, may show good accuracy in (n+1)th attempt.