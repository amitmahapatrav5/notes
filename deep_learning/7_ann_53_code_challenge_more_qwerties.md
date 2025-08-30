# Reference

Section: 7 \
Lecture: 53 \
Title: CodeChallenge: more qwerties! \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842148 \
Udemy Reference Link: \
Pre-Requisite:

# Code Challenge more qwerties

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
```

```python
# Prepare Data
N = 100
A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 3, -2 ]

a = torch.stack( ( A[0] + torch.randn(N), A[1] + torch.randn(N) ), dim=1 )
b = torch.stack( ( B[0] + torch.randn(N), B[1] + torch.randn(N) ), dim=1 )
c = torch.stack( ( C[0] + torch.randn(N), C[1] + torch.randn(N) ), dim=1 )

data = torch.vstack((a, b, c))
labels = torch.hstack( ( torch.zeros(N, dtype=torch.int64), torch.ones(N, dtype=torch.int64), torch.zeros(N, dtype=torch.int64)+2 ) )

data.shape, labels.shape
```

    (torch.Size([300, 2]), torch.Size([300]))

```python
# Visualize the data
plt.scatter(data[ torch.where(labels==0) ][:, 0], data[ torch.where(labels==0) ][:, 1], marker='s', color='r', facecolor='w')
plt.scatter(data[ torch.where(labels==1) ][:, 0], data[ torch.where(labels==1) ][:, 1], marker='s', color='g', facecolor='w')
plt.scatter(data[ torch.where(labels==2) ][:, 0], data[ torch.where(labels==2) ][:, 1], marker='s', color='b', facecolor='w')
plt.show()
```

![png](7_ann_53_code_challenge_more_qwerties_files/7_ann_53_code_challenge_more_qwerties_4_0.png)

```python
# Build the model
class ANNMultiClassQwerties(nn.Module):
    def __init__(self):
        super(ANNMultiClassQwerties, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.stack(X)
ANNMultiClassQwerties()
```

    ANNMultiClassQwerties(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=3, bias=True)
        (3): Softmax(dim=1)
      )
    )

```python
# train the model
model = ANNMultiClassQwerties()
epochs = 10000
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = torch.zeros(epochs)
accuracies = torch.zeros(epochs)

for epoch in range(epochs):
    # feed forward
    yHat = model(data)

    # calculate loss
    loss = loss_func(yHat, labels)
    losses[epoch] = loss
    accuracies[epoch] = torch.mean( (yHat.argmax(dim=1) == labels).float() ) * 100

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```python
# plot the model performance

_, axes = plt.subplots(1, 2, figsize=(12, 5))

# epoch vs loss
axes[0].plot(range(epochs), losses.detach(), marker='o', markerfacecolor='w')
axes[0].set_title(f'Epochs vs Loss MIN={losses.min()}')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')

axes[1].plot(range(epochs), accuracies.detach(), marker='o', color='g', markerfacecolor='w')
axes[1].set_title(f'Epochs vs Accuracy MAX={accuracies.max()}')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')

plt.show()
```

![png](7_ann_53_code_challenge_more_qwerties_files/7_ann_53_code_challenge_more_qwerties_7_0.png)

## Question

CrossEntropyLoss computes log-softmax internally. Does that mean that the Softmax() layer in the model needs to be there? Does it hurt or help? If you remove that final layer, what would change and what would be the same in the rest of the notebook?