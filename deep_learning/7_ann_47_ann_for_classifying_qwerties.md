# Reference

Section: 7 \
Lecture: 47 \
Title: ANN for classifying qwerties \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842130 \
Udemy Reference Link: \
Pre-Requisite:

# ANN for Classification

```python
import torch
from torch import nn

from matplotlib import pyplot as plt
```

```python
# Prepare data
N = 100
A = [ 1, 1 ]
B = [ 5, 1 ]

a = torch.vstack( ( A[0] + torch.randn(N), A[1] + torch.randn(N) ) )
b = torch.vstack( ( B[0] + torch.randn(N), B[1] + torch.randn(N) ) )

data = torch.vstack( ( a.T, b.T ) )
labels = torch.vstack( ( torch.zeros(N, 1), torch.ones(N, 1) ) )

plt.scatter(data [ torch.where(labels==0)[0], 0 ], data [ torch.where(labels==0)[0], 1 ], marker='s', color='b', facecolor='w')
plt.scatter(data [ torch.where(labels==1)[0], 0 ], data [ torch.where(labels==1)[0], 1 ], marker='s', color='g', facecolor='w')
plt.show()
```

![png](7_ann_47_ann_for_classifying_qwerties_files/7_ann_47_ann_for_classifying_qwerties_3_0.png)

```python
# Building Model
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
# Metaparameter Set Up
learning_rate = 0.01

loss_func = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

```python
# Training Model
epochs = 2000
losses = torch.zeros(epochs)

for epoch in range(epochs):

    # forward pass
    yHat = model(data)

    # Calculating Loss
    loss = loss_func(yHat, labels)
    losses[epoch] = loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```python
# Evaluating Model Performance
prediction = model(data)

prediction [ torch.where(prediction > 0.5) ] = 1.
prediction [ torch.where(prediction <= 0.5) ] = 0.

performance = torch.corrcoef ( torch.vstack( ( prediction.T, labels.T ) ) )[0, 1].detach()*100

plt.scatter(data[ torch.where(labels == 1.)[0], 0 ], data[ torch.where(labels == 1.)[0], 1 ], marker='s', color='g', facecolor='w')
plt.scatter(data[ torch.where(labels == 0.)[0], 0 ], data[ torch.where(labels == 0.)[0], 1 ], marker='s', color='b', facecolor='w')
plt.scatter(data[ torch.where(prediction != labels)[0], 0 ], data[ torch.where(prediction != labels)[0], 1 ], marker='x', color='r')
plt.title(f'Performance = {torch.round(performance)} %')
plt.show()
```

![png](7_ann_47_ann_for_classifying_qwerties_files/7_ann_47_ann_for_classifying_qwerties_7_0.png)

### Why ReLU is used in the internal layers but sigmoid in the output layer?

ReLU is commonly used in the internal layers of a neural network because it helps to mitigate the vanishing gradient(**Need to knaow what is this**) problem.
Sigmoid function is used in the output layer because it outputs values between 0 and 1, which is suitable for binary classification tasks.

### Why BCELoss is used instead of MSELoss?

This problem is a BCELoss problem. What does that mean?

### How to interpret the loss function?

If you can see here, with 1500 epochs, the curve has still not yet asymptoted. So the model can still learn. May be here we can increase the number of epochs or vary some other metaparameters to get to that phase.

```python
plt.plot( range(1, epochs+1), losses.detach(), marker='o', markerfacecolor='w' )
plt.show()
```

![png](7_ann_47_ann_for_classifying_qwerties_files/7_ann_47_ann_for_classifying_qwerties_11_0.png)
