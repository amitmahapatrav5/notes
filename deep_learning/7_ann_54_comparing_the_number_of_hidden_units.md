# Reference

Section: 7 \
Lecture: 54 \
Title: Comparing the number of hidden units \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842152 \
Udemy Reference Link: \
Pre-Requisite:

```python
import torch
from torch import nn

import seaborn as sns

import matplotlib.pyplot as plt
```

```python
iris = sns.load_dataset('iris')

data = torch.tensor( iris[ iris.columns[:-1] ].values, dtype=torch.float32 )

labels = torch.zeros(len(iris), dtype=torch.int64)
# labels[ iris['species'] == 'versicolor' ] = 0
labels[ iris['species'] == 'versicolor' ] = 1
labels[ iris['species'] == 'virginica' ] = 2

data.shape, labels.shape
```

    (torch.Size([150, 4]), torch.Size([150]))

```python
# build model
class ANNIrisClassifier(nn.Module):
    def __init__(self, n_units):
        super(ANNIrisClassifier, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, n_units),
            nn.ReLU(),
            nn.Linear(n_units, 3)
        )

    def forward(self, X):
        return self.stack(X)

ANNIrisClassifier(5)
```

    ANNIrisClassifier(
      (stack): Sequential(
        (0): Linear(in_features=4, out_features=5, bias=True)
        (1): ReLU()
        (2): Linear(in_features=5, out_features=3, bias=True)
      )
    )

```python
# train the model
epochs = 150

def train_model(n_units):
    losses, accuracies = torch.zeros(epochs), torch.zeros(epochs)

    model = ANNIrisClassifier(n_units)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # forward pass
        yHat = model(data)

        # evaluate loss
        loss = loss_func(yHat, labels)
        losses[epoch] = loss
        accuracies[epoch] = torch.mean( (yHat.argmax(dim=1) == labels).float() )*100

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses, accuracies
```

```python
# test the code once and plot the result
losses, accuracies = train_model(16)

_, axes = plt.subplots(1, 2, figsize=(12, 5))

# Epoch vs Loss
axes[0].plot(range(epochs), losses.detach(), marker='o', markerfacecolor='w')
axes[0].set_title(f'Epoch vs Loss MIN={torch.min(losses)}')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

# Epoch vs Accuracy
axes[1].plot(range(epochs), accuracies.detach(), marker='o', color='g', markerfacecolor='w')
axes[1].set_title(f'Epoch vs Accuracy MAX={torch.max(accuracies)}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')

plt.show()
```

![png](7_ann_54_comparing_the_number_of_hidden_units_files/7_ann_54_comparing_the_number_of_hidden_units_5_0.png)

```python
# Parametric Experiment
n_unit_range = torch.arange(1, 129)

max_accuracies = torch.zeros(len(n_unit_range))

for n_units in range(len(n_unit_range)):
    _ , accuracies = train_model(n_units)
    max_accuracies[n_units] = accuracies.max()
```

```python
# Epoch vs Accuracy
plt.plot(n_unit_range.detach(), max_accuracies.detach(), marker='o', color='g', markerfacecolor='w')
plt.title(f'Hidden Units vs Accuracy MAX={torch.max(max_accuracies)}')
plt.xlabel('Hidden Units')
plt.ylabel('Max Accuracy')
plt.show()
```

![png](7_ann_54_comparing_the_number_of_hidden_units_files/7_ann_54_comparing_the_number_of_hidden_units_7_0.png)

```python

```