# Reference

Section: 7 \
Lecture: 52 \
Title: Multi-output ANN (iris dataset) \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842144 \
Udemy Reference Link: \
Pre-Requisite:

# Multi-output ANN (iris dataset)

## Why the Sigmoid Function is Inappropriate for Multiclass Classifier Models

In the case of a binary classification model, we use the sigmoid function because its output range is between 0 and 1. If the output value is less than 0.5, it is classified as Type 1; otherwise, it is classified as Type 2. However, for a multiclass classifier, the sigmoid function is not suitable. We need an activation function that gives the probabilities for each class, so that all the probabilities add up to 1. This is precisely what the softmax function accomplishes.

```python
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# Load data
iris = sns.load_dataset('iris')
data = torch.tensor(iris [ iris.columns[:-1] ].values, dtype=torch.float32)
labels = torch.zeros(len(iris), dtype=torch.long)
# # labels[ iris['species'] == 'setosa'] = 0
labels[ iris['species'] == 'versicolor'] = 1
labels[ iris['species'] == 'virginica'] = 2

data.shape, labels.shape

class ANNMultiClassClassifier(nn.Module):
    def __init__(self):
        super(ANNMultiClassClassifier, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, X):
        return self.stack(X)

model = ANNMultiClassClassifier()
model
```

```python
# train the model
loss_func= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr=0.01)

epochs=1000
losses = torch.zeros(epochs)
accuracies = torch.zeros(epochs)

for epoch in range(epochs):
    # feed forward
    yHat = model(data)

    # claculate loss
    loss = loss_func(yHat, labels)
    losses[epoch] = loss
    accuracies[epoch] = torch.mean((yHat.argmax(dim=1) == labels).float())*100

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```python
# Evaluate the model performance
prediction = model(data)
accuracy = torch.mean((prediction.argmax(dim=1) == labels).float())*100
accuracy
```

    tensor(98.)

```python
# plot the performance
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

![png](7_ann_52_multi-output_ANN_%28iris_dataset%29_files/7_ann_52_multi-output_ANN_%28iris_dataset%29_7_0.png)

## Question

### `BCEWithLogitsLoss` vs `CrossEntropyLoss`

#### BCEWithLogitsLoss

- **Purpose**: This loss function is used for binary classification problems. It combines a sigmoid layer and the binary cross-entropy loss in one single class. This is particularly useful when you have a single output neuron that predicts the probability of the positive class.
- **Input**: It expects raw logits (the output of the last layer before applying the sigmoid function) and target labels that are either 0 or 1.
- **Formula**: The loss is calculated as:
  $$
  text{loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\sigma(x_i)) + (1 - y_i) \log(1 - \sigma(x_i))]
  $$
  where $( \sigma )$ is the sigmoid function.

#### CrossEntropyLoss

- **Purpose**: This loss function is used for multi-class classification problems. It expects the model's output to be a vector of raw logits for each class and the target labels to be class indices.
- **Input**: It takes raw logits (not probabilities) for multiple classes and target labels that are integers representing the class index.
- **Formula**: The loss is calculated as:
  $$
  text{loss} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\frac{e^{x_{i,y_i}}}{\sum_{j} e^{x_{i,j}}}\right)
  $$
  where $( x_{i,j})$ is the logit for class $( j )$ for the $( i )-th$ sample, and $( y_i )$ is the true class index for the $( i )-th$ sample.

### Why are we not using softmax in the Sequential itself?

In PyTorch, when using `nn.CrossEntropyLoss`, you do not need to apply a softmax activation function to the output of your model. This is because `nn.CrossEntropyLoss` combines both the **softmax** activation and the **negative log-likelihood** loss in a single function. This is similar to what `BCEWithLogitsLoss` which combines both `nn.Sigmoid` and `nn.BCELoss` loss in a single function.

### Why do labels have to be of long type, but not float32?

`nn.CrossEntropyLoss` is used for multi-class classification problem. Also it combines both the **softmax** activation and the **negative log-likelihood** loss in a single function.
As it is used for multi-class classification problem, it expects the model's output to be of shape `(N, C)` (where `N` is the batch size and `C` is the number of classes) and target to be of shape `(N,)`. Each label corresponds to a class, and the `CrossEntropyLoss` expects the target labels to be integers that indicate the class index (0, 1, 2, etc.).

### Labels.shape is 1D, model is returning (150,3), still CrossEntropy can calculate loss, how?

The labels for classification tasks in PyTorch should be of type `torch.long` (or `int64`) because they represent class indices. Each label corresponds to a class, and the `CrossEntropyLoss` expects the target labels to be integers that indicate the class index (0, 1, 2, etc.). Using `float32` would not be appropriate here, as it could lead to incorrect interpretations of the labels. The model's output is a set of logits for each class, and the loss function needs to compare these logits against integer class indices.

```python

```