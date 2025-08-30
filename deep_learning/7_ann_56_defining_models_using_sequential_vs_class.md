# Reference

Section: 7 \
Lecture: 56 \
Title: Defining models using sequential vs. class \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842158 \
Udemy Reference Link: \
Pre-Requisite:

# Defining Models using Sequential vs Class

`nn.Sequential` is pretty easy and quick to create. However it is pretty limited in terms of creating model architecture. However using a class, we can create complex model architecture and the using the class approach we can perform any level of customization we want.

`nn.ModuleDict` is used

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```

## Building ANN using just nn.Sequential()

```python
ANNSequential = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)

ANNSequential(torch.randn(10, 2))
```

    tensor([[-0.2822,  0.3955, -0.2888],
            [-0.4650,  0.2975, -0.1183],
            [-0.0393,  0.4018, -0.3267],
            [-0.4650,  0.2975, -0.1183],
            [ 0.0518,  0.4587, -0.4060],
            [ 0.1414,  0.4263, -0.3784],
            [-0.0868,  0.4884, -0.4243],
            [-0.1864,  0.4833, -0.4058],
            [-0.4650,  0.2975, -0.1183],
            [-0.3524,  0.4256, -0.3161]], grad_fn=<AddmmBackward0>)

## Building ANN without nn.Sequential() in a class

```python
class ANNClass(nn.Module):
    def __init__(self):
        super(ANNClass, self).__init__()
        self.l01 = nn.Linear(2, 4)
        self.l12 = nn.Linear(4, 3)

    def forward(self, X):
        Y = self.l01(X)
        Y = nn.ReLU(Y)
        Y = self.l12(Y)
        return Y

ANNClass()
```

    ANNClass(
      (l01): Linear(in_features=2, out_features=4, bias=True)
      (l12): Linear(in_features=4, out_features=3, bias=True)
    )

## Building ANN using nn.Sequential() in a class

```python
class ANNSequentialClass(nn.Module):
    def __init__(self):
        super(ANNSequentialClass, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )

    def forward(self, X):
        return self.stack(X)

ANNSequentialClass()
```

    ANNSequentialClass(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=3, bias=True)
      )
    )

## Building ANN using nn.ModuleDict()

```python
class ANNModuleDict(nn.Module):
    def __init__(self, n_layers, n_units_per_layer):
        super(ANNModuleDict, self).__init__()

        self.n_hidden_layers = n_layers
        self.stack = nn.ModuleDict()

        # Input => Hidden 0
        self.stack['ih1'] = nn.Linear(4, n_units_per_layer)

        # Building Hidden Layers
        for layer_no in range(1, n_layers):
            self.stack[f'h{layer_no}h{layer_no+1}'] = nn.Linear(n_units_per_layer, n_units_per_layer)

        # Hidden n => output
        self.stack[f'h{self.n_hidden_layers}o'] = nn.Linear(n_units_per_layer, 3)

    def forward(self, X):
        Y = self.stack['ih1'](X)

        for layer_no in range(1, self.n_hidden_layers):
            Y = F.relu( self.stack[f'h{layer_no}h{layer_no+1}'](Y) )

        Y = self.stack[f'h{self.n_hidden_layers}o'](Y)

        return Y

ANNModuleDict(n_layers=3, n_units_per_layer=10)
```

    ANNModuleDict(
      (stack): ModuleDict(
        (ih1): Linear(in_features=4, out_features=10, bias=True)
        (h1h2): Linear(in_features=10, out_features=10, bias=True)
        (h2h3): Linear(in_features=10, out_features=10, bias=True)
        (h3o): Linear(in_features=10, out_features=3, bias=True)
      )
    )

```python

```