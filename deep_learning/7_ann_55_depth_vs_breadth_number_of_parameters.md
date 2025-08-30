# Reference

Section: 7 \
Lecture: 55 \
Title: Depth vs. breadth: number of parameters \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842154 \
Udemy Reference Link: \
Pre-Requisite:

# Depth vs Breadth Number Of Parameters

## Breadth vs Depth of ANN

Depth is the number of hidden layers (layers between input and output)

Breadth/width is the number of units per hidden layer(can vary across layers)

```python
import torch.nn as nn
from torchsummary import summary
```

```python
class ANNWide(nn.Module):
    def __init__(self):
        super(ANNWide, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 3)
        )
    def forward(self, X):
        return self.stack(X)

ANNWide()
```

    ANNWide(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=4, bias=True)
        (1): Linear(in_features=4, out_features=3, bias=True)
      )
    )

```python
class ANNDeep(nn.Module):
    def __init__(self):
        super(ANNDeep, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 3)
        )
    def forward(self, X):
        return self.stack(X)

ANNDeep()
```

    ANNDeep(
      (stack): Sequential(
        (0): Linear(in_features=2, out_features=2, bias=True)
        (1): Linear(in_features=2, out_features=2, bias=True)
        (2): Linear(in_features=2, out_features=3, bias=True)
      )
    )

### Counting number of Nodes/Units in ANN [`.named_parameters()`]

```python
# named_parameters() is an iterable that returns the tuple (name,numbers)

n_ann_wide_nodes = 0
for param_name, param_weights in ANNWide().named_parameters():
    # print(param_name, end='\n\n')
    # print(param_tensor, end='\n\n')
    if 'bias' in param_name:
        n_ann_wide_nodes += len(param_weights)

n_ann_deep_nodes = 0
for param_name, param_weights in ANNDeep().named_parameters():
    if 'bias' in param_name:
        n_ann_deep_nodes += len(param_weights)


# In both the cases, we are not considering the nodes in input layer
print(f'Number of nodes in ANNWide is {n_ann_wide_nodes}')
print(f'Number of nodes in ANNDeep is {n_ann_deep_nodes}')
```

    Number of nodes in ANNWide is 7
    Number of nodes in ANNDeep is 7

### Counting number of Trainable Parameters in ANN [`.numel()`, `.parameters()`]

```python
# Note, we pass ANNWide().parameters in optimizer constructor
n_ann_wide_trained_param = 0
for param in ANNWide().parameters():
    # print(param)
    # print(param.requires_grad)
    n_ann_wide_trained_param += param.numel()

n_ann_deep_trained_param = 0
for param in ANNDeep().parameters():
    # print(param)
    # print(param.requires_grad)
    n_ann_deep_trained_param += param.numel()

print(f'Number of trainable parameters in ANNWide is {n_ann_wide_trained_param}')
print(f'Number of trainable parameters in ANNDeep is {n_ann_deep_trained_param}')
```

    Number of trainable parameters in ANNWide is 27
    Number of trainable parameters in ANNDeep is 21

```python
# summary(ANNWide, (1,2))
# this is supposed to work, but not sure why it is not working
```