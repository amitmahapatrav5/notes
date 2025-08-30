# Reference

Section: 7 \
Lecture: 57 \
Title: Model depth vs. breadth \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842160 \
Udemy Reference Link: \
Pre-Requisite:

# Model depth vs breadth

```python
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
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
```

    (torch.Size([150, 4]), torch.Size([150]))

```python
# Build the model
class ANNIris(nn.Module):
    def __init__(self, n_layers, n_units_per_layer):
        super(ANNIris, self).__init__()

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

ANNIris(n_layers=4, n_units_per_layer=12)
```

    ANNIris(
      (stack): ModuleDict(
        (ih1): Linear(in_features=4, out_features=12, bias=True)
        (h1h2): Linear(in_features=12, out_features=12, bias=True)
        (h2h3): Linear(in_features=12, out_features=12, bias=True)
        (h3h4): Linear(in_features=12, out_features=12, bias=True)
        (h4o): Linear(in_features=12, out_features=3, bias=True)
      )
    )

```python
# A quick test of running some numbers through the model.
# This simply ensures that the architecture is internally consistent.

# 10 samples, 4 dimensions
tmpx = torch.randn(10,4)

# run it through the DL
y = ANNIris(n_layers=4, n_units_per_layer=12)(tmpx)

# exam the shape of the output
print( y.shape ), print(' ')

# and the output itself
print(y)
```

    torch.Size([10, 3])

    tensor([[-0.0290, -0.1348, -0.3418],
            [-0.0212, -0.1566, -0.3328],
            [-0.0133, -0.1377, -0.3232],
            [-0.0182, -0.1663, -0.3292],
            [-0.0145, -0.1673, -0.3272],
            [-0.0162, -0.1685, -0.3233],
            [-0.0115, -0.1704, -0.3232],
            [ 0.0047, -0.1331, -0.3053],
            [-0.0078, -0.1660, -0.3306],
            [-0.0262, -0.1481, -0.3279]], grad_fn=<AddmmBackward0>)

```python
# train the model
epochs=1000

def train_the_model(model):
    loss_func= nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(model.parameters(), lr=0.01)

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

    accuracy = torch.mean( ( torch.argmax( prediction, dim=1 ) == labels ).float() )*100
    n_trainable_params = sum(param.numel() for param in model.parameters())

    return accuracy, n_trainable_params
```

```python
model = ANNIris(n_layers=4, n_units_per_layer=12)
accuracy, n_trainable_params = train_the_model(model)
```

## Parametric Experiment

```python
# define the model parameters
numlayers = range(1,6)         # number of hidden layers
numunits  = np.arange(4,101,3) # units per hidden layer

# initialize output matrices
accuracies  = np.zeros((len(numunits),len(numlayers)))
totalparams = np.zeros((len(numunits),len(numlayers)))

# number of training epochs
numepochs = 500


# start the experiment!
for unitidx in range(len(numunits)):
  for layeridx in range(len(numlayers)):

    # create a fresh model instance
    net = ANNIris(numunits[unitidx],numlayers[layeridx])

    # run the model and store the results
    acc,nParams = train_the_model(net)
    accuracies[unitidx,layeridx] = acc

    # store the total number of parameters in the model
    totalparams[unitidx,layeridx] = nParams
```

```python
# show accuracy as a function of model depth
fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(numunits,accuracies,'o-',markerfacecolor='w',markersize=9)
ax.plot(numunits[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numunits[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.legend(numlayers)
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()
```

![png](7_ann_57_model_depth_vs_breadth_files/7_ann_57_model_depth_vs_breadth_10_0.png)

```python
# Maybe it's simply a matter of more parameters -> better performance?

# vectorize for convenience
x = totalparams.flatten()
y = accuracies.flatten()

# correlation between them
r = np.corrcoef(x,y)[0,1]

# scatter plot
plt.plot(x,y,'o')
plt.xlabel('Number of parameters')
plt.ylabel('Accuracy')
plt.title('Correlation: r=' + str(np.round(r,3)))
plt.show()
```

    C:\my_learning\python\ai\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:3045: RuntimeWarning: invalid value encountered in divide
      c /= stddev[:, None]
    C:\my_learning\python\ai\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:3046: RuntimeWarning: invalid value encountered in divide
      c /= stddev[None, :]

![png](7_ann_57_model_depth_vs_breadth_files/7_ann_57_model_depth_vs_breadth_11_1.png)

## Learning

It is not necessary that more number of layers means more better performance

It is not nacessary that more number of units per later means more better performance

Also it is not necessary that the more the number of trainable parameters means the better the performance

The model gets complecated very quickly when we have a need to tune various metaparameters.
