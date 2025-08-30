# Reference

Section: 7 \
Lecture: 45 \
Title: ANN for regression \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842126 \
Udemy Reference Link: \
Pre-Requisite:

# ANN for Regression

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

```python
# Generate Data
size = 50

x = torch.randn(size,1)
y = x + torch.randn(size,1)/2
# torch.randn(size,1)/2 is a noise. Why?, because it is not constant like y-intercept
x.shape, y.shape
```

    (torch.Size([50, 1]), torch.Size([50, 1]))

```python
# Plot
plt.plot(x,y,'s', label='Training Data')
plt.legend()
plt.show()
```

![png](7_ann_45_ann_for_regression_files/7_ann_45_ann_for_regression_4_0.png)

```python
class ANNRegression(nn.Module):
    def __init__(self):
        super(ANNRegression, self).__init__()
        self. stack = nn.Sequential(
            nn.Linear(1, 1)
        )
    def forward(self, data):
        return self.stack(data)
model = ANNRegression()
```

```python
epochs = 100
loss_func = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

```python
# training
losses = torch.zeros(epochs)

for epoch in range(epochs):
    yHat = model(x)

    loss = loss_func(y, yHat)
    losses[epoch] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```python
# Measure Model Performance
prediction = model(x)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(x, y, 's', color='b', label='Training Data')
axes[0].plot(x, prediction.detach(), 'o', color='g', label='Model Prediction')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Reality vs Prediction')
axes[0].legend()

axes[1].scatter(range(epochs), losses.detach(), color='b')
axes[1].set_xlabel('epochs')
axes[1].set_ylabel('cost')
axes[1].set_title('Loss Per Epoch')
plt.show()
```

![png](7_ann_45_ann_for_regression_files/7_ann_45_ann_for_regression_8_0.png)

### If DL is so great, why don't we all switch to DL models instead of traditional statistical models?

Traditional statistical models tend to work better on smaller datasets, are better mathematically characterize (e.g. guaranteed optimal solutions) and are more interpretable. But DL models, does not provide any guarantee the optimal solution, but proceed towards a good solution.
