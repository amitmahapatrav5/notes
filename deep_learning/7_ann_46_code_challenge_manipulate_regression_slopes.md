# Reference

Section: 7 \
Lecture: 46 \
Title: CodeChallenge: manipulate regression slopes \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842128 \
Udemy Reference Link: \
Pre-Requisite:

# Parametric Experiment - Slope vs Loss and Slope vs Performance

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
```

```python
class ANNRegression(nn.Module):
    def __init__(self):
        super(ANNRegression, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, 1),
                nn.ReLU(),
            nn.Linear(1, 1)
        )
    def forward(self, X):
        return self.stack(X)

ANNRegression()
```

    ANNRegression(
      (stack): Sequential(
        (0): Linear(in_features=1, out_features=1, bias=True)
        (1): ReLU()
        (2): Linear(in_features=1, out_features=1, bias=True)
      )
    )

```python
def build_and_train(X, y):

    # built the model
    model = ANNRegression()

    loss_func = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train the model
    epochs = 500
    losses = torch.zeros(epochs)
    for epoch in range(epochs):
        # feed forward
        yHat = model(X)

        # compute loss
        loss = loss_func(yHat, y)
        losses[epoch] = loss

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prediction = model(X)
    return prediction, losses
```

```python
def generate_data(m):
    N = 50

    X = torch.randn(N, 1)
    y = m*X + torch.randn(N, 1)/2

    return X, y
```

```python
# Test everything once
X, y = generate_data(m=1)
prediction, losses = build_and_train(X, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Reality vs Model Prediction
axes[0].scatter(X, y, color='b', marker='+', label='Reality')
axes[0].scatter(X, prediction.detach(), marker='*', color='g', label='Model Prediction')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Reality vs Model Prediction')
axes[0].legend()

# epoch vs loss
axes[1].plot(range(len(losses)), losses.detach(), color='m', marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Epoch vs Loss')

plt.show()
```

![png](7_ann_46_code_challenge_manipulate_regression_slopes_files/7_ann_46_code_challenge_manipulate_regression_slopes_6_0.png)

```python
# Parametric Experiment
slopes = torch.linspace(-2, 2, 21)
exps = 50

result = torch.zeros(len(slopes), exps, 2)

for slp in range(len(slopes)):
    for exp in range(exps):
        X, y = generate_data(slopes[slp])
        prediction, losses = build_and_train(X, y)

        # store the performance and final loss
        result[slp, exp, 0] = torch.corrcoef(torch.vstack((prediction.T, y.T)))[0, 1]
        result[slp, exp, 1] = losses[-1]
result[torch.isnan(result)]=0
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(slopes.detach(), torch.mean(result[:, :, 0], dim=1).detach(), marker='o', color='b')
axes[0].set_title('Slope Vs Performance')
axes[0].set_xlabel('Slope')
axes[0].set_ylabel('co-relation between reality and prediction')

axes[1].plot(slopes.detach(), torch.mean(result[:, :, 1], dim=1).detach(), marker='s', color='m')
axes[1].set_title('Slope Vs Loss')
axes[1].set_xlabel('Slope')
axes[1].set_ylabel('Last Training Epoch Model Loss')

plt.show()
```

![png](7_ann_46_code_challenge_manipulate_regression_slopes_files/7_ann_46_code_challenge_manipulate_regression_slopes_8_0.png)

### Why Performance is very low in case of slope=0?

When slope=0, the model does not get more relation between x and y. However, in case of other slopes, the model could learn the relation between the x and y. You can see the same from below diagram.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

X, y = generate_data(m=0)
axes[0].scatter(X, y)

X, y = generate_data(m=2)
axes[1].scatter(X, y)

plt.show()
```

![png](7_ann_46_code_challenge_manipulate_regression_slopes_files/7_ann_46_code_challenge_manipulate_regression_slopes_10_0.png)

### Why Loss is very low in case of slope=0 and high in other cases?

This is because, in case of higher slopes, the variance of y is higher, but in case where slope is 0, the variance is lower. Normalization of data can solve this issue. In the below diagram, when slope=0, the variance is from -1 to 1, but when slope=2, variance is from -4 to 1.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

X, y = generate_data(m=0)
axes[0].scatter(X, y)

X, y = generate_data(m=2)
axes[1].scatter(X, y)

plt.show()
```

![png](7_ann_46_code_challenge_manipulate_regression_slopes_files/7_ann_46_code_challenge_manipulate_regression_slopes_12_0.png)
