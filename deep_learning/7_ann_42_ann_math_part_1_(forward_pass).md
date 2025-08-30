# Reference

Section: 7 \
Lecture: 42 \
Title: ANN math part 1 (forward prop) \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842118 \
Udemy Reference Link: \
Pre-Requisite:

# Basic Mathematics that happens in a single Neuron

## Math that happens in every neuron

In every Neuron, this is what basically happens \
$Y = \sigma( bias + \sum(X^T W) )$ \
But we can write the same thing as $Y = \sigma( \sum(X^T W) )$ \
HOW ?

$Y = \sigma( bias + \sum(X^T W) )$ \
$Y = \sigma( 1*bias + \sum(X^T W) )$ \
$Y = \sigma( 1*W_0 + \sum(X^T W) )$ \
$Y = \sigma( 1*W_0 + X_1W_1 + X_2W_2 + ... + X_nW_n )$ \
$Y = \sigma( X_0*W_0 + X_1W_1 + X_2W_2 + ... + X_nW_n )$ \
$Y = \sigma( \sum(X^T W) )$

## Why do we need activation function

**This is mainly happeing because, if you observe the heatmaps carefully, \
The left figure, where we have not used the activation function, the band value is in range from -10 to 12 \
But in the right figure, where we have used the activation function, the band value is in range from 0 to 1. \
So, in a way, we are kind of transforming the coordinate.**

```python
import torch
import matplotlib.pyplot as plt

def func_linear(x):
    return 1 - 2*x[:, 0] + x[:, 1]

def func_nonlinear(x):
    return torch.sigmoid(1 - 2*x[:, 0] + x[:, 1])

start, end = -4, 4
nCount = 100

# Generate a grid of points
x1 = torch.linspace(start, end, nCount)
x2 = torch.linspace(start, end, nCount)
x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='xy')
x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

# Compute the function values
z_linear = func_linear(x_grid).reshape(nCount, nCount)
z_nonlinear = func_nonlinear(x_grid).reshape(nCount, nCount)

# Convert to numpy for plotting
x1_np = x1_grid.numpy()
x2_np = x2_grid.numpy()
z_linear_np = z_linear.numpy()
z_nonlinear_np = z_nonlinear.numpy()

# Create the heatmaps
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Linear function heatmap
contour1 = axs[0].contourf(x1_np, x2_np, z_linear_np, levels=50, cmap='viridis')
axs[0].set_title('Linear Function Heatmap')
axs[0].set_xlabel('x1')
axs[0].set_ylabel('x2')
fig.colorbar(contour1, ax=axs[0], label='Function Value')
axs[0].axhline(0, color='black', linewidth=0.5, ls='--')
axs[0].axvline(0, color='black', linewidth=0.5, ls='--')
axs[0].grid(color='gray', linestyle='--', linewidth=0.5)

# Nonlinear function heatmap
contour2 = axs[1].contourf(x1_np, x2_np, z_nonlinear_np, levels=50, cmap='viridis')
axs[1].set_title('Nonlinear Function Heatmap (with Activation)')
axs[1].set_xlabel('x1')
axs[1].set_ylabel('x2')
fig.colorbar(contour2, ax=axs[1], label='Function Value')
axs[1].axhline(0, color='black', linewidth=0.5, ls='--')
axs[1].axvline(0, color='black', linewidth=0.5, ls='--')
axs[1].grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

```

![png](7_ann_42_ann_math_part_1_%28forward_pass%29_files/7_ann_42_ann_math_part_1_%28forward_pass%29_6_0.png)

Most Common Activation Functions are

1. Sigmoid (Often used in the nodes in the output layer of the model)
2. Hyperbolic Tangent (Often used in the nodes in the middle of the model)
3. ReLU (Often used in the nodes in the middle of the model)

There are lot of other activation functions as well. \
Also, many activation functions are just variants of these 3.

```python

```