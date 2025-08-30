# Reference

Section: 5 \
Lecture: 23 \
Title: Mean and variance \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841948 \
Udemy Reference Link: \
Pre-Requisite:

# Mean and Variance

## Mean/Average

Mean and Average are the same things.

We calculate mean, median, and mode of a distribution. All these three are different, though, and each one has its own use case and is useful in specific distributions to display meaning.

Mean/Average basically tells us a value at the center of a distribution.

**Formula:**
$
\text{Mean} = \frac{\sum_{i=1}^{n} x_i}{n}
$

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
np.random.seed(42)
unimodal_data_1 = np.random.normal(loc=0, scale=1, size=100000)  # First mode
unimodal_data_2 = np.random.normal(loc=5, scale=1, size=100000)  # Second mode
bimodal_data = np.hstack([unimodal_data_1, unimodal_data_2])
left_skewed_data = np.random.exponential(scale=0.125, size=100000)
```

```python
plt.figure(figsize=(6, 3))
sns.kdeplot(unimodal_data_1, color='b', alpha=0.5)
plt.axvline(np.mean(unimodal_data_1), color='k', linestyle='--', label='Mean')
plt.show()
```

**But mean does not always work in all kinds of distributions. For example, in bimodal distributions or non-Gaussian distributions.**

```python
plt.figure(figsize=(6, 3))
sns.kdeplot(bimodal_data, color='b', alpha=0.5)
plt.axvline(np.mean(bimodal_data), color='k', linestyle='--', label='Mean')
plt.show()
```

```python
plt.figure(figsize=(6, 3))
sns.kdeplot(left_skewed_data, color='b', alpha=0.5)
plt.axvline(np.mean(left_skewed_data), color='k', linestyle='--', label='Mean')
plt.show()
```

## Variance and Standard Deviation

Variance/Standard Deviation is the measure of how much **data points differ from the mean**. Standard Deviation is basically the square root of Variance. Nothing else.

Larger variance means that if we plot the points on a number line, the numbers will be largely distributed. However, smaller variance means that if we plot the points on a number line, the numbers will be close to each other comparatively.

**Variance Formula:**
$
\text{Variance} = \frac{\sum_{i=1}^{n} (x_i - \text{Mean})^2}{n}
$

**Standard Deviation Formula:**
$
\text{Standard Deviation} = \sqrt{\text{Variance}}
$

**Why do we subtract the mean from each number?**

Because we want the same variance for [1, 2, 3, 3, 2, 1] and [101, 102, 103, 103, 102, 101] (just added 100 to each number).

**Why do we square?**

Squaring adds a nice property to numbers, like being able to differentiate, it is continuous, etc. If we do not square, then the distribution variance will always result in 0.

**Why not take MOD to solve the 0 Variance problem?**

There is a formula called Mean Absolute Difference (MAD). But this is less common. This concept is used in L1 regularization. However, the squared variation is used in L2 Regularization.

```python
plt.figure(figsize=(6, 3))
sns.kdeplot(unimodal_data_1, color='b', alpha=0.5)
plt.axvline(np.mean(unimodal_data_1), color='k', linestyle='--', label='Mean')

plt.axvline(np.mean(unimodal_data_1), color='k', linestyle='--', label='Mean')
plt.axvline(np.mean(unimodal_data_1) + np.std(unimodal_data_1), color='r', linestyle='--', label='Mean + 1 Std Dev')
plt.axvline(np.mean(unimodal_data_1) - np.std(unimodal_data_1), color='g', linestyle='--', label='Mean - 1 Std Dev')

plt.legend()
plt.show()
```

## Example in numpy

```python
x = [1,2,4,6,5,4,0]
```

```python
np.mean(x), np.sum(x)/len(x) # both are same
```

```python
np.var(x), np.sum((x - np.mean(x))**2) / ( len(x)-1 ) # both are different => because of degree of freedom
```

```python
np.var(x, ddof=1), np.sum((x - np.mean(x))**2) / ( len(x)-1 ) # now both are same. BUT this does not matter in huge data
```

## Example in torch

```python
x = torch.tensor([1,2,4,6,5,4,0], dtype=torch.float32)
```

```python
# in torch, ddof param is same as correction and equal to 1 by default
torch.mean(x), torch.var(x), torch.sum(( x - torch.mean(x) )**2) / (len(x)-1)
```