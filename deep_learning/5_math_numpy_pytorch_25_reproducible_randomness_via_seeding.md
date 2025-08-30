# Reference

Section: 5 \
Lecture: 25 \
Title: Reproducible randomness via seeding \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841958 \
Udemy Reference Link: \
Pre-Requisite:

# Reproducible Randomness via Seeding

**Key Concept of Seed Phrases**

A seed phrase is crucial because it establishes a fixed starting point for generating a sequence of random numbers. Once a seed is set, the resulting sequence remains consistent, regardless of the method used or the structure in which the data is stored.

**For Example:**

Consider a seed value of 10. For the sake of simplicity, let's assume that the random number generation algorithm produces numbers by adding the seed to the previous random number, with the initial random number being the seed itself. In this case, the sequence would be: 10, [20, 30], [[40], [50]], 60,... and so on.

This illustrates that the shape, size, or method of data storage does not affect the sequence; it will always remain constant as long as the same seed is used.

In practical applications, however, random number generation algorithms are typically much more complex than simply adding the seed to the previous number in the sequence.

**Why required?**

Before starting to train our deep learning model, we initialize the weights to random values. But when we share our algorithm, model and dataset we used to achieve the result, with anyone, and the other person initialize the weights randomly, those initial weights will be different from ours. To ensure the other person can reproduce the result exactly same to ours, we use this seeding mechanism.

## Seeding in numpy

```python
import numpy as np
```

### Older Seeding Approach

```python
np.random.seed(10) # Set at module level

print( np.random.rand(5) ) # [0.77132064 0.02075195 0.63364823 0.74880388 0.49850701]
print( np.random.rand(5) ) # [0.22479665 0.19806286 0.76053071 0.16911084 0.08833981]
```

    [0.77132064 0.02075195 0.63364823 0.74880388 0.49850701]
    [0.22479665 0.19806286 0.76053071 0.16911084 0.08833981]

### Newer Seeding Approach

```python
seed1 = np.random.RandomState(10) # Set at object level
seed2 = np.random.RandomState(13) # Set at object level

print( seed1.rand(5) ) # [0.77132064 0.02075195 0.63364823 0.74880388 0.49850701]
print( seed2.rand(5) ) # [0.77770241 0.23754122 0.82427853 0.9657492  0.97260111]

print( seed1.rand(5) ) # [0.22479665 0.19806286 0.76053071 0.16911084 0.08833981]
print( seed2.rand(5) ) # [0.45344925 0.60904246 0.77552651 0.64161334 0.72201823]
```

    [0.77132064 0.02075195 0.63364823 0.74880388 0.49850701]
    [0.77770241 0.23754122 0.82427853 0.9657492  0.97260111]
    [0.22479665 0.19806286 0.76053071 0.16911084 0.08833981]
    [0.45344925 0.60904246 0.77552651 0.64161334 0.72201823]

## Seeding in torch

```python
import torch
```

```python
torch.manual_seed(10) # Torch however seems to use the module level approach

print( torch.rand(5) ) # tensor([0.4581, 0.4829, 0.3125, 0.6150, 0.2139])
print( torch.rand(5) ) # tensor([0.4118, 0.6938, 0.9693, 0.6178, 0.3304])
```

    tensor([0.4581, 0.4829, 0.3125, 0.6150, 0.2139])
    tensor([0.4118, 0.6938, 0.9693, 0.6178, 0.3304])
