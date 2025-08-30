# Reference

Section: 5 \
Lecture: 22 \
Title: Min/max and argmin/argmax \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841946 \
Udemy Reference Link: \
Pre-Requisite:

# Min/Max and argmin/argmax

`max`/`min`: These functions return the minimum and maximum elements in a collection.

`argmin`/`argmax`: These functions return the indices of the minimum and maximum elements in a collection.

**Purpose of `argmin`/`argmax`**: In a neural network model trained for a classification task that utilizes the SoftMax activation function in the output layer, the `argmax` function is essential. It identifies the index of the maximum output, which indicates the class to which the input belongs.

```python
import numpy as np

arr = np.array([1, -1, 0, 3, 5])
print( np.min(arr), np.max(arr), np.argmin(arr), np.argmax(arr) )

matrix = np.array([ [1, -1, 3],
                    [0, 1, -6]
                  ])

print( np.min(matrix), np.max(matrix) ) # Min/Max in the entire matrix
print( np.min(matrix, axis=1), np.max(matrix, axis=1) ) # Min/Max in axis=1

```

```python
import torch

arr = torch.tensor([1, -1, 0, 3, 5])
print( torch.min(arr), torch.max(arr), torch.argmin(arr), torch.argmax(arr) )

matrix = torch.tensor([ [1, -1, 3],
                        [0, 1, -6]
                      ])

print( torch.min(matrix), torch.max(matrix) ) # Min/Max in the entire matrix
print( torch.min(matrix, axis=1), torch.max(matrix, axis=1) ) # Min/Max in axis=1
```