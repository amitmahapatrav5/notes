# Reference

Section: 5 \
Lecture: 19 \
Title: Softmax \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841936 \
Udemy Reference Link: \
Pre-Requisite:

# Softmax

### Softmax Function Definition

The softmax function is defined as:

$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \ldots, K
$

Where:

- $ z $ is a vector of raw scores (logits)
- $ K $ is the number of classes
- $ \sigma(z_i) $ is the probability of the $ i $-th class

### Softmax Example for $ Z = \{1, 2, 3\} $

1. **Input Vector**:
   $$Z = \{1, 2, 3\}$$

2. **Calculate Exponentials**:

   - $ e^{1} \approx 2.718 $
   - $ e^{2} \approx 7.389 $
   - $ e^{3} \approx 20.085 $

3. **Sum of Exponentials**:
   $$\text{Sum} = e^{1} + e^{2} + e^{3} \approx 2.718 + 7.389 + 20.085 \approx 30.192$$

4. **Calculate Softmax Probabilities**:

   - For $ z_1 = 1 $:
     $$\sigma(z_1) = \frac{e^{1}}{\text{Sum}} \approx \frac{2.718}{30.192} \approx 0.0907$$
   - For $ z_2 = 2 $:
     $$\sigma(z_2) = \frac{e^{2}}{\text{Sum}} \approx \frac{7.389}{30.192} \approx 0.2447$$
   - For $ z_3 = 3 $:
     $$\sigma(z_3) = \frac{e^{3}}{\text{Sum}} \approx \frac{20.085}{30.192} \approx 0.6652$$

5. **Softmax Output**:
   $$\sigma(Z) \approx \{0.0907, 0.2447, 0.6652\}$$

The outputs of the softmax function can be interpreted as probabilities since their sum always equals 1. However, it's important to note that these values do not necessarily represent true probabilities in all contexts.

```python
import numpy as np

Z = [1, 2, 3]
a, b, c = np.exp(Z[0]), np.exp(Z[1]), np.exp(Z[2])
abc = a + b + c

print( a/abc, b/abc, c/abc )
np.sum([a/abc, b/abc, c/abc])
```

```python
import torch

Z = torch.tensor([1, 2, 3])
a, b, c = torch.exp(Z[0]), torch.exp(Z[1]), torch.exp(Z[2])
abc = a + b + c

print( a/abc, b/abc, c/abc )
torch.sum(torch.tensor( [a/abc, b/abc, c/abc] ) )
```

```python
import torch

Z = torch.tensor([1, 2, 3], dtype=torch.float32)
softmax = torch.nn.Softmax(dim=0)
softmax(Z)
```