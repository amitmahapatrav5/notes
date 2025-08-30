# Reference

Section: 7 \
Lecture: 51 \
Title: Why multilayer linear models don't exist \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842142 \
Udemy Reference Link: \
Pre-Requisite:

# Why multilayer linear models don't exist

**Without non-linear activations, multiple "layers" collapse into a single linear transformation.**
Any sequence of linear functions can be combined into one equivalent linear function by multiplying their weight matrices.

**Non-linearities are essential between layers.**  
 You must include some form of non-linearity—such as ReLU, tanh, sigmoid, pooling, etc.—to give depth to your model’s representational power.

### Example (with non-linearity)

- Layer 1 output: $\hat y_1 = \sigma(W_1^T . x )$
- Layer 2 output: $\hat y_2 = \sigma(W_2^T . x )$
  - $\sigma$ can be any non-linear activation (ReLU, tanh, etc.)

### What Happens If $\sigma$ is Linear or Omitted

- If $\sigma$ is the identity (a linear function), it can be absorbed into the weight matrices.
- Layer 2 becomes:
  $$
  \hat y_2 = W_2^T \bigl(W_1^T x\bigr)
           = (W_1 W_2)^T x
           = W_{\alpha}^T x
  $$
- All subsequent layers similarly collapse: effectively one layer with weights.

### Numerical Illustration

- Non-linear example: $log_{10}( 5 + 5 ) = 1$ but $log_{10}( 5 + 5 ) \neq 1$ \
  Demonstrates that non-linear functions do not distribute over addition.
- Linear example: $A.( 5 + 5 ) = A . 10$ and $ A.5 + A.5 = A.10 $ \
  Shows linear functions distribute, allowing collapse into a single weighted sum.