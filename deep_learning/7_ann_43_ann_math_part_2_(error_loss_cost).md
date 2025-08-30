# Reference

Section: 7 \
Lecture: 43 \
Title: ANN math part 2 (errors, loss, cost) \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27842120 \
Udemy Reference Link: \
Pre-Requisite:

# Loss Functions and Cost Function

## Loss Function

Most Commonly used Loss Functions are Mean Squared Error and Cross Entropy Error.
There are lots of other errors, but majority are basically variations of these 2 types of Loss Function.

**Mean Squared Error**

- This error function is mainly used when the model predicts continuous data
- Example: House Price Prediction, Temprature Prediction etc
- $ L = \frac{1}{2} ( \hat{y} - y )^{2} $

**Cross Entropy Error**

- This error function is used when the model predicts probability
- Example: Text Sentiment Classification, Digit Recognization etc
- $ L = -( y _ log ({\hat{y}}) + (1-y) _ log(1 - \hat{y}) ) $

## Loss Function Vs Cost Function

- Cost Function is literally just the average of losses for all the training data.
- We calculate the loss for every single data point using loss function. And cost function is just the average of all those losses.
- $ C = \frac{1}{n} \sum\_{i=1}^{n} L(\hat{y}, y)$
- The entire goal of deep learning is **find the weights such that it minimizes the cost function.**

### Why train on cost but not loss

- Training the model on each sample is time consuming and may lead to overfitting.
- Also averaging over too many sample, may decrease the sensitivity.
- The best approach is to train the model in "batches" of samples.

#### Example

- Say we have 2000 Samples in training data.
- Calculating Loss for each sample and then chainging the weights will overfit the model
- Calculating Cost from Loss of each sample and the changing the weights might decrease the sensitivity of the model
- Best way is, create batches, each having say 20 samples and calculate the cost by averaging the loss of each sample in the batch and change the weights.

```python

```

```python

```