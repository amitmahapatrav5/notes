# Reference

Section: 8 \
Lecture: 61 \
Title: What is overfitting and is it as bad as they say? \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27844168 \
Udemy Reference Link: \
Pre-Requisite:

# What is overfitting and is it as bad as they say?

```python
from IPython.display import Image, display
```

## Problem of overfitting and underfitting

Higher parameter model is not necessarily always better than low parameter model.
In the below diagram, the 10 parameter model fits the data more accurately than the 2 parameter model. But with 10 parameter model we see the model is getting overfitted. But turns out that, the 10 parameter model is performing very bad in Dataset 2.

```python
display(Image(filename='overfitting_61_03_23.png', width=500, height=400))
```

![png](8_overfitting_and_cross_validation_61_what_is_overfitting_and_is_it_as_bad_as_they_say_files/8_overfitting_and_cross_validation_61_what_is_overfitting_and_is_it_as_bad_as_they_say_4_0.png)

```python
display(Image(filename='underfitting_63_03_33.png', width=500, height=400))
```

![png](8_overfitting_and_cross_validation_61_what_is_overfitting_and_is_it_as_bad_as_they_say_files/8_overfitting_and_cross_validation_61_what_is_overfitting_and_is_it_as_bad_as_they_say_5_0.png)

| Overfitting                                            | Underfitting                        |
| ------------------------------------------------------ | ----------------------------------- |
| Overly sensitive to noise                              | **Less sensitive to noise**         |
| Increased sensitivity to subtle effects                | Less likely to detect true effects  |
| Reduced generalizability                               | Reduced generalizability            |
| Over-parameterized models become difficult to estimate | **Parameters are better estimated** |
|                                                        | **Good results with less data**     |

generalizability means, the ability of the model to predict on new data, data that the model has not seen before.

## Researcher Overfitting / Researcher Degrees of Freedom

Researcher degrees of freedom refers to the various choices that a researcher or data analyst has when it comes to cleaning, organizing, and selecting data, as well as the decisions made in model creation, including which parameters and architectures to use. This flexibility can lead to issues, as fine-tuning a model for a specific dataset may hinder its ability to generalize to new datasets.

### Example:

Imagine you are analyzing a dataset and decide to experiment with three different deep learning model architectures, which we’ll label as Model A, Model B, and Model C. After running these models, you find their performance unsatisfactory.

To improve the results, you revisit the dataset and apply a different set of criteria for data cleaning and selection. After reprocessing the data, you test the three models again. This time, Model B emerges as the top performer, achieving the highest accuracy.

Excited by these results, you publish your findings in a scientific journal, share the model on GitHub, or write a blog post about it. However, because you selected and tested these models on two different versions of the same dataset, you risk overfitting the entire model space to this specific scenario.

As a result, you cannot confidently assert that Model B will perform equally well on a different dataset. The concept of researcher degrees of freedom highlights that the more decisions you make based on a particular dataset, the less likely it is that your chosen model will be effective when applied to new or different data.

### How to Avoid Researcher Overfitting

To mitigate the risk of researcher overfitting, consider the following strategies:

1. **Predefine Model Architecture**:

   - Choose the model architecture in advance and make only minor adjustments as needed.
   - This approach is commonly used in traditional statistics and machine learning, where researchers have established effective models over time for frequently studied problems.
   - For example, in image recognition tasks, starting with a well-known architecture like ResNet and applying transfer learning can help avoid overfitting. Transfer learning leverages pre-trained models, allowing for better generalization.

2. **Reserve a Test Set**:
   - Set aside a portion of the data as a test set that is not used during the model training process.
   - Build and train your models (e.g., Model A, B, and C) using the training data, but keep the test data completely separate.
   - Only evaluate the models on the test data after all training and parameter selection is complete. This ensures that the model's performance is assessed on unseen data, providing a more accurate measure of its generalizability.

```python

```