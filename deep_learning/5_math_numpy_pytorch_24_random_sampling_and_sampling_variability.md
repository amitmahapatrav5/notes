# Reference

Section: 5 \
Lecture: 24 \
Title: Random sampling and sampling variability \
TCS Udemy Reference Link: https://tcsglobal.udemy.com/course/deeplearning_x/learn/lecture/27841952 \
Udemy Reference Link: \
Pre-Requisite:

# Random Sampling and Sampling Variability

Deep Learning requires a lot of data to train the model. But **WHY**?

Different samples from the same population can have different values of same measurement. This called **Sampling Variability**.

For example, "What is the average height of an Indian?" If we collect sample from Assam, and another sample from Punjab, the values will be different.

So a Single measurement may be a unreliable estimate of a population parameter.

Similarly, not all cat pictures are same. Because not all cats look the same. If that were the case, then we need only 1 picture. But because not all cat looks exactly same, but similar, neural networks need lots of cat images to recognize the pattern of similarity. What we are ultimately doing is, averaging togather many samples. **Averaging togather many samples to approximate the true population mean. This is basically Law of Large Numbers.**

The higher the variability there is, the more the number of samples we need. So if we are working with a simple data values, we do not need that many number of samples. But if we are going with a much more complex deep convolutional neural network, we need more number of samples.

Below 2 statement need more explanation. \
**Non-random sampling can introduce systematic biases in DL models.** \
**Non-representative sampling causes overfitting and limits generalizability.**

## Random Sampling Example

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]

population_mean = np.mean(x) # This will always be constant
sample_mean = np.random.choice(x, size=5, replace=False).mean() # This will change

population_mean, sample_mean
```

**But if we perform the above example so many times, then we can approximate very close to population mean. This is called Law of Large Numbers**

```python
# Need to learn about histograms to understand the below code

nExpers = 10000

# run the experiment!
sampleMeans = np.zeros(nExpers)
for i in range(nExpers):

  # step 1: draw a sample
  sample = np.random.choice(x,size=15,replace=True)

  # step 2: compute its mean
  sampleMeans[i] = np.mean(sample)



# show the results as a histogram
plt.hist(sampleMeans,bins=40,density=True)
plt.plot([population_mean,population_mean],[0,.3],'m--')
plt.ylabel('Count')
plt.xlabel('Sample mean')
plt.show()
```