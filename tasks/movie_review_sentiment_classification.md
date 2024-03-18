## Movie Review Sentiment Classification

### Goal

In this task, learners will practice implementing a Multi-Layer Perceptron (MLP) for the task of movie review classification.

### Objectives

- Layer Definition: Define the architecture of the neural network by specifying the parameters for each fully connected layer, activation functions, and the output layer.

 - Activation Functions: Understand the role of activation functions (ReLU in this case) in introducing non-linearity to the model.

- Softmax Activation for Classification: Recognize the use of the softmax activation function in the output layer for obtaining probability scores.

- Forward Pass: Understand the forward pass through the layers of the neural network, where input data is processed to produce output log probabilities.

### Data

The dataset comprises 10,662 movie reviews, each represented as an embedding vector of size 100. The labels are binary, with the class 'negative' mapped to 0 and 'positive' mapped to 1. The necessary data preprocessing, including vectorization and label mapping, has already been performed.

### Learner Instructions

**Step 1: Define the First Fully Connected Layer**  
In the `__init()__` method, define the first fully connected layer (fc1) with an input size of 100 and an output size of 4. Then, define a ReLU activation (relu1) to introduce non-linearity.

**Step 2: Define the Second Fully Connected Layer**  
In the `__init()__` method, define the second fully connected layer (fc2) with an input size of 4 and an output size of 3. Again, define another ReLU activation (relu2).

**Step 3: Define the Third Fully Connected Layer**  
In the `__init()__` method, define the third fully connected layer (fc3) with an input size of 3 and an output size of 3. Again, a ReLU activation (relu3) follows.

**Step 4: Define the Output Layer**  
In the `__init()__` method, define the output layer (out) with an input size of 3 and an output size of 2. This layer represents the final prediction.

**Step 5: Define the Log Softmax Layer**  
Finally, in the `__init__()` method, define the log softmax layer, which is applied to the output for obtaining log probabilities.

**Step 6: Define the Forward Pass**  
In the `forward()` method, add `X` as an input parameter. Then, specifies how input `X` passes through the layers to produce output log probabilities. Note that the model should accept **input of shape (batch_size, 100)** and return **output shape of (batch_size, 2)**.

