## Movie Review Sentiment Classification with Bi-GRU

### Goal

In this task, learners will implement a sentiment classification model using a Bidirectional Gated Recurrent Unit (Bi-GRU) architecture for movie review sentiment analysis.

### Objectives

- Neural Network Architecture: Understand the architecture of a Bidirectional GRU-based neural network for sequential data processing.

- Bidirectional GRU: Comprehend the functionality of Bidirectional GRU layers in capturing both past and future contexts in sequential data.

- Dropout Regularization: Recognize the use of dropout regularization to prevent overfitting in neural networks.

- Softmax Activation for Classification: Understand the application of softmax activation in the output layer for multi-class classification tasks.

### Data
The dataset consists of 10,662 movie reviews. Each review is represented as a sequence of vocabulary indices, where each index corresponds to a specific word in the vocabulary. Assume each review is exactly 100 words long and the vocabulary size is 10000. The sentiment labels are binary, with 'negative' mapped to 0 and 'positive' mapped to 1. 

### Learner Instructions

**Step 1: Define the Embedding Layer**  
In the `__init__()` method, define the embedding layer to convert input word indices into dense word embeddings. Use an embedding matrix with vocabulary size 10000 and embedding dimensionality 300.

**Step 2: Define the Bidirectional GRU Layer**  
In the `__init__()` method, define a Bidirectional GRU layer with an input size of 300, hidden size of 512, and 2 stacked layers. Apply dropout regularization with a rate of 0.5 to prevent overfitting.

**Step 3: Define the First Fully Connected Layer**  
Define the first fully connected layer (fc1) with an input size of 1024 (512 * 2, considering bidirectionality) and an output size of 128. Apply ReLU activation and dropout regularization with a rate of 0.5.

**Step 5: Define the Second Fully Connected Layer**  
Define the second fully connected layer (fc2) with an input size of 128 and an output size of 64. Apply ReLU activation and dropout regularization with a rate of 0.5.

**Step 6: Define the Output Layer**  
Define the output layer (out) with an input size of 64 and an output size of 2, representing the two sentiment classes.

**Step 7: Define the Log Softmax Layer**  
Define the log softmax layer, applied to the output for obtaining log probabilities over the sentiment classes.

**Step 8: Define the Forward Pass**  
In the `forward()` method, add `input` and `hidden` as the method's parameters. The method should accept **input tensor of shape (batch_size, 100)**, **hidden tensor of shape (2 * 2, batch_size, 512)** and return **output of shape (batch_size, 2)**.

**Step 9: Initialize Hidden State**  
Implement the `init_hidden()` method to initialize the hidden state of the Bi-GRU layer. The method should return a tensor of zeros with the shape (2 * 2, batch_size, 512).