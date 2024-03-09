## Name Nationality Classification

### Goal

In this task, learners will practice implementing a Recurrent Neural Network (RNN) classifier without using the built-in RNN layer in PyTorch for the task of name nationality classification.

### Objectives

- Hidden state: Understand the role of the hidden state in a RNN.

- Softmax Activation for Classification: Recognize the use of the softmax activation function in the output layer for obtaining probability scores, particularly in multi-class classification problems.

- Forward Pass: Understand the forward pass through the layers of the neural network, where input data is processed to produce output log probabilities.

### Data

The features are the last names, and the class labels are nationalities. Assume the dataset consists of a list of last names with corresponding nationalities. The data processing, including converting characters to numerical values, is done for you.

### Learner Instructions

**Step 1: Define the init parameters**
Edit the initialization method (`__init__`) to take three parameters: `input_size`, `hidden_size`, and `output_size`. These parameters represent the dimensions of the input, hidden state, and output layers.

**Step 2: Define the input to hidden Layer**
In the `__init()__` method, define an input-to-hidden layer (`i2h`). This layer is essentially a linear transformation that converts the input data into the hidden state.

**Step 3: Define the hidden to hidden Layer**
In the `__init()__` method, define a hidden-to-hidden layer (`h2h`). This layer captures the recurrent nature of the network, allowing information to persist across sequential inputs.

**Step 4: Define the hidden to output Layer**
In the `__init()__` method, define a hidden-to-output layer (`h2o`). This layer processes the hidden state to produce the final output.

**Step 5: Define the Log Softmax Layer**
Finally, in the `__init__()` method, define the log softmax layer, which is applied to the output for obtaining log probabilities.

**Step 6: Define the Forward Pass**
Edit the `forward()` method to take two parameters: `inputs` (representing the current input) and `hidden` (representing the previous hidden state). Inside this method, combine the input and previous hidden state, apply the hyperbolic tangent (tanh) activation function, and pass the result through the output layer. Apply the LogSoftmax activation to obtain log probabilities. Return the output and the hidden state.

**Step 7: Instantiate the RNN Model**
Outside the class, instantiate an object of the class. Provide values for `input_size`, `hidden_size`, and `output_size` as 58, 256 and 18 respectively. This object represents your RNN model.
