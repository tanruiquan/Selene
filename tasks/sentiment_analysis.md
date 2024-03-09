## Text Classification with GRU

### Goal

In this task, learners will implement a Gated Recurrent Unit (GRU)-based neural network for text classification. The goal is to understand the architecture of a GRU model, the role of each component, and how to define and use such a model for text classification.

### Objectives

- Model Initialization: Understand the significance of model parameters and how to initialize a GRU-based text classification model.

- GRU Architecture: Gain familiarity with the structure of a Gated Recurrent Unit (GRU) layer and its bidirectional nature.

- Linear Layers: Understand the purpose of linear layers, activation functions, and dropout in the context of text classification.

- Forward Pass: Grasp the concept of a forward pass through the layers of the neural network, including embedding layers, GRU layers, linear layers, and log softmax for obtaining log probabilities.

- Hidden State Initialization: Learn the necessity of initializing the hidden state, especially when dealing with recurrent neural networks like GRU.

### Data

The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), commonly known as the IMDb dataset, is a widely used dataset for sentiment analysis and text classification tasks. It was created by Andrew Maas and his team at Stanford University and is freely available for research purposes. 

Assume we are using the IMDb dataset and features are textual content of movie reviews and the class labels are either negative or positive. Also assume the features are vectorized, the negative and positive labels are mapped to 0 and 1 respectively. In other words, the data processing is done for you.

### Learner Instructions

#### Step 1: Define the GRUTextClassifier Class

Add the following parameters to the `__init__()` method:

- `vocab_size`: The size of the vocabulary, indicating the number of unique words in the dataset.

- `embed_size`: The size of the word embeddings. Each word in the vocabulary will be represented by a vector of this size.

- `output_size`: The number of classes or categories in the text classification task.

- `rnn_num_layers`: The number of layers in the GRU. This determines the depth of the recurrent neural network.

- `rnn_bidirectional`: A boolean indicating whether the GRU should be bidirectional or not. Bidirectional GRU processes input sequences from both forward and backward directions.

- `rnn_hidden_size`: The number of features in the hidden state of the GRU.

- `rnn_dropout`: The dropout rate applied to the input of the GRU layers. Dropout helps prevent overfitting.

- `linear_hidden_sizes`: A list of integers representing the sizes of hidden layers in the linear (fully connected) layers that follow the GRU.

- `linear_dropout`: The dropout rate applied to the input of the linear layers.

#### Step 2: GRU and Linear Layers

In the `__init__()` method:

- **Embedding Layer:** Define the embedding layer using `nn.Embedding(vocab_size, embed_size)`. This layer converts word indices into dense vectors.

- **GRU Layer:** Define the GRU layer using `nn.GRU()`. Pay attention to parameters such as `embed_size`, `rnn_hidden_size`, `rnn_num_layers`, `rnn_bidirectional`, and `rnn_dropout`. Understand the role of bidirectionality in capturing information from both directions in the input sequence.

- **Linear Layers:** Define a sequence of linear layers using `nn.ModuleList()`. Each linear layer should be followed by a ReLU activation and a dropout layer. The sizes of these layers are specified by `linear_hidden_sizes`. The last linear layer should have an output size equal to `output_size`.

- **Log Softmax Layer:** Define the log softmax layer using `nn.LogSoftmax(dim=1)`. This layer is applied to the final output for obtaining log probabilities.

#### Step 3: Forward Pass

In the `forward()` method:

- **Embedding:** Pass the input sequence through the embedding layer to convert word indices into dense vectors.

- **GRU:** Pass the embedded sequence through the GRU layer to capture sequential dependencies and obtain hidden states.

- **Hidden State Handling:** Extract the last hidden state from the GRU output, considering bidirectionality if applicable.

- **Linear Layers:** Pass the final hidden state through the sequence of linear layers, applying ReLU activations and dropout.

- **Output Layer:** Pass the result through the output layer to obtain raw scores.

- **Log Softmax:** Apply the log softmax layer to obtain log probabilities.

#### Step 4: Hidden State Initialization

In the `init_hidden()` method:

- Initialize the hidden state for the GRU. The method should return a tensor of zeros with the appropriate dimensions.

#### Step 5: Instantiate the Model

Instantiate the model with example values for the parameters:

```python
model = GRUTextClassifier(
    vocab_size=10000,
    embed_size=300,
    output_size=5,
    rnn_num_layers=2,
    rnn_bidirectional=True,
    rnn_hidden_size=128,
    rnn_dropout=0.5,
    linear_hidden_sizes=[64, 32],
    linear_dropout=0.3
)
```
