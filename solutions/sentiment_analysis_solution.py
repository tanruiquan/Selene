import torch
import torch.nn as nn


class GRUTextClassifier(nn.Module):

    def __init__(self, vocab_size, embed_size, output_size, rnn_num_layers, rnn_bidirectional, rnn_hidden_size, rnn_dropout, linear_hidden_sizes, linear_dropout):
        super().__init__()

        # We have to memorize this for initializing the hidden state
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size

        # Calculate number of directions
        self.rnn_num_directions = 2 if rnn_bidirectional == True else 1

        #################################################################################
        # Create layers
        #################################################################################

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU Layer
        self.gru = nn.GRU(embed_size, rnn_hidden_size, rnn_num_layers,
                          batch_first=True, dropout=rnn_dropout, bidirectional=rnn_bidirectional)

        # Linear layers (incl. Dropout and Activation)
        linear_sizes = [rnn_hidden_size *
                        self.rnn_num_directions] + linear_hidden_sizes

        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(p=linear_dropout))

        self.out = nn.Linear(linear_sizes[-1], output_size)

        # Define log softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        #################################################################################

    def forward(self, inputs, hidden):

        batch_size, seq_len = inputs.shape

        # Push through embedding layer
        X = self.embedding(inputs)

        # Push through RNN layer
        rnn_outputs, hidden = self.gru(X, hidden)

        last_hidden = hidden.view(
            self.rnn_num_layers, self.rnn_num_directions, batch_size, self.rnn_hidden_size)[-1]

        # Handle directions
        if self.rnn_num_directions == 1:
            final_hidden = last_hidden.squeeze(0)
        elif self.rnn_num_directions == 2:
            h_1, h_2 = last_hidden[0], last_hidden[1]
            final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states

        X = final_hidden

        # Push through linear layers (incl. Dropout & Activation layers)
        for l in self.linears:
            X = l(X)

        X = self.out(X)
        log_probs = self.log_softmax(X)
        return log_probs

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_num_layers * self.rnn_num_directions, batch_size, self.rnn_hidden_size)
