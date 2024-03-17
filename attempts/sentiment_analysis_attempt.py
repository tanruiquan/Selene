class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 300)
        self.gru = nn.GRU(input_size=300, hidden_size=512,
                          num_layers=2, dropout=0.5, bidirectional=True)

        self.fc1 = nn.Linear(512*2, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.out = nn.Linear(64, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        batch_size, _ = inputs.shape

        X = self.embedding(inputs)  # (batch_size, seq_len, embed_size)
        X = X.view(seq_len, batch_size, embed_size)
        rnn_outputs, hidden = self.gru(X, hidden)

        last_hidden = hidden.view(2, 2, batch_size, 512)[-1]
        h_1, h_2 = last_hidden[0], last_hidden[1]
        final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states

        X = self.fc1(final_hidden)
        X = self.relu1(X)
        X = self.dropout1(X)

        X = self.fc2(X)
        X = self.relu2(X)
        X = self.dropout2(X)

        X = self.out(X)
        log_probs = self.log_softmax(X)
        return log_probs

    def init_hidden(self, batch_size):
        return torch.zeros(2 * 2, batch_size, 512)
