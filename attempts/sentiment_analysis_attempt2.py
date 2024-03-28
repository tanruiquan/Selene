class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
        self.gru = nn.GRU(input_size=300, hidden_size=512,
                          num_layers=2, dropout=0.5)

        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.out = nn.Linear(in_features=64, out_features=2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        batch_size, seq_len = inputs.shape

        X = self.embedding(inputs)
        X = X.view(seq_len, batch_size, 300)
        _, hidden = self.gru(X, hidden)
        final_hidden = hidden[0]

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
        return torch.zeros(2, batch_size, 512)
