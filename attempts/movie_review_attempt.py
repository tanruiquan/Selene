class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4, out_features=3)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(in_features=3, out_features=2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.out(out)
        log_probs = self.log_softmax(out)
        return log_probs
