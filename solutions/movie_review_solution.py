class ExpectedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define 1st fully connected (i.e., linear) hidden layer
        self.fc1 = nn.Linear(100, 4)
        self.relu1 = nn.ReLU()
        # Define 2st fully connected (i.e., linear) hidden layer
        self.fc2 = nn.Linear(4, 3)
        self.relu2 = nn.ReLU()
        # Define 3st fully connected (i.e., linear) hidden layer
        self.fc3 = nn.Linear(3, 3)
        self.relu3 = nn.ReLU()
        # Define output layer (which is also a linear layer)
        self.out = nn.Linear(3, 2)
        # Define log softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.out(out)
        log_probs = self.log_softmax(out)
        return log_probs
