import torch


class Model(torch.nn.Module):
    def __init__(self, in_deg, dropout_rate=0.4):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(in_deg, 250)
        self.linear2 = torch.nn.Linear(250, 300)
        self.linear3 = torch.nn.Linear(300, 150)
        self.linear4 = torch.nn.Linear(150, 1)

        self.activation = torch.nn.Sigmoid()

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear4(x)
        x = x.squeeze(1)
        x = self.sigmoid(x)
        return x
