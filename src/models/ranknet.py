import torch.nn as nn


class RankNet(nn.Module):

    def __init__(self, num_features):
        super(RankNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1))

        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        diff = s1 - s2
        prob = self.output(diff)

        return prob

    def predict_proba(self, input_):
        return self.model(input_)
