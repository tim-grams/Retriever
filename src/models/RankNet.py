import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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


def create_dataloader(X_relevant, X_irrelevant, y, batch_size: int = 50):
    dataset = TensorDataset(torch.LongTensor(X_relevant), torch.LongTensor(X_irrelevant),
                            torch.IntTensor(y))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_pairwise(
        X_relevant_train, X_irrelevant_train, y_train,
        X_relevant_val, X_irrelevant_val, y_val, num_epochs: int = 200):
    ranknet = RankNet(len(X_relevant_train.columns))

    train_loader = create_dataloader(X_relevant_train, X_irrelevant_train, y_train)
    val_loader = create_dataloader(X_relevant_val, X_irrelevant_val, y_val)

    optimizer = torch.optim.Adam(ranknet.parameters())

    criterion = nn.BCELoss()

    losses = []

    batch_processed_counter = 0
    for epoch in range(num_epochs):
        epoch_val_losses = []

        for step, (X_relevant, X_irrelevant, y) in enumerate(train_loader):
            batch_processed_counter += 1

            outputs = ranknet(X_relevant, X_irrelevant)

            y = y.reshape(-1, 1)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                val_losses = []
                ranknet.eval()
                for step, (X_relevant, X_irrelevant, y) in enumerate(val_loader):
                    outputs = ranknet(X_relevant, X_irrelevant)

                    y = y.reshape(-1, 1)
                    loss = criterion(outputs, y)

                    val_losses.append(loss.item())
                    epoch_val_losses.append(loss.item())

                    ranknet.train()
                    print(
                        "Epoch: {:2d}/{:2d}   ".format(epoch + 1, num_epochs),
                        "Batch: {:2d}  ".format(batch_processed_counter),
                        "Batch loss: {:.6f}   ".format(loss.item()),
                        "Val loss: {:.6f}".format(np.mean(val_losses)),
                    )
