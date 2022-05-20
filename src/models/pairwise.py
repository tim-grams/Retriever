import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from itertools import combinations


def create_dataloader(X, y, batch_size: int = 50):
    X_relevant = X[y == 1]
    X_irrelevant = X[y == 0]

    dataset = TensorDataset(torch.tensor(pd.concat([X_relevant, X_irrelevant]).values).float(),
                            torch.tensor(pd.concat([X_irrelevant, X_relevant]).values).float(),
                            torch.tensor(y.values).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_pairwise(network, X, y, num_epochs: int = 200):

    train_loader = create_dataloader(X, y)

    optimizer = torch.optim.Adam(network.parameters())

    criterion = nn.BCELoss()

    losses = []

    batch_processed_counter = 0
    for epoch in range(num_epochs):
        for step, (X_relevant, X_irrelevant, y) in enumerate(train_loader):
            batch_processed_counter += 1

            outputs = network(X_relevant, X_irrelevant)

            y = y.reshape(-1, 1)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    "Epoch: {:2d}/{:2d}   ".format(epoch + 1, num_epochs),
                    "Batch: {:2d}  ".format(batch_processed_counter),
                    "Batch loss: {:.6f}   ".format(loss.item())
                )


def create_test_combinations(top: pd.DataFrame):
    X_relevant_test = pd.DataFrame()
    X_irrelevant_test = pd.DataFrame()
    for comb in list(combinations(range(50), 2)):
        X_relevant_test = pd.concat(
            [X_relevant_test, pd.DataFrame(top.iloc[comb[0]].to_dict(), index=[0])]).reset_index(drop=True)
        X_irrelevant_test = pd.concat(
            [X_irrelevant_test, pd.DataFrame(top.iloc[comb[1]].to_dict(), index=[0])]).reset_index(drop=True)

    return X_relevant_test, X_irrelevant_test


def bubble_sort(pairwise_results, documents):
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(documents) - 1):
            comp = pairwise_results[
                ((pairwise_results['d1'] == documents[i]) & (pairwise_results['d2'] == documents[i + 1]))]

            if len(comp) > 0:
                if comp['predictions'].values[0] >= 0.5:
                    documents[i], documents[i + 1] = documents[i + 1], documents[i]
                    swapped = True
            else:
                comp = pairwise_results[
                    ((pairwise_results['d1'] == documents[i + 1]) & (pairwise_results['d2'] == documents[i]))]
                if comp['predictions'].values[0] < 0.5:
                    documents[i], documents[i + 1] = documents[i + 1], documents[i]
                    swapped = True


def pairwise_optimize(model, results: pd.DataFrame, X, y, top_k: int = 50):
    train_pairwise(model, X, y)

    top = results.sort_values('confidence', ascending=False).head(top_k)
    X_relevant_test, X_irrelevant_test = create_test_combinations(top)

    predictions = model(torch.tensor(X_relevant_test.values).float(),
                        torch.tensor(X_irrelevant_test.values).float())

    pairwise_results = pd.DataFrame({
        'predictions': predictions.reshape(-1).detach().numpy(),
        'd1': X_relevant_test['pID'],
        'd2': X_irrelevant_test['pID']
    })
    bubble_sort(pairwise_results, list(top['pID']))

    return results
