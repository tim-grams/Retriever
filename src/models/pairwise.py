import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from itertools import combinations


def create_dataloader(X, y, batch_size: int = 50) -> DataLoader:
    """ .

    Args:
        X ():
        y ():
        batch_size (int):

    Returns:
        :

    """
    X_relevant = X[y.reset_index(drop=True) == 1]
    X_irrelevant = X[y.reset_index(drop=True) == 0]

    dataset = TensorDataset(torch.tensor(pd.concat([X_relevant, X_irrelevant]).values).float(),
                            torch.tensor(pd.concat([X_irrelevant, X_relevant]).values).float(),
                            torch.tensor(y.values).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_pairwise(network, X, y, num_epochs: int = 20):
    """ .

    Args:
        network ()
        X (int):
        y (int):
        num_epochs (int):

    Returns:
        :

        """

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


def create_test_combinations(top: pd.DataFrame, k: int = 50) -> tuple:
    """ Creates test combinations.

    Args:
        top (pd.DataFrame)
        k (int):

    Returns:
        X_relevant_test ():
        X_irrelevant_test ():

    """
    X_relevant_test = pd.DataFrame()
    X_irrelevant_test = pd.DataFrame()
    for comb in list(combinations(range(k), 2)):
        X_relevant_test = pd.concat(
            [X_relevant_test, pd.DataFrame(top.iloc[comb[0]].to_dict(), index=[0])]).reset_index(drop=True)
        X_irrelevant_test = pd.concat(
            [X_irrelevant_test, pd.DataFrame(top.iloc[comb[1]].to_dict(), index=[0])]).reset_index(drop=True)

    return X_relevant_test, X_irrelevant_test


def bubble_sort(pairwise_results, documents) -> list:
    """ .

    Args:
        pairwise_results (list)
        documents (list):

    Returns:
        documents (list):

    """
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(documents) - 1):
            comp = pairwise_results[
                ((pairwise_results['d1'] == documents[i]) & (pairwise_results['d2'] == documents[i + 1]))]

            if len(comp) > 0:
                if comp['predictions'].values[0] < 0.5:
                    documents[i], documents[i + 1] = documents[i + 1], documents[i]
                    swapped = True
            else:
                comp = pairwise_results[
                    ((pairwise_results['d1'] == documents[i + 1]) & (pairwise_results['d2'] == documents[i]))]
                if comp['predictions'].values[0] >= 0.5:
                    documents[i], documents[i + 1] = documents[i + 1], documents[i]
                    swapped = True

    return documents


def pairwise_optimize(model, results: pd.DataFrame, X, y, X_test, top_k: int = 50, train: bool = True) -> pd.DataFrame:
    """ .

    Args:
        model ():
        results (pd.DataFrame):
        X ():
        y (pd.DataFrame):
        X_test ():
        top_k (int):
        train (Boolean):


    Returns:
        results (pd.DataFrame):

    """
    if train:
        train_pairwise(model, X, y)

    top = pd.concat([results, X_test], axis=1).sort_values('confidence', ascending=False).head(top_k)
    X_test = top.drop(columns=['confidence', 'qID', 'relevant'])
    top = results.sort_values('confidence', ascending=False).head(top_k)

    X_relevant_test, X_irrelevant_test = create_test_combinations(X_test, top_k)

    predictions = model(torch.tensor(X_relevant_test.drop(columns=['pID']).values).float(),
                        torch.tensor(X_irrelevant_test.drop(columns=['pID']).values).float())

    pairwise_results = pd.DataFrame({
        'predictions': predictions.reshape(-1).detach().numpy(),
        'd1': X_relevant_test['pID'],
        'd2': X_irrelevant_test['pID']
    })

    result = pd.DataFrame()
    for document in bubble_sort(pairwise_results, list(top['pID'])):
        result = pd.concat([result, top[top['pID'] == document]])

    results.iloc[:top_k] = result

    return results
