import numpy as np
import pandas as pd
import torch


def train_network(model, train_loader, device) -> float:
    """
    Performs one epoch of model training

    Args:
        model: pytorch model
        train_loader: pytorch DataLoader
        device: str - 'cuda' or 'cpu'

    Returns:
        train_loss: float - training loss
    """

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        model.optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim=1)))
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)


def eval_network(model, loader, device) -> float:
    """
    Evaluates the model on a dataset

    Args:
        model: pytorch model
        loader: pytorch DataLoader
        device: str - 'cuda' or 'cpu'

    Returns:
        loss: float - evaluation loss
    """

    model.eval()
    loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss += (
            torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim=1))).item()
            * batch.num_graphs
        )
    return loss / len(loader.dataset)


def predict_network(model, loader, return_emb=False) -> tuple:
    """
    Predicts the output of the model on a dataset

    Args:
        model: pytorch model
        loader: pytorch DataLoader
        return_emb: bool - whether to return the embeddings

    Returns:
        y_pred: np.array - predicted values
        y_true: np.array - true values
        idx: np.array - indices
        embeddings: pd.DataFrame - embeddings
    """

    model.to("cpu")
    model.eval()

    y_pred, y_true, idx, embeddings = [], [], [], []

    for batch in loader:
        batch = batch.to("cpu")
        out, emb = model(
            batch.x, batch.edge_index, batch.batch, return_graph_embedding=True
        )

        y_pred.append(out.cpu().detach().numpy())
        y_true.append(batch.y.cpu().detach().numpy())
        idx.append(batch.idx.cpu().detach().numpy())
        embeddings.append(emb.detach().numpy())

    y_pred = np.concatenate(y_pred, axis=0).ravel()
    y_true = np.concatenate(y_true, axis=0).ravel()
    idx = np.concatenate(idx, axis=0).ravel()
    embeddings = np.concatenate(embeddings, axis=0)

    embeddings = pd.DataFrame(embeddings)

    embeddings["ddG_exp"] = y_true
    embeddings["ddG_pred"] = y_pred
    embeddings["index"] = idx

    if return_emb == True:
        return (y_pred, y_true, idx, embeddings)
    else:
        return (y_pred, y_true, idx)
