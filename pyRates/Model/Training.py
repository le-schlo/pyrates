import torch
import numpy as np
from pyRates.Model.Loss import mdn_loss

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        mu, sigma, pi = model(batch)
        loss = mdn_loss(mu, sigma, pi, batch.y)

        if torch.isnan(loss):
            raise ValueError("NaN loss detected!")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, device):
    """Validate and return average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            mu, sigma, pi = model(batch)
            loss = mdn_loss(mu, sigma, pi, batch.y)

            total_loss += loss.item()

    return total_loss / len(loader)


