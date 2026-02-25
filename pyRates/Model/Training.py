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

        outputs = model(batch)

        # Check for dual output (Graph + Node)
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], tuple) and isinstance(outputs[1], tuple):
            out_graph, out_node = outputs

            # Graph Loss (using batch.y)
            mu_g, sigma_g, pi_g = out_graph
            loss_g = mdn_loss(mu_g, sigma_g, pi_g, batch.y)

            # Node Loss (using batch.node_y)
            mu_n, sigma_n, pi_n = out_node
            loss_n = mdn_loss(mu_n, sigma_n, pi_n, batch.node_y)

            loss = loss_g + loss_n
        else:
             # Standard Single Head
             mu, sigma, pi = outputs
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
    total_loss = total_loss_g = total_loss_n = 0
    total_nodes = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)

            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], tuple) and isinstance(outputs[1], tuple):
                out_graph, out_node = outputs
                mu_g, sigma_g, pi_g = out_graph
                loss_g = mdn_loss(mu_g, sigma_g, pi_g, batch.y)
                mu_n, sigma_n, pi_n = out_node
                loss_n = mdn_loss(mu_n, sigma_n, pi_n, batch.node_y)

                total_loss_g += loss_g.item() * batch.num_graphs
                total_loss_n += loss_n.item() * batch.num_nodes
                total_nodes += batch.num_nodes
                total_loss += (loss_g.item() + loss_n.item()) * batch.num_graphs
            else:
                mu, sigma, pi = outputs
                total_loss = mdn_loss(mu, sigma, pi, batch.y)

    n_graphs = len(loader.dataset)
    avg_loss_g = total_loss_g / n_graphs
    avg_loss_n = total_loss_n / total_nodes if total_nodes > 0 else 0
    avg_loss = total_loss / n_graphs

    return avg_loss, avg_loss_g, avg_loss_n


