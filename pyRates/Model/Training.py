import torch
from torch_scatter import scatter_mean
from pyRates.Model.Loss import mdn_loss, mdn_loss_per_sample

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)

        loss_dict = compute_loss(model, outputs, batch)
        loss = loss_dict["loss"]

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
    totals = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            loss_dict = compute_loss(model, outputs, batch)
            n = batch.num_graphs
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item() * n

    n_graphs = len(loader.dataset)
    return {key: total / n_graphs for key, total in totals.items()}

def compute_loss(model, outputs, batch):

    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], tuple) and isinstance(outputs[1], tuple):
        out_graph, out_node = outputs

        # Graph Loss (using batch.y)
        mu_g, sigma_g, pi_g = out_graph
        loss_g = mdn_loss_per_sample(mu_g, sigma_g, pi_g, batch.y).mean()  # Mean over graph samples

        # Node Loss (using batch.node_y) returning mean over nodes per graph, then mean over graphs
        mu_n, sigma_n, pi_n = out_node
        node_losses = mdn_loss_per_sample(mu_n, sigma_n, pi_n, batch.node_y)
        loss_n = scatter_mean(node_losses, batch.batch, dim=0).mean()

        #  weighting
        weight_g = model.weight_graph_loss
        weight_n = 1 - model.weight_graph_loss

        loss = (weight_g * loss_g +
                weight_n * loss_n)

        return {"loss": loss, "loss_g": loss_g, "loss_n": loss_n}

    else:
        # No weighting needed if only one type of output is present, just compute the corresponding loss
        if model.num_graph_tasks > 0 and model.num_node_tasks == 0:
            mu, sigma, pi = outputs
            loss = mdn_loss(mu, sigma, pi, batch.y)  # equivalent to mdn_loss_per_sample(...).mean() for graph-level loss
            return {"loss": loss}

        elif model.num_node_tasks > 0 and model.num_graph_tasks == 0:
            mu, sigma, pi = outputs
            node_losses = mdn_loss_per_sample(mu, sigma, pi, batch.node_y)
            loss = scatter_mean(node_losses, batch.batch, dim=0).mean()  # mean over nodes per graph, then mean over graphs
            return {"loss": loss}

        else:
            raise ValueError("Model output format not recognized. Expected either graph-level or node-level outputs, or a tuple of both.")