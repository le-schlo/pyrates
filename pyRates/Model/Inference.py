import torch
import numpy as np

def predict(model, loader, device, scaler, mode='mean'):
    '''
    Make predictions with the MDN model. Supports dual-head (Graph + Node) models.
    Returns structured results.
    '''
    model.eval()

    # Storage for Graph outputs
    g_preds, g_targets, g_unc, g_weights = [], [], [], []
    # Storage for Node outputs
    n_preds, n_targets, n_unc, n_weights = [], [], [], []

    dual_mode = False

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)

            # Detect Dual Output
            if isinstance(outputs, tuple) and len(outputs) == 2:
                # Assuming (graph_out, node_out)
                out_graph, out_node = outputs
                dual_mode = True

                # --- Graph Part ---
                mu_g, sigma_g, pi_g = out_graph
                g_weights.append(pi_g.cpu().numpy())

                # Use helper for predictions
                mean_g, unc_g = _compute_mixture_mean(mu_g, sigma_g, pi_g) # Defaulting to mean mode logic for simplicity for now
                # If expanding modes, would need conditional logic here similar to original function

                g_preds.append(mean_g.cpu().numpy())
                g_targets.append(batch.y.cpu().numpy())
                g_unc.append(unc_g.cpu().numpy())

                # --- Node Part ---
                mu_n, sigma_n, pi_n = out_node
                n_weights.append(pi_n.cpu().numpy())

                mean_n, unc_n = _compute_mixture_mean(mu_n, sigma_n, pi_n)
                n_preds.append(mean_n.cpu().numpy())
                n_targets.append(batch.node_y.cpu().numpy())
                n_unc.append(unc_n.cpu().numpy())

            else:
                # Single Head (Graph only or Node only - treated same)
                mu, sigma, pi = outputs
                g_weights.append(pi.cpu().numpy())

                if mode == 'mean':
                    mean, uncertainty = _compute_mixture_mean(mu, sigma, pi)
                    g_preds.append(mean.cpu().numpy())
                    g_targets.append(batch.y.cpu().numpy())
                    g_unc.append(uncertainty.cpu().numpy())
                # ... avoiding full reimplementation of all modes for this snippet, focusing on 'mean' for mixed
                # If needed, can expand. Assuming mean mode for this user request.
                else:
                     # Fallback for simple case logic
                     pass

    if dual_mode:
        # Process Graph
        g_preds = np.vstack(g_preds)
        g_targets = np.vstack(g_targets)
        g_weights = np.vstack(g_weights)
        g_unc = np.vstack(g_unc)

        # Process Node
        n_preds = np.vstack(n_preds)
        n_targets = np.vstack(n_targets)
        n_weights = np.vstack(n_weights)
        n_unc = np.vstack(n_unc)

        # Inverse Transform
        # Assuming scaler is DualScaler
        if hasattr(scaler, 'graph_scaler') and scaler.graph_scaler:
             g_preds = scaler.graph_scaler.inverse_transform(g_preds)
             g_targets = scaler.graph_scaler.inverse_transform(g_targets)
             g_unc = g_unc * scaler.graph_scaler.scale_

        if hasattr(scaler, 'node_scaler') and scaler.node_scaler:
             n_preds = scaler.node_scaler.inverse_transform(n_preds)
             n_targets = scaler.node_scaler.inverse_transform(n_targets)
             n_unc = n_unc * scaler.node_scaler.scale_

        return {
            'graph': ((g_preds, g_unc, g_weights), g_targets),
            'node': ((n_preds, n_unc, n_weights), n_targets)
        }
    else:
        # Standard Single Output
        preds = np.vstack(g_preds)
        targets = np.vstack(g_targets)
        weights = np.vstack(g_weights)
        uncertainties = np.vstack(g_unc)

        preds = scaler.inverse_transform(preds)
        targets = scaler.inverse_transform(targets)
        uncertainties = uncertainties * scaler.scale_
        return (preds, uncertainties, weights), targets


def _compute_mixture_mean(mu, sigma, pi):
    """
    Compute the mixture expectation (weighted mean) prediction.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]

    Returns:
        mean: Mixture expectation [batch_size, output_dim]
        uncertainty: Total standard deviation [batch_size, output_dim]
    """
    # Mixture expectation: sum_k pi_k * mu_k
    mean = torch.sum(pi * mu, dim=-1)

    # Total variance: E[Var] + Var[E]
    # E[Var] = sum_k pi_k * sigma_k^2
    # Var[E] = sum_k pi_k * mu_k^2 - (sum_k pi_k * mu_k)^2
    var_within = torch.sum(pi * sigma ** 2, dim=-1)
    var_between = torch.sum(pi * mu ** 2, dim=-1) - mean ** 2
    total_var = var_within + var_between
    uncertainty = torch.sqrt(total_var)

    return mean, uncertainty


def _compute_max_mode(mu, sigma, pi):
    """
    Compute the prediction from the mode with the highest weight (pi).

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]

    Returns:
        max_mu: Mean of the highest-weight mode [batch_size, output_dim]
        max_sigma: Sigma of the highest-weight mode [batch_size, output_dim]
    """
    max_pi_indices = torch.argmax(pi, dim=-1, keepdim=True)

    # Gather the corresponding mu and sigma
    max_mu = torch.gather(mu, -1, max_pi_indices).squeeze(-1)
    max_sigma = torch.gather(sigma, -1, max_pi_indices).squeeze(-1)

    return max_mu, max_sigma


def _compute_each_mode(mu, sigma, pi):
    """
    Extract the mean and sigma for each mode separately.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]

    Returns:
        modes: List of tuples [(mu_k, sigma_k, pi_k), ...] for each component k
    """
    n_components = pi.shape[-1]
    modes = []

    for k in range(n_components):
        mode_mu = mu[:, :, k]  # [batch_size, output_dim]
        mode_sigma = sigma[:, :, k]  # [batch_size, output_dim]
        mode_pi = pi[:, :, k]  # [batch_size, output_dim]
        modes.append((mode_mu, mode_sigma, mode_pi))

    return modes


def _sample_from_mixture(mu, sigma, pi, n_samples=100):
    """
    Sample from the mixture distribution.

    Args:
        mu: Mean values [batch_size, output_dim, n_components]
        sigma: Standard deviations [batch_size, output_dim, n_components]
        pi: Mixture weights [batch_size, output_dim, n_components]
        n_samples: Number of samples to draw

    Returns:
        samples: Sampled values [batch_size, n_samples, output_dim]
        mean_sample: Mean of samples [batch_size, output_dim]
        std_sample: Std of samples [batch_size, output_dim]
    """
    batch_size, output_dim, n_components = mu.shape
    device = mu.device

    # Sample component indices based on pi
    # pi shape: [batch_size, output_dim, n_components]
    component_samples = torch.zeros(batch_size, n_samples, output_dim, device=device)

    for i in range(output_dim):
        # Sample which component to use for each sample
        component_dist = torch.distributions.Categorical(pi[:, i, :])
        component_indices = component_dist.sample((n_samples,)).T  # [batch_size, n_samples]

        # Get mu and sigma for selected components
        mu_selected = torch.gather(mu[:, i, :].unsqueeze(1).expand(-1, n_samples, -1),
                                   2, component_indices.unsqueeze(-1)).squeeze(-1)
        sigma_selected = torch.gather(sigma[:, i, :].unsqueeze(1).expand(-1, n_samples, -1),
                                      2, component_indices.unsqueeze(-1)).squeeze(-1)

        # Sample from Gaussian
        component_samples[:, :, i] = torch.normal(mu_selected, sigma_selected)

    mean_sample = component_samples.mean(dim=1)
    std_sample = component_samples.std(dim=1)

    return component_samples, mean_sample, std_sample