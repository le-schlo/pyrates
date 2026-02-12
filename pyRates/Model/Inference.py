import torch
import numpy as np

def predict(model, loader, device, scaler, mode='mean'):
    '''
    Make predictions with the MDN model.

    Args:
        model: The MDN model
        loader: DataLoader with the data
        device: Device to run on
        scaler: Scaler for inverse transformation
        mode: Prediction mode ('mean', 'sample', 'each_mode', 'max_mode')

    Returns:
        Tuple of (predictions_tuple, targets) where predictions_tuple contains
        the predictions, optionally uncertainties, and mode weights.
    '''
    model.eval()
    preds = []
    targets = []
    uncertainties = []
    mode_weights = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            mu, sigma, pi = model(batch)
            mode_weights.append(pi.cpu().numpy())

            if mode == 'mean':
                mean, uncertainty = _compute_mixture_mean(mu, sigma, pi)
                preds.append(mean.cpu().numpy())
                targets.append(batch.y.cpu().numpy())
                uncertainties.append(uncertainty.cpu().numpy())

            elif mode == 'sample':
                _, mean_sample, std_sample = _sample_from_mixture(mu, sigma, pi, n_samples=100)
                preds.append(mean_sample.cpu().numpy())
                targets.append(batch.y.cpu().numpy())
                uncertainties.append(std_sample.cpu().numpy())

            elif mode == 'each_mode':
                modes = _compute_each_mode(mu, sigma, pi)
                for mode_mu, mode_sigma, _ in modes:
                    preds.append(mode_mu.cpu().numpy())
                    targets.append(batch.y.cpu().numpy())
                    uncertainties.append(mode_sigma.cpu().numpy())

            elif mode == 'max_mode':
                max_mu, max_sigma = _compute_max_mode(mu, sigma, pi)
                preds.append(max_mu.cpu().numpy())
                targets.append(batch.y.cpu().numpy())
                uncertainties.append(max_sigma.cpu().numpy())

            else:
                raise ValueError(f"Invalid mode: {mode}. Choose from 'mean', 'sample', 'each_mode', or 'max_mode'.")


    preds = np.vstack(preds)
    targets = np.vstack(targets)
    mode_weights = np.vstack(mode_weights)

    preds = scaler.inverse_transform(preds)
    targets = scaler.inverse_transform(targets)

    uncertainties = np.vstack(uncertainties)
    # Scale uncertainties appropriately (multiply by scale factor)
    uncertainties = uncertainties * scaler.scale_
    return (preds, uncertainties, mode_weights), targets


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