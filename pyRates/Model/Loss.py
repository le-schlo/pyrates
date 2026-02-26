import torch

def mdn_loss(mu, sigma, pi, target):
    target = target.unsqueeze(-1)

    log_prob = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        ((target - mu)**2) / (sigma**2)
    )

    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_prob + log_pi
    log_sum = torch.logsumexp(log_prob_weighted, dim=-1)

    return -log_sum.mean()

def mdn_loss_per_sample(mu, sigma, pi, target):
    target = target.unsqueeze(-1)

    log_prob = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        ((target - mu)**2) / (sigma**2)
    )

    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_prob + log_pi
    log_sum = torch.logsumexp(log_prob_weighted, dim=-1)

    return -log_sum