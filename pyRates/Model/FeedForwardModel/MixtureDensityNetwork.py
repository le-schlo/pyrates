import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):

    def __init__(self, input_dim, output_dim, n_components=3, dropout=0.1, hidden_dims=[256, 128], epsilon=1e-6):
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components
        self.epsilon = epsilon

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(prev_dim, output_dim * n_components)
        self.sigma_layer = nn.Linear(prev_dim, output_dim * n_components)
        self.pi_layer = nn.Linear(prev_dim, output_dim * n_components)

    def forward(self, x):
        h = self.hidden_layers(x)

        mu = self.mu_layer(h).view(-1, self.output_dim, self.n_components)

        log_sigma = self.sigma_layer(h).view(-1, self.output_dim, self.n_components)
        sigma = F.softplus(log_sigma) + self.epsilon

        pi_logits = self.pi_layer(h).view(-1, self.output_dim, self.n_components)
        pi = F.softmax(pi_logits, dim=-1)

        return mu, sigma, pi