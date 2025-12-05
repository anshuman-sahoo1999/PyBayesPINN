import torch
import torch.nn as nn

class BayesianNetwork(nn.Module):
    """
    A Fully Connected Neural Network with Monte Carlo Dropout.
    This architecture allows for uncertainty quantification by keeping dropout active during inference.
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[50, 50, 50, 50], dropout_rate=0.1):
        super(BayesianNetwork, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Construct hidden layers with Tanh activation (standard for Physics)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh()) 
            layers.append(nn.Dropout(p=dropout_rate)) # The source of Bayesian uncertainty
            in_dim = h_dim
            
        # Output layer (No activation, just linear regression)
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)