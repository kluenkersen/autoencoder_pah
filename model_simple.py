import torch
from torch import nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoencoder, self).__init__()
        
        layers = []
        layers += [nn.Linear(input_dim, 200)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(200, 100)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(100, 50)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(50, 10)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(10, 5)]
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Linear(5, 10)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(10, 50)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(50, 100)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(100, 200)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(200, input_dim)]
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
