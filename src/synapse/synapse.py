import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)

class Synapse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)

    def process(self, inputs):
        logging.info(f"Processing inputs: {inputs}")
        logging.info(f"Specialization state: {self.specialization_state}")
        return self.forward(inputs)