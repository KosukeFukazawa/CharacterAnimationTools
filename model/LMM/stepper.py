from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class Stepper(nn.Module):
    """A simple MLP network."""
    output_size: int
    hidden_size: int = 512
    
    @nn.compact
    def __call__(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        for _ in range(3):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.elu(x)
        x = nn.Dense(self.output_size)(x)
        
        return x.reshape([nbatch, nwindow, -1])