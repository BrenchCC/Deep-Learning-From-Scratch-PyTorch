import os
import sys
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("Normalization_Layers")

class BatchNormalization(nn.module):
    """
    Implementation of Batch Normalization from scratch.
    Based on Paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    """
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1):
        super.__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (Not learnable, but part of state_dict)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # TODO: following coding is not complete, please complete it.