import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np

class ANet(nn.Module):

    def __init__(self, logger, args, num_users, num_items):
        super(ANet, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt):
        lstAsp = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)
        itemAspRep = torch.transpose(itemAspRep, 0, 1)

        for k in range(self.args.num_aspects):
            # Calculate &beta_{u,a_i} · p_{u,a_i} for userANetLF
            userAspImpt_k = torch.unsqueeze(userAspImpt[:, k], 1)  # Shape: (bsz, 1)
            userAspRep_k = userAspRep[k]  # Shape: (bsz, h1)
            userANetLF_k = userAspImpt_k * userAspRep_k  # Element-wise multiplication

            # Calculate &beta_{i,a_i} · p_{i,a_i} for itemANetLF
            itemAspImpt_k = torch.unsqueeze(itemAspImpt[:, k], 1)  # Shape: (bsz, 1)
            itemAspRep_k = itemAspRep[k]  # Shape: (bsz, h1)
            itemANetLF_k = itemAspImpt_k * itemAspRep_k  # Element-wise multiplication

            lstAsp.append((userANetLF_k, itemANetLF_k))

        # Sum the vectors over all aspects
        userANetLF = torch.sum(torch.stack([user for user, item in lstAsp], dim=1), dim=1)  # Shape: (bsz, h1)
        itemANetLF = torch.sum(torch.stack([item for user, item in lstAsp], dim=1), dim=1)  # Shape: (bsz, h1)

        return userANetLF, itemANetLF
