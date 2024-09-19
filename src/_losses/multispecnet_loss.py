import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .spectralnet_loss import SpectralNetLoss
from _utils import get_laplacian, get_normalized_laplacian


class MultiSpectralNetLoss(nn.Module):
    def __init__(self):
        super(MultiSpectralNetLoss, self).__init__()

    def forward(
        self, Ws: list, Y: torch.Tensor, weights: torch.Tensor, is_normalized: bool = False,
    ) -> torch.Tensor:
        """
        This function computes the loss of the MultiSpectralNet model.
        The loss is the sum of the rayleigh quotient of the Laplacian matrix obtained from each W,
        and the orthonormalized output of the network.

        Args:
            Ws (list):                             Affinity matrices
            Y (torch.Tensor):                      Outputs of the network
            is_normalized (bool, optional):        Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        num_of_views = len(Ws)
        m = Y.shape[0]
        loss = 0
        for i in range(num_of_views):
            loss += weights[i] * SpectralNetLoss()(Ws[i], Y, is_normalized)

        return loss / (m * num_of_views)


# class MultiSpectralNetLoss(nn.Module):
#     def __init__(self):
#         super(MultiSpectralNetLoss, self).__init__()

#     def forward(
#         self, Ws: list, Y: torch.Tensor, is_normalized: bool = False
#     ) -> torch.Tensor:
#         """
#         This function computes the loss of the MultiSpectralNet model.
#         The loss is the sum of the rayleigh quotient of the Laplacian matrix obtained from each W,
#         and the orthonormalized output of the network.

#         Args:
#             Ws (list):                             Affinity matrices
#             Y (torch.Tensor):                      Outputs of the network
#             is_normalized (bool, optional):        Whether to use the normalized Laplacian matrix or not.

#         Returns:
#             torch.Tensor: The loss
#         """
#         num_of_views = len(Ws)
#         loss = 0
#         Ls = []

#         for i in range(num_of_views):
#             L_i = torch.from_numpy(get_laplacian(Ws[i]))
#             Ls.append(L_i.T @ L_i)

#         L_avg = sum(Ls)
#         # L_avg = Ls[0]
#         # for matrix in Ls[1:]:
#         #     L_avg = torch.mm(L_avg, matrix)
#         loss = torch.trace(Y.T @ L_avg @ Y) / (num_of_views * Ws[0].shape[0] ** 2)
#         return loss
