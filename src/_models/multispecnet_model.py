import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP

from .spectralnet_model import SpectralNetModel


class MultiSpectralNetModel(nn.Module):
    def __init__(self, architectures: list, input_dims: list, output_dim: int, softmax_temp: int):
        super(MultiSpectralNetModel, self).__init__()
        self.architectures = architectures
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.softmax_temp = softmax_temp
        self.num_of_views = len(self.architectures)
        self.models = nn.ModuleList()

        for i in range(self.num_of_views):
            model = SpectralNetModel(self.architectures[i], self.input_dims[i])
            self.models.append(model)


        self.hnet = self._get_hnet()


    def _get_hnet(self) -> HMLP:
        uncond_in_size = sum(self.input_dims)
        mnet = MLP(n_in=self.num_of_views, use_bias=False, n_out=1, hidden_layers=[], no_weights=True)
        hnet = HMLP(mnet.param_shapes, uncond_in_size=uncond_in_size,  num_cond_embs=0, cond_in_size=0, layers=[100, 100, 100])

        return hnet


    def _get_weights_from_hnet(self, views):
        X_concat = torch.cat(views, dim=1)
        weights  = self.hnet(uncond_input=X_concat, ret_format='flattened')

        temperture = self.softmax_temp
        weights = nn.functional.softmax(weights / temperture, dim=1)
        return weights


    def _make_fusion(self, Ys: list, views) -> torch.Tensor:
        """
        This function performs the fusion of the outputs of the different
        spectral nets.

        Args:
            Ys (list): The outputs of the different spectral nets

        Returns:
            torch.Tensor: The fused output
        """

        weights = self._get_weights_from_hnet(views)
        Y_fused = torch.zeros_like(Ys[0])
        for i in range(self.num_of_views):
            weight_column = weights[:, i]
            Y_fused += weight_column[:, None] * Ys[i] 

        weights = torch.mean(weights, dim=0)
        return Y_fused, weights

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies the Cholesky decomposition to orthonormalize the output (`Y`) of the network.
        The orthonormalized output is returned as a tensor.
        """

        m = Y.shape[0]

        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(self, views: list, is_orthonorm: bool = True) -> list:
        """
        This function performs the forward pass of the model.
        If is_orthonorm is True, the output of the network is orthonormalized
        using the Cholesky decomposition.

        Args:
            views (list):                   The input tensors
            is_orthonorm (bool, optional):  Whether to orthonormalize the output or not.
                                            Defaults to True.

        Returns:
            list: The output tensors
        """

        Ys = []
        for i in range(self.num_of_views):
            Y = self.models[i](views[i], is_orthonorm)
            Ys.append(Y)

        Y, weights = self._make_fusion(Ys, views)

        Y_tilde = Y
        if is_orthonorm:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = torch.mm(Y_tilde, self.orthonorm_weights)

        
        return Y, weights 