import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from _utils import *
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset

from _models.multispecnet_model import MultiSpectralNetModel
from _losses.multispecnet_loss import MultiSpectralNetLoss


class MultiViewDataset(Dataset):
    def __init__(self, views: list, labels: torch.Tensor):
        self.views = views
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        views = []
        for i in range(len(self.views)):
            views.append(self.views[i][index])
        return views, self.labels[index]


class MultiSpectralTrainer:
    def __init__(self, config: dict, device: torch.device):
        """
        This class is responsible for training the SpectralNet model.

        Args:
            config (dict):                  The configuration dictionary
            device (torch.device):          The device to use for training
            is_sparse (bool, optional):     Whether the graph-laplacian obtained from a mini-batch is sparse or not.
                                            In case it is sparse, we build the batch by taking 1/5 of the original random batch,
                                            and then we add to each sample 4 of its nearest neighbors. Defaults to False.
        """

        self.device = device
        self.n_clusters = config["n_clusters"]
        self.spectral_config = config["spectral"]

        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.temperture = self.spectral_config["temperture"]
        self.architectures = self.spectral_config["architectures"]

    def train(
        self, views: list, labels: torch.Tensor = None, siamese_nets: list = None
    ) -> MultiSpectralNetModel:
        """
        This function trains the SpectralNet model.

        Args:
            X (torch.Tensor):                       The dataset to train on
            y (torch.Tensor):                       The labels of the dataset in case there are any
            siamese_net (nn.Module, optional):      The siamese network to use for computing the affinity matrix.

        Returns:
            SpectralNetModel: The trained SpectralNet model
        """

        self.counter = 0
        self.views = views
        self.labels = labels
        self.siamese_nets = siamese_nets
        self.criterion = MultiSpectralNetLoss()
        self.input_dims = [view.shape[1] for view in self.views]
        self.multi_spectral_net = MultiSpectralNetModel(
            self.architectures, input_dims=self.input_dims, output_dim=self.n_clusters, softmax_temp=self.temperture
        ).to(self.device)
        self.optimizer = optim.Adam(self.multi_spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader = self._get_data_loader()
        print("Training MultiSpectralNet:")
        for epoch in range(self.epochs):
            train_loss = 0.0
            for (X_grad, y_grad), (X_orth, _) in zip(train_loader, ortho_loader):
                for i in range(len(X_grad)):
                    X_orth[i] = X_orth[i].to(device=self.device)
                    X_grad[i] = X_grad[i].to(device=self.device)

                # Orthogonalization step
                self.multi_spectral_net.eval()
                self.multi_spectral_net(X_orth, is_orthonorm=True)

                # Gradient step
                self.multi_spectral_net.train()
                self.optimizer.zero_grad()

                Y, weights = self.multi_spectral_net(X_grad, is_orthonorm=False)
                if len(self.siamese_nets) > 0:
                    with torch.no_grad():
                        for i in range(len(X_grad)):
                            X_grad[i] = self.siamese_nets[i].forward_once(X_grad[i])

                Ws = []
                for i in range(len(X_grad)):
                    Ws.append(self._get_affinity_matrix(X_grad[i]))

                loss = self.criterion(Ws, Y, weights)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            print(
                "Epoch: {}/{}, Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    epoch + 1, self.epochs, train_loss, valid_loss, current_lr
                )
            ) 
            print(weights)
        
        
        return self.multi_spectral_net

    def validate(self, valid_loader: DataLoader) -> float:
        """
        This function validates the SpectralNet model during the training process.

        Args:
            valid_loader (DataLoader):  The validation data loader

        Returns:
            float: The validation loss
        """

        valid_loss = 0.0
        self.multi_spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch

                for i in range(len(X)):
                    X[i] = X[i].to(device=self.device)
                y = y.to(self.device)

                Y, weights = self.multi_spectral_net(X, is_orthonorm=True)
                if len(self.siamese_nets) > 0:
                    for i in range(len(X)):
                        X[i] = self.siamese_nets[i].forward_once(X[i])

                Ws = []
                for i in range(len(X)):
                    Ws.append(self._get_affinity_matrix(X[i]))

                loss = self.criterion(Ws, Y, weights)
                valid_loss += loss.item()

        if self.counter == 0:
            for i, W in enumerate(Ws):
                plot_sorted_laplacian(W, y,i)
                # plot_diagonalization(W, Y)

        # plot_laplacian_eigenvectors(Y, y)
        self.counter += 1

        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.spectral_config["is_local_scale"]
        n_neighbors = self.spectral_config["n_neighbors"]
        scale_k = self.spectral_config["scale_k"]
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        # W = get_t_kernel(Dx, indices, device=self.device, is_local=is_local)
        return W

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.labels is None:
            self.labels = torch.zeros(len(self.views[0]))

        train_size = int(0.9 * len(self.views[0]))
        valid_size = len(self.views[0]) - train_size
        dataset = MultiViewDataset(self.views, self.labels)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )

        # train_size = int(0.9 * len(self.views[0]))
        # valid_size = len(self.views[0]) - train_size

        # # Create the full dataset
        # dataset = MultiViewDataset(self.views, self.labels)

        # # Create train and validation datasets
        # train_dataset = Subset(dataset, range(train_size))
        # valid_dataset = Subset(dataset, range(train_size, len(dataset)))

        # # Create data loaders
        # train_loader = DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=False
        # )
        # ortho_loader = DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=False
        # )
        # valid_loader = DataLoader(
        #     valid_dataset, batch_size=self.batch_size, shuffle=False
        # )
        return train_loader, ortho_loader, valid_loader
