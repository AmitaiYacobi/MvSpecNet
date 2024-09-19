import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles


from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy.random import permutation



def generate_global_anomalies(X_normal, percentage=0.1):
    """
    Generate global anomalies for the given normal data.

    Args:
        X_normal (numpy.ndarray): The normal data.
        percentage (float): Percentage increase for anomaly generation.

    Returns:
        numpy.ndarray: The synthetic global anomalies.
    """
    X_synthetic_anomalies = []

    for i in range(X_normal.shape[1]):  # Iterate over features
        low = np.min(X_normal[:, i]) * (1 + percentage)
        high = np.max(X_normal[:, i]) * (1 + percentage)

        X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=X_normal.shape[0]))

    return np.array(X_synthetic_anomalies).T



def inject_global_anomalies(X, percentage=0.01):
    """
    Inject global anomalies into a portion of the dataset.

    Args:
        X (numpy.ndarray): The original dataset.
        percentage (float): Percentage of anomalies to inject.

    Returns:
        numpy.ndarray: The dataset with injected anomalies.
    """
    num_anomalies = int(len(X) * percentage)

    # Generate synthetic anomalies
    X_normal = X[np.random.choice(X.shape[0], size=num_anomalies, replace=False)]
    X_anomalies = generate_global_anomalies(X_normal, 0.1)



    # Select random indices to inject anomalies
    anomaly_indices = np.random.choice(X.shape[0], size=num_anomalies, replace=False)
    np.savetxt('anomalies_indices.txt', anomaly_indices)

    # Replace data at anomaly indices with anomalies
    X_with_anomalies = X.copy()
    X_with_anomalies[anomaly_indices] = X_anomalies

    return X_with_anomalies



class BDGP(Dataset):
    def __init__(self, path):
        self.x1 = scipy.io.loadmat(path + "BDGP.mat")["X1"].astype(np.float32)
        self.x2 = scipy.io.loadmat(path + "BDGP.mat")["X2"].astype(np.float32)
        self.y = scipy.io.loadmat(path + "BDGP.mat")["Y"].transpose()

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.x1[idx]),
            torch.from_numpy(self.x2[idx]),
        ], torch.from_numpy(self.y[idx])


class NoisyMNIST(Dataset):
    def __init__(self, path):
        data = np.load(path)
        scaler = MinMaxScaler()
        view1 = data["view_0"].reshape(70000, -1)
        view2 = data["view_1"].reshape(70000, -1)

        self.views = [torch.from_numpy(view1), torch.from_numpy(view2)]
        self.labels = torch.from_numpy(data["labels"])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]


class Caltech20(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()

        data = list(data["X"].transpose())
        views = [scaler.fit_transform(v.astype(np.float32)) for v in data]
        self.views = [torch.from_numpy(v) for v in views]
        self.labels = scipy.io.loadmat(path)["Y"] - 1
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return 2386
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]


class Handwritten(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()
        data = list(data["X"].transpose())

        views = [scaler.fit_transform(v.astype(np.float32)) for v in data]
        # views[0], views[2] = inject_global_anomalies(views[0]), inject_global_anomalies(views[2])
        self.views = [torch.from_numpy(views[0]), torch.from_numpy(views[2])]
        self.labels = scipy.io.loadmat(path)["Y"]
        self.labels = torch.from_numpy(self.labels)
    
    def __len__(self):
        return 2000
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



class Reuters(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path, simplify_cells=True)
        scaler = MinMaxScaler()
        train = data['x_train']
        test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']

        views = []

        for v_train, v_test in zip(train, test):
            v = np.vstack((v_train, v_test))
            views.append(scaler.fit_transform(v.astype(np.float32)))

        self.views = [torch.from_numpy(v) for v in views]
        self.labels = np.hstack((y_train, y_test))
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return 18758
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]



# class Synthetic(Dataset):
#     def __init__(self, n_samples=2000, noise=0.075, random_state=42):
#         self.n_samples = n_samples
#         self.noise = noise
#         self.random_state = random_state
#         self.n_views = 2

#         # Generate the two moons dataset
#         X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#         # Duplicate the dataset for each view and apply transformations
#         self.views = []
#         # self.views = [torch.from_numpy(X.copy()).float() for _ in range(self.n_views)]
#         for i in range(self.n_views):
#             X_view = X.copy()
#             if i % 2 == 0:
#                 # Standardize features
#                 scaler = StandardScaler()
#                 X_view = scaler.fit_transform(X_view)
#             else:
#                 # Apply random rotation
#                 rotation_angle = np.radians(np.random.uniform(0, 360))
#                 rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
#                                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
#                 X_view = np.dot(X_view, rotation_matrix)

#             self.views.append(X_view)

#         self.labels = torch.from_numpy(y)
#         self.views = [torch.from_numpy(view).float() for view in self.views]

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         return [view[idx] for view in self.views], self.labels[idx]



class Synthetic(Dataset):
    def __init__(self, n_samples=5000, noise=0.075, random_state=42, contamination=0.0):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.n_views = 2
        self.contamination = contamination  # proportion of contamination in the dataset

        # Generate the two moons dataset
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Duplicate the dataset for each view and apply transformations
        self.views = []
        for i in range(self.n_views):
            X_view = X.copy()
            if i % 2 == 0:
                # Standardize features
                scaler = StandardScaler()
                X_view = scaler.fit_transform(X_view)
            else:
                # Apply random rotation
                rotation_angle = np.radians(np.random.uniform(0, 360))
                rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                            [np.sin(rotation_angle), np.cos(rotation_angle)]])
                X_view = np.dot(X_view, rotation_matrix)

            # Introduce contamination to one of the views
            if i == 1 and self.contamination > 0:
                X_view = self._add_contamination(X_view, self.contamination)

            self.views.append(X_view)

        self.labels = torch.from_numpy(y)
        self.views = [torch.from_numpy(view).float() for view in self.views]

    def _add_contamination(self, X, contamination_level):
        """Add random noise to a percentage of the data."""
        num_contaminated = int(self.n_samples * contamination_level)
        contaminated_indices = np.random.choice(self.n_samples, num_contaminated, replace=False)
        
        # Add random noise to the selected indices
        noise = np.random.normal(0, 0.5, X[contaminated_indices].shape)  # increase noise scale
        X[contaminated_indices] += noise
        return X

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]


def train_test_split(data : Dataset):
    n_views = len(data[0][0])
    X, y = zip(*data)
    indices = permutation(range(len(data)))
    train_size = int(len(data) * 0.8)
    train_indices = indices[: train_size]
    test_indices = indices[train_size:]
    
    X_train = [X[i] for i in train_indices]  
    X_test = [X[i] for i in test_indices]   
    y = torch.tensor(y)

    v_train = [[] for _ in range(n_views)]
    v_test = [[] for _ in range(n_views)]
    for element in X_train:
        for i in range(n_views):
            v_train[i].append(element[i])
 
    for element in X_test:
        for i in range(n_views):
            v_test[i].append(element[i])
    
    X_train = [torch.stack(v) for v in v_train]
    X_test = [torch.stack(v) for v in v_test]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
    

def load_data(dataset: str) -> tuple:
    if dataset == "bdgp":
        data = BDGP("../data/")
        return train_test_split(data)

    elif dataset == "caltech20":
        data = Caltech20("../data/Caltech101-20.mat")
        return train_test_split(data)

    elif dataset == "handwritten":
        data = Handwritten("../data/handwritten.mat")
        return train_test_split(data)
    
    elif dataset == "reuters":
        data = Reuters("../data/Reuters_dim10.mat")
        return train_test_split(data)

    elif dataset == "noisy":
        data = NoisyMNIST("../data/noisymnist_train.npz")
        return train_test_split(data)

    elif dataset == "2d":
        data = Synthetic()
        return train_test_split(data)



