import torch
from torch_geometric.datasets import MoleculeNet, QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np




def prepare_train_val_test_loaders(dataset_name, batch_size=32, property_idx=0):
    if dataset_name == "BACE":
        dataset = MoleculeNet(root="data/BACE", name="BACE")

        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    elif dataset_name == "QM9":
        dataset = QM9(root="data/QM9")
        dataset.data.y = dataset.data.y[:, property_idx:property_idx + 1]

        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader