import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree

from task3.data import prepare_train_val_test_loaders


class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_output_dim, transformer_heads=3, graph_layer_type="gcn", predictor="linear"):
        super().__init__()
        if graph_layer_type == "gcn":
            transformer_heads = 1
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif graph_layer_type == "transformer":
            self.conv1 = TransformerConv(input_dim, hidden_dim, heads=transformer_heads)
            self.conv2 = TransformerConv(transformer_heads * hidden_dim, hidden_dim, heads=transformer_heads)

        self.pool = global_mean_pool

        if predictor == "linear":
            self.predictor = Linear(transformer_heads * hidden_dim, output_dim)
        elif predictor == "nonlinear":
            self.hidden_layer = Linear(transformer_heads * hidden_dim, hidden_output_dim)
            self.predictor = Linear(hidden_output_dim, output_dim)

        self.predictor_type = predictor

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)

        if self.predictor_type == "nonlinear":
            x = self.hidden_layer(x)
            x = x.relu()

        return self.predictor(x)


def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
    plt.colorbar(label='Class')
    plt.xlabel('Embedding dimension 1')
    plt.ylabel('Embedding dimension 2')
    plt.title('2D Embeddings of Beta-secretase Inhibitors')
    plt.show()


def visualize(h, color):
    print(len(h))
    z = TSNE(n_components=2).fit_transform(np.array(h))

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, name, device):
    print("Training model")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch.x = batch.x.float()
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.squeeze(-1).type(torch.int64))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch.x = batch.x.float()
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, batch.y.squeeze(-1).type(torch.int64))
                optimizer.step()
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'./models/{name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            batch.x = batch.x.float()
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)  # Ensure output is 1D
            predictions = output.argmax(dim=1)  # Convert logits/scores to binary predictions
            all_preds.extend(predictions.cpu().squeeze(-1).numpy())
            all_labels.extend(batch.y.cpu().squeeze(-1).numpy())
            all_inputs.extend(batch.x.cpu().squeeze(-1).numpy())
            all_outputs.extend(output.cpu().squeeze(-1).numpy())

    visualize(all_outputs, color=all_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return accuracy, precision, recall, f1


def run_bace():
    dataset = MoleculeNet(root="data/BACE", name="BACE")
    train_loader, val_loader, test_loader = prepare_train_val_test_loaders("BACE")

    input_dim = dataset.num_node_features
    hidden_dim = 128
    output_dim = dataset.num_classes
    hidden_output_dim = 64
    learning_rate = 0.0005
    num_epochs = 1000
    patience = 25
    name = "bace_gcn_linear"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    model = GNNClassifier(
        input_dim,
        hidden_dim,
        output_dim,
        hidden_output_dim,
        graph_layer_type="gcn",
        predictor="linear"
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Training loop
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, name, device)

    # Load best model and test
    model.load_state_dict(torch.load(f"./models/{name}.pth"))
    test_acc, test_prec, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(
        f"Test Accuracy = {test_acc:.4f}, Precision = {test_prec:.4f}, Recall = {test_recall:.4f}, F1 = {test_f1:.4f}")


if __name__ == "__main__":
    run_bace()


# Hyperparameters - bace_transformer_nonlinear
# Test Accuracy = 0.7697, Precision = 0.8333, Recall = 0.6962, F1 = 0.7586
# input_dim = dataset.num_node_features
# hidden_dim = 64
# output_dim = dataset.num_classes
# hidden_output_dim = 32
# learning_rate = 0.0001
# num_epochs = 1000
# patience = 25
# name = "bace_transformer_nonlinear"

# Hyperparameters - bace_transformer_linear
# Test Accuracy = 0.7697, Precision = 0.7164, Recall = 0.7500, F1 = 0.7328
# input_dim = dataset.num_node_features
# hidden_dim = 64
# output_dim = dataset.num_classes
# hidden_output_dim = 32
# learning_rate = 0.0001
# num_epochs = 1000
# patience = 25
# name = "bace_transformer_linear"

# Hyperparameters - bace_gcn_nonlinear
# Test Accuracy = 0.7303, Precision = 0.6364, Recall = 0.7119, F1 = 0.6720
# input_dim = dataset.num_node_features
# hidden_dim = 32
# output_dim = dataset.num_classes
# hidden_output_dim = 16
# learning_rate = 0.001
# num_epochs = 1000
# patience = 25
# name = "bace_gcn_nonlinear"

# Hyperparameters - bace_gcn_linear
# Test Accuracy = 0.7039, Precision = 0.6588, Recall = 0.7778, F1 = 0.7134
# input_dim = dataset.num_node_features
# hidden_dim = 128
# output_dim = dataset.num_classes
# hidden_output_dim = 64
# learning_rate = 0.0005
# num_epochs = 1000
# patience = 25
# name = "bace_gcn_linear"