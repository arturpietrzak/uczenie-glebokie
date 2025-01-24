import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.datasets import MoleculeNet, QM9

from task3.GNNClassifier import GNNClassifier
from task3.GNNRegressor import GNNRegressor
from task3.data import prepare_train_val_test_loaders
from task3.train import train_classifier, train_regressor
from task3.utils import calculate_class_weights, evaluate_classifier, evaluate_regressor


def run_bace(embedding_dim, graph_layer_type, predictor):
    dataset = MoleculeNet(root="data/BACE", name="BACE")
    train_loader, val_loader, test_loader = prepare_train_val_test_loaders("BACE")

    input_dim = dataset.num_node_features
    # embedding_dim = 2
    output_dim = dataset.num_classes
    hidden_output_dim = 128
    learning_rate = 0.0005
    num_epochs = 1000
    patience = 25
    # graph_layer_type = "transformer"
    # predictor = "nonlinear"

    name = f"bace_{graph_layer_type}_{predictor}_{embedding_dim}"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    class_weights = calculate_class_weights(dataset.data).to(device)

    model = GNNClassifier(
        input_dim,
        embedding_dim,
        output_dim,
        hidden_output_dim,
        graph_layer_type=graph_layer_type,
        predictor=predictor
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss(weight=class_weights)
    # Test Accuracy = 0.6776, Precision = 0.6000, Recall = 0.7031, F1 = 0.6475
    # Training loop
    # train_classifier(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, name, device)

    # Load best model and test
    model.load_state_dict(torch.load(f"./models/{name}.pth"))
    test_acc, test_prec, test_recall, test_f1 = evaluate_classifier(model, test_loader, device)
    print(name)
    print(
        f"acc = {test_acc:.2f}, pre = {test_prec:.2f}, rec = {test_recall:.2f}, f1 = {test_f1:.2f}")


def run_qm9(embedding_dim, graph_layer_type, predictor):
    dataset = QM9(root="data/QM9")
    train_loader, val_loader, test_loader = prepare_train_val_test_loaders("QM9", batch_size=64)

    input_dim = dataset.num_node_features
    # embedding_dim = 128
    hidden_output_dim = 64
    learning_rate = 0.001
    num_epochs = 1000
    patience = 10
    # graph_layer_type = "transformer"
    # predictor = "linear"

    name = f"qn9_{graph_layer_type}_{predictor}_{embedding_dim}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNRegressor(
        input_dim,
        embedding_dim,
        hidden_output_dim,
        graph_layer_type=graph_layer_type,
        predictor=predictor
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # train_regressor(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, name, device)

    model.load_state_dict(torch.load(f"./models/{name}.pth"))
    test_mae, test_r2 = evaluate_regressor(model, test_loader, device)
    print(f"Test mae = {test_mae:.4f}, r2 = {test_r2:.4f}")


if __name__ == "__main__":
    # run_bace()
    # run_qm9(1, "transformer", "linear")
    # run_qm9(2, "transformer", "linear")
    # run_qm9(128, "transformer", "linear")
    #
    # run_qm9(1, "transformer", "nonlinear")
    # run_qm9(2, "transformer", "nonlinear")
    # run_qm9(128, "transformer", "nonlinear")
    #
    # run_qm9(1, "gcn", "linear")
    # run_qm9(2, "gcn", "linear")
    # run_qm9(128, "gcn", "linear")
    #
    # run_qm9(1, "gcn", "nonlinear")
    # run_qm9(2, "gcn", "nonlinear")
    # run_qm9(128, "gcn", "nonlinear")

    run_bace(1, "transformer", "linear")
    run_bace(2, "transformer", "linear")
    run_bace(128, "transformer", "linear")

    run_bace(1, "transformer", "nonlinear")
    run_bace(2, "transformer", "nonlinear")
    run_bace(128, "transformer", "nonlinear")

    run_bace(1, "gcn", "linear")
    run_bace(2, "gcn", "linear")
    run_bace(128, "gcn", "linear")

    run_bace(1, "gcn", "nonlinear")
    run_bace(2, "gcn", "nonlinear")
    run_bace(128, "gcn", "nonlinear")
