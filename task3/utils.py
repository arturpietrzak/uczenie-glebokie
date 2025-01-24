import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight


def evaluate_regressor(model, data_loader, device):
    model.eval()
    all_targets = []
    all_outputs = []
    all_embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            batch.x = batch.x.float()
            batch = batch.to(device)
            output, embeddings = model(batch.x, batch.edge_index, batch.batch, True)
            all_targets.extend(batch.y.cpu().squeeze(-1).numpy())
            all_outputs.extend(output.cpu().squeeze(-1).numpy())
            all_embeddings.extend(embeddings.cpu().squeeze(-1).numpy())

    # Calculate evaluation metrics
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)

    visualize_regressor_embeddings(np.array(all_embeddings), np.array(all_outputs), np.array(all_targets), model, "Dipole Moment")

    return mae, r2


def evaluate_classifier(model, data_loader, device):
    model.eval()
    all_preds = []
    all_inputs = []
    all_outputs = []
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch.x = batch.x.float()
            batch = batch.to(device)
            output, embeddings = model(batch.x, batch.edge_index, batch.batch, True)
            predictions = output.argmax(dim=1)
            all_preds.extend(predictions.cpu().squeeze(-1).numpy())
            all_labels.extend(batch.y.cpu().squeeze(-1).numpy())
            all_inputs.extend(batch.x.cpu().squeeze(-1).numpy())
            all_outputs.extend(output.cpu().squeeze(-1).numpy())
            all_embeddings.extend(embeddings.cpu().squeeze(-1).numpy())

    visualize_classifier_embeddings(np.array(all_embeddings), np.array(all_preds), all_labels, model, "Is beta-secretase inhibitor")

    # Calculate evaluation metrics
    accuracy = accuracy_score(np.array(all_labels), np.array(all_preds))
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return accuracy, precision, recall, f1


def calculate_class_weights(data):
    labels = [int(y.item()) for y in data.y]
    classes = torch.unique(torch.tensor(labels))
    class_weights = compute_class_weight('balanced', classes=classes.numpy(), y=labels)

    return torch.tensor(class_weights, dtype=torch.float32)


def visualize_regressor_embeddings(embeddings, predictions, labels, model, label=""):
    if len(embeddings.shape) == 2 and embeddings.shape[1] == 2:
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=25)
        # plt.colorbar(scatter, label=label)
        # plt.title('Visualization of Embeddings')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.show()

        x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
        y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            function_values = model.from_embeddings(grid_tensor).cpu().numpy()

        function_values = function_values.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, function_values, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(label=label)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=predictions, cmap='viridis', edgecolor='k')
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.title("Approximation Function in Embedding Space")
        plt.show()
    if len(embeddings.shape) == 1:
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embeddings[:], labels, c=labels, cmap='viridis', s=25)
        # plt.colorbar(scatter, label=label)
        # plt.title('Visualization of Embeddings')
        # plt.xlabel('Dimension 1')
        # plt.show()

        sorted_indices = np.argsort(embeddings)
        embeddings = embeddings[sorted_indices]
        predictions = predictions[sorted_indices]

        x_min, x_max = embeddings.min() - 1, embeddings.max() + 1
        grid_points = np.linspace(x_min, x_max, 200).reshape(-1, 1)

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            function_values = model.from_embeddings(grid_tensor).cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.plot(grid_points, function_values, color="red", label="Approximation Function", linewidth=2)
        plt.scatter(embeddings, predictions, c=labels, cmap='viridis', alpha=0.7, zorder=10)
        plt.xlabel("1D Embedding")
        plt.ylabel("Function Value")
        plt.title("Approximation Function in 1D Embedding Space")
        plt.legend()
        plt.grid()
        plt.show()


def visualize_classifier_embeddings(embeddings, predictions, labels, model, label=""):
    if len(embeddings.shape) == 2 and embeddings.shape[1] == 2:
        x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
        y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            boundary_predictions = np.argmax(model.from_embeddings(grid_tensor).cpu().numpy(), axis=1)
            boundary_predictions = boundary_predictions.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, boundary_predictions, alpha=0.5, cmap="viridis", c=boundary_predictions)
        plt.colorbar(label="Predicted Class")
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="viridis", edgecolor="k")
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.title("2D Embedding with Decision Boundaries")
        plt.grid()
        plt.show()

    # if len(embeddings.shape) == 1:
    #     plt.figure(figsize=(10, 8))
    #     scatter = plt.scatter(embeddings[:], labels, c=labels, cmap='viridis', s=25)
    #     plt.colorbar(scatter, label=label)
    #     plt.title('Visualization of Embeddings')
    #     plt.xlabel('Dimension 1')
    #     plt.show()
    #
    #     x_min, x_max = embeddings.min() - 1, embeddings.max() + 1
    #     grid_points = np.linspace(x_min, x_max, 500).reshape(-1, 1)  # 1D grid
    #
    #     with torch.no_grad():
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    #         grid_predictions = model.from_embeddings(grid_tensor).cpu().numpy()
    #         class_probs = torch.softmax(torch.tensor(grid_predictions), dim=1).numpy()
    #         boundary_predictions = np.argmax(class_probs, axis=1)
    #
    #     plt.figure(figsize=(10, 8))
    #     for class_idx in range(class_probs.shape[1]):
    #         plt.plot(grid_points, class_probs[:, class_idx], label=f"Class {class_idx} Probability")
    #     plt.scatter(embeddings, labels, c=labels, cmap='viridis', alpha=0.7, zorder=10)
    #     plt.xlabel("1D Embedding")
    #     plt.ylabel("Function Value")
    #     plt.title("Decision Boundaries in 1D Embedding Space")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

