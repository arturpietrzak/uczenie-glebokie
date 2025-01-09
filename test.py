import torch
from sklearn.metrics import confusion_matrix
from GenderClassifierCNN import GenderClassifierCNN


def test(model, test_dataloader, attribute_idx):
    print("Testing model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_dataloader:
            labels = targets[:, attribute_idx].float()
            images = images.to(device)

            outputs = model(images).cpu().squeeze()
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predictions)

    print(accuracy)
    print(cm)


def test_with_custom_wider(model, test_dataloader, attribute="male"):
    print("Testing model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_dataloader:
            if attribute == "male":
                images = images.to(device)
                labels = targets[0].to(device)
            else:
                images = images.to(device)
                labels = targets[1].to(device)

            outputs = model(images)
            predictions = (outputs > 0.5).float().squeeze()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predictions)

    print(accuracy)
    print(cm)