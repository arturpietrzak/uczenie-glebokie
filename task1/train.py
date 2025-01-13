import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CelebA


def train(model, criterion, epochs, train_dataloader, val_dataloader, attribute_idx, patience=5, save_path='./models/gender_classifier.pth'):
    print("Training classifier")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for i, (images, targets) in enumerate(train_dataloader):
            labels = targets[:, attribute_idx].float().unsqueeze(1)
            images, labels = images.to(device), labels.to(device)

            model.optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], '
                      f'Loss: {train_loss / 100:.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
                train_loss = 0

        # Validation phase
        val_loss, val_accuracy = validate(model, criterion, attribute_idx, val_dataloader)
        print(f'Epoch [{epoch + 1}] Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_model(save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break


def validate(model, criterion, attribute_idx, val_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_dataloader:
            labels = targets[:, attribute_idx].float().unsqueeze(1)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return val_loss / len(val_dataloader), 100 * correct / total