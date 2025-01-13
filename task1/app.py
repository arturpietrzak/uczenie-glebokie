import torch
import torch.nn as nn
from data import prepare_train_val_test_loaders, CustomWiderDataset, prepare_custom_wider_loader
from GenderClassifierCNN import GenderClassifierCNN
from SmilingClassifierResnet import SmilingClassifierResnet
from test import test, test_with_custom_wider
from train import train


def gender_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_dataloader, test_dataloader = prepare_train_val_test_loaders(batch_size=512)

    # training
    # model = GenderClassifierCNN().to(device)
    # criterion = nn.BCELoss()
    # train(model, criterion, 20, train_loader, val_loader, 20, 3, "./models/gender_classifier.pth")

    # testing
    model = GenderClassifierCNN().to(device)
    model.load_model("./models/gender_classifier.pth")
    test(model, test_dataloader, 20)

    # custom wider testing
    custom_wider_loader = prepare_custom_wider_loader()
    test_with_custom_wider(model, custom_wider_loader, "male")


def smiling_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataloader, val_dataloader, test_dataloader = prepare_train_val_test_loaders(batch_size=512, train_fraction=0.2)

    # training
    # model = SmilingClassifierResnet(learning_rate=0.00001).to(device)
    # criterion = nn.BCELoss()
    # train(model, criterion, 20, train_dataloader, val_dataloader, 31, 5, "./models/smiling_classifier.pth")

    # testing
    model = SmilingClassifierResnet().to(device)
    model.load_model("./models/smiling_classifier.pth")
    test(model, test_dataloader, 31)

    # custom wider testing
    custom_wider_loader = prepare_custom_wider_loader()
    test_with_custom_wider(model, custom_wider_loader, "smiling")

if __name__ == '__main__':
    gender_classifier()
    smiling_classifier()
