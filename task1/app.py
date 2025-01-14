import numpy as np
import torch
import torch.nn as nn
from data import prepare_train_val_test_loaders, CustomWiderDataset, prepare_custom_wider_loader
from GenderClassifierCNN import GenderClassifierCNN
from SmilingClassifierResnet import SmilingClassifierResnet
from test import test, test_with_custom_wider
from train import train
from PIL import Image
import torchvision.transforms as transforms



def gender_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # train_loader, val_dataloader, test_dataloader = prepare_train_val_test_loaders(batch_size=512)

    # training
    model = GenderClassifierCNN(learning_rate=0.0001).to(device)
    criterion = nn.BCELoss()
    # train(model, criterion, 50, train_loader, val_dataloader, 20, 3, "./models/gender_classifier1.pth")

    # testing
    model = GenderClassifierCNN().to(device)
    model.load_model("./models/gender_classifier1.pth")
    model.eval()
    # test(model, test_dataloader, 20)

    # custom wider testing
    # custom_wider_loader = prepare_custom_wider_loader()
    # test_with_custom_wider(model, custom_wider_loader, "male")

    for i in range(1, 385):
        img = Image.open(f"./data/wider_test/wider_test_processed/{i}.png").convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        tesnor = transform(img)
        tesnor = tesnor.cuda()

        result = model(tesnor.unsqueeze(0)).item()

        print(f"{i}", result)


def smiling_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # train_dataloader, val_dataloader, test_dataloader = prepare_train_val_test_loaders(batch_size=512, train_fraction=0.05, resnet=True)

    # training
    model = SmilingClassifierResnet(learning_rate=0.000005).to(device)
    criterion = nn.BCELoss()
    # train(model, criterion, 50, train_dataloader, val_dataloader, 31, 3, "./models/smiling_classifier1.pth")

    # testing
    model = SmilingClassifierResnet().to(device)
    model.load_model("./models/smiling_classifier1.pth")
    # test(model, test_dataloader, 31)
    model.eval()

    # custom wider testing
    # custom_wider_loader = prepare_custom_wider_loader(resnet=True)
    # test_with_custom_wider(model, custom_wider_loader, "smiling")

    for i in range(1, 385):
        img = Image.open(f"./data/wider_test/wider_test_processed/{i}.png").convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tesnor = transform(img)
        tesnor = tesnor.cuda()

        result = model(tesnor.unsqueeze(0)).item()

        print(f"{i}", result)


if __name__ == '__main__':
    gender_classifier()
    smiling_classifier()
