import pandas as pd
from torch import Generator
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CelebA
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

"""
From https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
download:
img_align_celeba.zip and extract into img_align_celeba/ directory
identity_CelebA.txt
list_attr_celeba.txt
list_bbox_celeba.txt
list_eval_partition.txt
list_landmarks_align_celeba_txt
list_landmarks_celeba.txt

And put all files in data/celeba/ directory
"""


class CustomWiderDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.data['male'] = self.data['male'].replace(-1, 0)
        self.data['smiling'] = self.data['smiling'].replace(-1, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.data.iloc[idx]['number']}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        male = self.data.iloc[idx]['male']
        smiling = self.data.iloc[idx]['smiling']

        if self.transform:
            image = self.transform(image)

        return image, (torch.tensor(male, dtype=torch.float), torch.tensor(smiling, dtype=torch.float))


def prepare_train_val_test_loaders(batch_size=512, train_fraction=0.8, resnet=False):
    print(f"Loading datasets")
    if resnet:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    full_train_dataset = CelebA(root='./data', split='train', target_type='attr', transform=transform, download=False)
    train_size = int(train_fraction * len(full_train_dataset))
    train_dataset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size], generator=Generator().manual_seed(42))

    val_dataset = CelebA(root='./data', split='valid', target_type='attr', transform=transform, download=False)
    test_dataset = CelebA(root='./data', split='test', target_type='attr', transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def prepare_custom_wider_loader(batch_size=512, resnet=False):
    if resnet:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    custom_dataset = CustomWiderDataset(
        "./data/wider_test/wider_test_processed_labels.csv",
        "./data/wider_test/wider_test_processed",
        transform=transform
    )

    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

