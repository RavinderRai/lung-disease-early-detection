from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .lung_disease_dataset import LungDiseaseDataset
from typing import Tuple

def create_data_loaders(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    img_dir: Path, 
    transform: transforms.Compose
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns data loaders for training and validation datasets.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing training data.
    val_df (pd.DataFrame): DataFrame containing validation data.
    img_dir (Path): Path to the directory containing images.
    transform (transforms.Compose): A composition of transforms to apply to images.

    Returns:
    tuple: A tuple containing the DataLoader for training data and the DataLoader for validation data.
    """
    train_dataset = LungDiseaseDataset(train_df, img_dir, transform=transform)
    val_dataset = LungDiseaseDataset(val_df, img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader

def create_transform(size: int = 224, rotation_degree: int = 10) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])