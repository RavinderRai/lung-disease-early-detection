from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .lung_disease_dataset import LungDiseaseDataset

def create_data_loaders(train_df, val_df, img_dir, transform):
    train_dataset = LungDiseaseDataset(train_df, img_dir, transform=transform)
    val_dataset = LungDiseaseDataset(val_df, img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader

def create_transform(size=224, rotation_degree=10):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])