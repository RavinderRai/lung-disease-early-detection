import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .data_loaders import create_data_loaders, create_transform
from ..utils.common import get_image_files

def main(df_file_path, index_list = [5]):
    if len(index_list) > 1:
        raise ValueError("You haven't adjusted the data class to intake images from more than one folder yet")
    
    image_files = get_image_files(index_list)

    df = pd.read_csv(df_file_path)

    subset_df = df[df["image_index"].isin(image_files)]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    img_dir = f"D:\BigData\images_00{index_list[0]}\images"

    transform = create_transform()
    train_loader, val_loader = create_data_loaders(train_df, val_df, img_dir, transform)

    torch.save(train_loader, "data/data_loaders/train_loader.pth")
    torch.save(val_loader, "data/data_loaders/val_loader.pth")

if __name__ == "__main__":
    main("data/lung_disease_labels.csv")