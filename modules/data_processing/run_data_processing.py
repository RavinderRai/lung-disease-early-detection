import logging
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .data_loaders import create_data_loaders, create_transform
from ..utils.common import get_image_files

def main(df_file_path, index_list = [5]):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if len(index_list) > 1:
        raise ValueError("You haven't adjusted the data class to intake images from more than one folder yet")
    
    logger.info("Retrieving image files...")
    image_files = get_image_files(index_list)

    logger.info(f"Loading dataset from {df_file_path}...")
    df = pd.read_csv(df_file_path)

    logger.info("Filtering dataset for selected image indices...")
    subset_df = df[df["image_index"].isin(image_files)]

    logger.info("Splitting dataset into training and validation sets...")
    train_df, val_df = train_test_split(subset_df, test_size=0.2, random_state=42)

    img_dir = f"D:\BigData\images_00{index_list[0]}\images"

    logger.info("Creating data transformation...")
    transform = create_transform()

    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_df, val_df, img_dir, transform)

    logger.info("Saving training data loader...")
    torch.save(train_loader, "artifacts/data_loaders/train_loader.pth")
    logger.info("Saving validation data loader...")
    torch.save(val_loader, "artifacts/data_loaders/val_loader.pth")

if __name__ == "__main__":
    main("artifacts/labels_data/lung_disease_labels.csv")