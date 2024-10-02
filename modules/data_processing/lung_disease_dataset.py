import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable

class LungDiseaseDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform: Optional[Callable] = None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        """
        This method retrieves the image and label for a given index from the dataset.
        
        Parameters:
        idx (int): The index of the image and label in the dataframe.
        
        The dataframe is expected to have an 'image_index' column containing the file names of the images.
        """
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_index'])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx]['finding_labels']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)