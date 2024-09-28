import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LungDiseaseDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_index'])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx]['finding_labels']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)