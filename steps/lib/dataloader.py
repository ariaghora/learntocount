import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, data_dir: str, n_views: int):
        self.n_views = n_views
        self.data_dir = data_dir
        self.metadata = pd.read_csv(
            os.path.join(data_dir, "generated_dataset_metadata.csv")
        )

        self.target_counts = self.metadata["count"].values
        self.view_filenames = self.metadata.iloc[:, -self.n_views :].values

    def __len__(self):
        return len(os.listdir(self.data_dir)) // self.n_views

    def __getitem__(self, idx):
        images = []
        for view in range(self.n_views):
            filename = os.path.join(
                self.data_dir,
                f"img_{idx + 1}_view_{view + 1}.png",
            )
            image = Image.open(filename)
            images.append(image)

        image_tensors = [
            torch.from_numpy(np.array(image) / 255.0).float().permute(2, 0, 1)
            for image in images
        ]

        return self.target_counts[idx], image_tensors
