import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

class OLPRDataset(Dataset):
    def __init__(self, root, split, img_size=224):
        """
        Args:
            root (str):  (e.g., './root_data')
            split (str): 'train', 'valid', or 'test'
            img_size (int): 
        """
        json_path = os.path.join(root, f"{split}.json")
        with open(json_path, 'r') as f:
            data_dict = json.load(f)

        self.images = [os.path.join(root, path) for path in data_dict.keys()]
        self.labels = list(data_dict.values())
        self.img_size = img_size
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.transforms = self._get_transforms(split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self._normalize(image)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.as_tensor(self.labels[index])
        return image, label

    def _normalize(self, img):
        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        return img

    def _get_transforms(self, split):
        if split == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.3,
                                   rotate_limit=180, border_mode=0, p=0.7),
            ])
        else:
            return A.Compose([A.Resize(self.img_size, self.img_size)])
