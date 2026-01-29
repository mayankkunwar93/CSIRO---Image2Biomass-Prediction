import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class GrassRGBDataset(Dataset):
    def __init__(self, df, base_dir, img_size, train=True):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.img_size = img_size
        self.train = train

        if train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ColorJitter(0.05, 0.05, 0.05),
                T.RandomHorizontalFlip(),
                T.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.base_dir, row["image_path"].strip()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            k = np.random.choice([0, 1, 2, 3])
            img = np.rot90(img, k).copy()

        img = self.transform(img)
        return img, torch.tensor(row["target"], dtype=torch.float32)
