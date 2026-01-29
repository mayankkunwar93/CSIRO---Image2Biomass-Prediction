import os, cv2, torch, random
import numpy as np
from torch.utils.data import Dataset

class GrassRGBDataset(Dataset):
    def __init__(self, df, img_dir, processor, img_size=980, crop_size=224, train=True, group_weights=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.img_size = img_size
        self.crop_size = crop_size
        self.train = train
        self.group_weights = group_weights

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"].strip())
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.train and random.random() > 0.5: img = cv2.flip(img, 1)

        img_global = self.processor(images=img, size=self.img_size, return_tensors="pt")["pixel_values"][0]

        h, w, _ = img.shape
        cs = self.crop_size
        coords = [(0,0,cs,cs), (h-cs,0,h,cs), (0,w-cs,cs,w), (h-cs,w-cs,h,w)]
        crop_list = [self.processor(images=img[y1:y2, x1:x2], size=cs, return_tensors="pt")["pixel_values"][0] for (y1, x1, y2, x2) in coords]
        crops = torch.stack(crop_list)

        target = torch.tensor(np.array(row["target"], dtype=np.float32))
        weight = torch.tensor(self.group_weights[row['group']] if self.group_weights else 1.0)
        interaction = row["Pre_GSHH_NDVI"] * row["Height_Ave_cm"]
        supervise_vals = torch.tensor([row["Pre_GSHH_NDVI"], row["Height_Ave_cm"], interaction, row["Month"]], dtype=torch.float32)
        subspecies_vals = torch.tensor(row["subspecies_multi_hot"], dtype=torch.float32)

        return img_global, crops, target, weight, supervise_vals, subspecies_vals