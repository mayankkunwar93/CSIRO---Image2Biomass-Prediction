# ============================================================
# FULL DINOv2 RGB TRAINING (LOG TARGETS)
# ============================================================

import os, cv2, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ------------------------
# SETTINGS
# ------------------------

warnings.simplefilter("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content/drive/MyDrive/mayank/CSIRO - Image2Biomass Prediction"
CSV_PATH = f"{BASE_DIR}/train.csv"
IMG_DIR = BASE_DIR
DINO_WEIGHTS = "/content/drive/MyDrive/mayank/dinov2_vits14_pretrain.pth"
# DINO_WEIGHTS = "/content/drive/MyDrive/mayank/dinov2_vitl14_reg4_pretrain.pth"

IMG_SIZES = [980,]
BATCH_SIZE = 32
EPOCHS = 50
NUM_WORKERS = 4

# ------------------------
# TARGET PREP
# ------------------------

df = pd.read_csv(CSV_PATH)

targets_to_use = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

df = df[df["target_name"].isin(targets_to_use)]
df = df.pivot(index="image_path", columns="target_name", values="target").reset_index()
df = df[["image_path"] + targets_to_use]

targets = np.log1p(df[targets_to_use].values.astype(np.float32))  # log targets
target_mean = targets.mean(axis=0)
target_std = targets.std(axis=0) + 1e-8

def normalize(y): return (y - target_mean) / target_std
def denormalize(y): return np.expm1(y * target_std + target_mean)

df["target"] = list(normalize(targets))

# ------------------------
# DATASET
# ------------------------

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
                T.ColorJitter(0.05,0.05,0.05),
                T.RandomHorizontalFlip(),
                T.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.base_dir, row["image_path"].strip()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            k = np.random.choice([0, 1, 2, 3])  # discrete rotation
            img = np.rot90(img, k).copy()

        img = self.transform(img)
        return img, torch.tensor(row["target"], dtype=torch.float32)

# ------------------------
# DINOv2 BACKBONE
# ------------------------

class DinoV2Backbone(nn.Module):

    def __init__(self, weight_path):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir="/content/drive/MyDrive/dinov2",
            model="dinov2_vits14",
            source="local",
            pretrained=False
        )
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.blocks[-4:].parameters():  # unfreeze last blocks
            p.requires_grad = True

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]

# ------------------------
# REGRESSION HEAD
# ------------------------

class DinoHierarchicalRegressor(nn.Module):

    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = DinoV2Backbone(DINO_WEIGHTS)

        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

# ------------------------
# LOSS & METRICS
# ------------------------

TARGET_WEIGHTS = torch.tensor([0.1,0.1,0.1,0.5,0.2], device=DEVICE)

def weighted_mse_loss(p, y, lambda_rel=0.05):
    base_loss = ((p - y)**2 * TARGET_WEIGHTS).sum() / TARGET_WEIGHTS.sum()
    return base_loss

def weighted_r2_score(y_true, y_pred):
    w = np.array([0.1,0.1,0.1,0.5,0.2])
    num = ((y_true - y_pred)**2 * w).sum()
    den = ((y_true - y_true.mean(axis=0))**2 * w).sum()
    return 1 - num / (den + 1e-8)

# ------------------------
# TRAIN / VALIDATE
# ------------------------

def train_one_epoch(model, loader, opt):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = weighted_mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(model(x.to(DEVICE)).cpu().numpy())
            T.append(y.numpy())
    P, T = np.vstack(P), np.vstack(T)
    P, T = denormalize(P), denormalize(T)
    return r2_score(T, P), weighted_r2_score(T, P), np.sqrt(((T - P) ** 2).mean())

# ------------------------
# K-FOLD TRAINING
# ------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for img_size in IMG_SIZES:
    print(f"\n==================== TRAINING IMG_SIZE={img_size} ====================")
    for fold, (tr, va) in enumerate(kf.split(df)):
        print(f"\n===== FOLD {fold+1} =====")

        train_loader = DataLoader(
            GrassRGBDataset(df.iloc[tr], IMG_DIR, img_size, train=True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

        val_loader = DataLoader(
            GrassRGBDataset(df.iloc[va], IMG_DIR, img_size, train=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        model = DinoHierarchicalRegressor().to(DEVICE)

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=3e-5, weight_decay=0.05
        )

        scheduler = CosineAnnealingLR(opt, T_max=EPOCHS-1, eta_min=1e-6)
        best_wr2 = -1

        for e in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, opt)
            r2, wr2, rmse = validate(model, val_loader)
            scheduler.step()

            print(f"Epoch {e+1:02d} | Loss {loss:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f} | WR2 {wr2:.4f}")

            if wr2 > best_wr2:
                best_wr2 = wr2
                torch.save(
                    model.state_dict(),
                    f"{BASE_DIR}/main_dino2_fold{fold+1}_img{img_size}_bestWR2.pth"
                )

        print(f"\n===== FOLD {fold+1} IMG_SIZE {img_size} BEST WR2: {best_wr2:.4f} =====\n")

print("\nALL TRAINING COMPLETE")
# # # #