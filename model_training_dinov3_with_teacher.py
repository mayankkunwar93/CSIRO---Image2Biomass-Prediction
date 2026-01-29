# ============================================================
# FULL DINOv3 RGB TRAINING (LOG TARGETS) + SUBSPECIES AUX
# 4-HEAD OUTPUT: Total, GDM, Green, Clover | Dead derived
# ============================================================

import os, cv2, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from collections import defaultdict, Counter
import random
import torchvision.transforms as T
import torch.nn.functional as F

warnings.simplefilter("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = "/content/drive/MyDrive/mayank/CSIRO - Image2Biomass Prediction"
CSV_PATH = f"{BASE_DIR}/train.csv"

IMG_DIR = BASE_DIR
DINOV3_DIR = f"{BASE_DIR}/dinov3"

IMG_SIZES = [980]
BATCH_SIZE = 20
EPOCHS = 70
NUM_WORKERS = 4
N_SPLITS = 5
SUPERVISE_WEIGHT = 0.5

layer_unfreeze = 5

# medium LR settings
head_lr = 3e-4
lr_start, lr_end = 10e-5, 10e-5
min_lr = 2e-6

df_orig = pd.read_csv(CSV_PATH)

# ---------------------------
# Define the 4 main heads
# ---------------------------
targets_to_use = ["Dry_Total_g", "GDM_g", "Dry_Green_g", "Dry_Clover_g"]

df = df_orig[df_orig["target_name"].isin(targets_to_use)]
df = df.pivot(index="image_path", columns="target_name", values="target").reset_index()
df = df[["image_path"] + targets_to_use]
df[targets_to_use] = df[targets_to_use].apply(pd.to_numeric, errors='coerce').fillna(0)

target_mean, target_std = None, None

def denormalize(y):
    return np.expm1(y * target_std + target_mean)

# Normalize NDVI and Height
for col in ["Pre_GSHH_NDVI", "Height_Ave_cm"]:
    df_orig[col] = (df_orig[col] - df_orig[col].mean()) / (df_orig[col].std() + 1e-8)

df_supervise = df_orig[['image_path','Pre_GSHH_NDVI','Height_Ave_cm']].drop_duplicates()
df = df.merge(df_supervise, on='image_path', how='left')

# ----- species preprocessing -----
df_meta = df_orig[['image_path','Sampling_Date','State','Species']].drop_duplicates()
df_meta['group'] = (
    df_meta['Sampling_Date'].astype(str) + '_' +
    df_meta['State'].astype(str) + '_' +
    df_meta['Species'].astype(str)
)
df_meta['Species_list'] = df_meta['Species'].str.split('_')
df = df.merge(df_meta[['image_path','group','Species_list']], on='image_path', how='left')

species_to_subspecies = {
    # Grassy types - all anchored with "grass"
    "BarleyGrass": ["grass", "barley"], "Barleygrass": ["grass", "barley"],
    "Bromegrass": ["grass", "brome"], "Fescue": ["grass", "fescue"],
    "Phalaris": ["grass", "phalaris"], "Ryegrass": ["grass", "rye"],
    "SilverGrass": ["grass", "silver"], "SpearGrass": ["grass", "spear"],

    # Clovers - specifically influencing the Clover Head
    "Clover": ["clover"],
    "WhiteClover": ["clover", "white"],
    "SubcloverDalkeith": ["clover", "subclover", "dalkeith"],
    "SubcloverLosa": ["clover", "subclover", "losa"],

    # Lucerne - The "Third Category"
    "Lucerne": ["legume", "lucerne"],

    # Weeds
    "Capeweed": ["broadleaf", "capeweed"],
    "CrumbWeed": ["broadleaf", "crumbweed"],

    # Mixed
    "Mixed": ["mixed"]
}

all_subspecies = sorted(set(s for sp_list in df['Species_list'] for sp in sp_list for s in species_to_subspecies.get(sp, [sp])))
subspecies_to_idx = {s: i for i, s in enumerate(all_subspecies)}

def species_to_subspecies_multihot(sp_list):
    vec = np.zeros(len(all_subspecies), dtype=np.float32)
    for sp in sp_list:
        subs = species_to_subspecies.get(sp, [sp])
        for s in subs:
            vec[subspecies_to_idx[s]] = 1.0
    return vec

df['subspecies_multi_hot'] = df['Species_list'].apply(species_to_subspecies_multihot)

# ----- fold splitting -----
groups = df['group'].unique()
random.seed(42)

group_to_species = {}
for g in groups:
    species = df[df['group']==g]['Species_list'].explode().unique()
    group_to_species[g] = set(species)

folds = defaultdict(list)
species_in_fold = [Counter() for _ in range(N_SPLITS)]

for g in sorted(groups, key=lambda x: len(df[df['group']==x]), reverse=True):
    best_fold = min(range(N_SPLITS), key=lambda f: sum([species_in_fold[f][s] for s in group_to_species[g]]))
    folds[best_fold].append(g)
    for s in group_to_species[g]:
        species_in_fold[best_fold][s] += len(df[df['group']==g])

df['fold'] = -1
for f, gs in folds.items():
    df.loc[df['group'].isin(gs), 'fold'] = f

group_counts = df['group'].value_counts().to_dict()
max_count = max(group_counts.values())
group_weights = {g: max_count / c for g, c in group_counts.items()}
mean_w = np.mean(list(group_weights.values()))
group_weights = {g: w/mean_w for g, w in group_weights.items()}

# --------------------------- Processor ---------------------------
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained(DINOV3_DIR, local_files_only=True)

# --------------------------- Dataset ---------------------------
class GrassRGBDataset(Dataset):
    def __init__(self, df, base_dir, img_size=512, crop_size=224, train=True, group_weights=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.img_size = img_size
        self.crop_size = crop_size
        self.train = train
        self.group_weights = group_weights
        self.aug = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.RandomHorizontalFlip() if train else nn.Identity(),
            T.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)) if train else nn.Identity(),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # 1. LOAD DATA FIRST
            img = cv2.imread(os.path.join(IMG_DIR, row["image_path"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. APPLY AUGMENTATION TO LOADED DATA
            if self.train and random.random() > 0.5:
                img = cv2.flip(img, 1)

            # 3. NOW PROCESS GLOBAL VIEW
            img_global = processor(images=img, size=self.img_size, return_tensors="pt")["pixel_values"][0]

            # 4. EXTRACT CROPS FROM THE AUGMENTED IMAGE
            h, w, _ = img.shape
            cs = self.crop_size
            coords = [(0,0,cs,cs), (h-cs,0,h,cs), (0,w-cs,cs,w), (h-cs,w-cs,h,w)]

            crop_list = []
            for (y1, x1, y2, x2) in coords:
                c_rgb = img[y1:y2, x1:x2]
                c_t = processor(images=c_rgb, size=cs, return_tensors="pt")["pixel_values"][0]
                crop_list.append(c_t)

            crops = torch.stack(crop_list)

            target = torch.tensor(np.array(row["target"], dtype=np.float32))
            weight = torch.tensor(self.group_weights[row['group']] if self.group_weights else 1.0)
            # Calculate interaction on the fly
            interaction_val = row["Pre_GSHH_NDVI"] * row["Height_Ave_cm"]
            supervise_vals = torch.tensor([row["Pre_GSHH_NDVI"], row["Height_Ave_cm"], interaction_val, row["Month"]], dtype=torch.float32)

            subspecies_vals = torch.tensor(row["subspecies_multi_hot"], dtype=torch.float32)
            return img_global, crops, target, weight, supervise_vals, subspecies_vals

# --------------------------- Teacher NN ---------------------------
class TeacherNN(nn.Module):
    def __init__(self, in_features): # Dynamic based on metadata + species
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), # Increased width for categorical info
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4) # Predicts Total, GDM, Green, Clover
        )
    def forward(self, x):
        return self.net(x)

def train_teacher(X, y, weights, epochs=200, lr=1e-3):
    model = TeacherNN(in_features=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    w = torch.tensor(weights, dtype=torch.float32).to(DEVICE).unsqueeze(1) # (B, 1)

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        pred = torch.clamp(pred, min=0, max=12)
        # Manually compute weighted MSE: mean(w * (pred - y)^2)
        loss = (w * (pred - y)**2).mean()
        loss.backward()
        opt.step()
    return model.to(DEVICE)

# --------------------------- DINOv3 backbone ---------------------------
from transformers import AutoModel

class DinoV3Backbone(nn.Module):
    def __init__(self, model_dir, num_heads=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

        # 1. Freeze almost everything
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. Unfreeze fewer layers? With 357 images, unfreezing 20 layers is risky.
        # Consider reducing layer_unfreeze to 4-8 if you see validation loss rising.
        if layer_unfreeze > 0:
            for p in self.model.layer[-layer_unfreeze:].parameters():
                p.requires_grad = True

        self.d_model = self.model.config.hidden_size

        # 3. Robust Attention Pooling
        self.pooler = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Dropout(0.1), # Crucial for small datasets
            nn.Linear(self.d_model // 2, num_heads)
        )

        self.embed_dim = self.d_model * 2

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state

        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]

        attn_logits = self.pooler(patch_tokens)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Weighted sum of patches per head
        weighted_patches = torch.matmul(patch_tokens.transpose(1, 2), attn_weights)
        patch_summary = weighted_patches.mean(dim=-1)

        return torch.cat([cls_token, patch_summary], dim=1)


# --------------------------- Hierarchical Regressor ---------------------------

class DinoHierarchicalRegressor(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = DinoV3Backbone(DINOV3_DIR)
        d = self.backbone.embed_dim

        # NEW: Crop Attention Head
        self.crop_attention = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # The input is: Global(d) + Crops(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d + d),
            nn.Linear(d + d, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x_rgb, x_crops):
        # 1. Global & Depth
        f_global = self.backbone(x_rgb)

        # 2. Process Crops
        b, n_c, c, h, w = x_crops.shape
        x_crops_reshaped = x_crops.view(-1, c, h, w)
        f_crops = self.backbone(x_crops_reshaped) # (B*4, d)
        f_crops = f_crops.view(b, n_c, -1)         # (B, 4, d)

        # 3. Calculate Attention Weights for the 4 crops
        attn_logits = self.crop_attention(f_crops) # (B, 4, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Weighted sum of crop features
        f_crops_weighted = torch.sum(f_crops * attn_weights, dim=1) # (B, d)

        # 4. Final Fusion
        f_final = torch.cat([f_global, f_crops_weighted], dim=1)
        return self.head(f_final)

# --------------------------- Loss + Metrics ---------------------------
TARGET_WEIGHTS = torch.tensor([0.5,0.2,0.1,0.2], device=DEVICE)

def weighted_huber_loss(p, y, weights=None, supervise_vals=None, teacher_pred=None, delta=1.0):
    # 1. HARD SAFETY ON SCALING
    t_std = torch.tensor(target_std, device=p.device).float()
    t_std = torch.clamp(t_std, min=1e-3)
    t_mean = torch.tensor(target_mean, device=p.device).float()

    EPS = 1e-8

    # 2. CLAMP LOG SPACE INPUTS
    y_log = y * t_std + t_mean
    y_grams = torch.expm1(torch.clamp(y_log, max=10))

    loss_total = F.huber_loss(p[:, 0], y[:, 0], delta=delta)

    # 3. RATIO PROTECTION (Aggressive Zero-Handling)
    y_grams = torch.clamp(y_grams, min=0.0)
    dead_grams = torch.clamp(y_grams[:, 0] - y_grams[:, 1], min=0.0)

    # Ratio 1: GDM vs Dead
    true_ratio_gdm_dead = torch.stack([y_grams[:, 1], dead_grams], dim=1)
    denom1 = true_ratio_gdm_dead.sum(dim=1, keepdim=True)
    # If total weight is 0, use a neutral 0.5/0.5 split to avoid nan
    true_ratio_gdm_dead = torch.where(denom1 > EPS, true_ratio_gdm_dead / (denom1 + EPS), torch.full_like(true_ratio_gdm_dead, 0.5))

    # Ratio 2: Green vs Clover
    true_ratio_green_clover = torch.stack([y_grams[:, 2], y_grams[:, 3]], dim=1)
    denom2 = true_ratio_green_clover.sum(dim=1, keepdim=True)
    # If no green or clover exists, use 0.5/0.5 to keep KL Div stable
    true_ratio_green_clover = torch.where(denom2 > EPS, true_ratio_green_clover / (denom2 + EPS), torch.full_like(true_ratio_green_clover, 0.5))

    # Stabilize log_softmax by clamping predictions
    p_clamped = torch.clamp(p, min=-15, max=15)
    pred_ratio_gdm_dead_log = F.log_softmax(p_clamped[:, 1:3], dim=1)
    pred_ratio_green_clover_log = F.log_softmax(p_clamped[:, 3:5], dim=1)

    # Batchmean KL Divergence
    loss_ratio1 = F.kl_div(pred_ratio_gdm_dead_log, true_ratio_gdm_dead, reduction='batchmean')
    loss_ratio2 = F.kl_div(pred_ratio_green_clover_log, true_ratio_green_clover, reduction='batchmean')

    student_loss = (3.0 * loss_total) + (1.2 * (loss_ratio1 + loss_ratio2))

    # 4. TEACHER CLAMPING
    loss_supervise = torch.tensor(0.0, device=p.device)
    if teacher_pred is not None and SUPERVISE_WEIGHT > 0:
        teacher_pred = torch.clamp(teacher_pred, min=0, max=10)
        teacher_norm = (teacher_pred - t_mean) / t_std
        loss_supervise = F.huber_loss(p_clamped[:, :4], teacher_norm, delta=delta)

    # print(f"student_loss:{student_loss} and loss_supervise:{loss_supervise} ")
    total_loss = student_loss + (SUPERVISE_WEIGHT * loss_supervise)

    if weights is not None:
        weights = torch.clamp(weights, min=0.1, max=5.0)
        return (total_loss * weights).mean()
    return total_loss

def weighted_r2_score(y_true, y_pred):
    w = np.array([0.5,0.2,0.1,0.2])
    num = ((y_true - y_pred)**2 * w).sum()
    den = ((y_true - y_true.mean(axis=0))**2 * w).sum()
    return 1 - num / (den + 1e-8)

def train_one_epoch(model, loader, opt, teacher_model):
    model.train()
    losses = []
    for x_rgb, x_crops, y, weights, supervise_vals, subspecies_vals in loader:
        subspecies_vals = subspecies_vals.to(DEVICE)
        x_rgb, x_crops, y, weights, supervise_vals = (
            x_rgb.to(DEVICE),
            x_crops.to(DEVICE),
            y.to(DEVICE),
            weights.to(DEVICE),
            supervise_vals.to(DEVICE)
        )

        with torch.no_grad():
            # supervise_vals now contains [NDVI, Height, Interaction, Month]
            teacher_input = torch.cat([supervise_vals, subspecies_vals], dim=1)
            teacher_pred = teacher_model(teacher_input) if SUPERVISE_WEIGHT > 0 else None

        opt.zero_grad()
        # Pass all three inputs to the model
        preds = model(x_rgb, x_crops)
        loss = weighted_huber_loss(preds, y, weights, supervise_vals, teacher_pred)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())

    return np.mean(losses)

def validate(model, loader):
    model.eval()
    P_final, T_final = [], []
    total_rmse_grams = 0

    t_std = torch.tensor(target_std, device=DEVICE).float()
    t_mean = torch.tensor(target_mean, device=DEVICE).float()

    with torch.no_grad():
        for x_rgb, x_crops, y, _, _, _ in loader:
            p = model(x_rgb.to(DEVICE), x_crops.to(DEVICE))

            # Reconstruction Logic
            total_grams = torch.expm1(p[:, 0] * t_std[0] + t_mean[0])
            r_gdm = torch.softmax(p[:, 1:3], dim=1)[:, 0]
            r_green_clover = torch.softmax(p[:, 3:5], dim=1)

            gdm_grams = total_grams * r_gdm
            green_grams = gdm_grams * r_green_clover[:, 0]
            clover_grams = gdm_grams * r_green_clover[:, 1]

            p_reconstructed = torch.stack([total_grams, gdm_grams, green_grams, clover_grams], dim=1)
            p_log_norm = (torch.log1p(p_reconstructed) - t_mean) / t_std

            P_final.append(p_log_norm.cpu().numpy())
            T_final.append(y.numpy())

            y_grams = torch.expm1(y.to(DEVICE) * t_std + t_mean)
            total_rmse_grams += torch.sqrt(F.mse_loss(p_reconstructed, y_grams)).item()

    P, T = np.vstack(P_final), np.vstack(T_final)
    return r2_score(T, P), weighted_r2_score(T, P), total_rmse_grams / len(loader)


# --------------------------- Main Training Loop ---------------------------

# 1. Normalize original values in the source
for col in ["Pre_GSHH_NDVI", "Height_Ave_cm"]:
    df_orig[col] = (df_orig[col] - df_orig[col].mean()) / (df_orig[col].std() + 1e-8)

# 2. FIX: Drop the old/broken columns from df before merging fresh ones
cols_to_fix = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Sampling_Date', 'Month']
df = df.drop(columns=[c for c in cols_to_fix if c in df.columns])

# 3. Re-merge metadata correctly
df_supervise = df_orig[['image_path','Pre_GSHH_NDVI','Height_Ave_cm', 'Sampling_Date']].drop_duplicates()
df = df.merge(df_supervise, on='image_path', how='left')

# 4. Create the numerical Month and handle NAs
df['Month'] = pd.to_datetime(df['Sampling_Date']).dt.month.astype(np.float32)
df[['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Month']] = df[['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Month']].fillna(0)


for img_size in IMG_SIZES:
    print(f"\n==================== TRAINING IMG_SIZE={img_size} ====================")
    # for fold in range(N_SPLITS):
    for fold in [1, 2, 4, 3, 0]:
        tr_idx = df[df['fold'] != fold].index
        va_idx = df[df['fold'] == fold].index

        print(f"\n===== FOLD {fold+1} =====")
        print(f"Train samples: {len(tr_idx)} | Val samples: {len(va_idx)}")

        targets_tr = np.log1p(df.loc[tr_idx, targets_to_use].values.astype(np.float32))
        target_mean = targets_tr.mean(axis=0)

        # Hard safety floor to prevent division by zero in Fold 5
        target_std = np.maximum(targets_tr.std(axis=0), 1e-3)
        df.loc[tr_idx, "target"] = pd.Series(
            ((targets_tr - target_mean) / target_std).tolist(),
            index=tr_idx
        )
        targets_va = np.log1p(df.loc[va_idx, targets_to_use].values.astype(np.float32))
        df.loc[va_idx, "target"] = pd.Series(
            ((targets_va - target_mean) / target_std).tolist(),
            index=va_idx
        )

        # 1. Base Metadata
        teacher_X_meta = df.loc[tr_idx, ["Pre_GSHH_NDVI","Height_Ave_cm"]].values.astype(np.float32)

        # 2. Add Interaction Term (Height * NDVI) -> Proxy for Volume
        interaction = (teacher_X_meta[:, 0] * teacher_X_meta[:, 1]).reshape(-1, 1)

        # 3. Add Seasonality (Month)
        months = df.loc[tr_idx, ['Month']].values.astype(np.float32)

        # 4. Prepare Species (Multi-hot)
        subspecies_array = np.stack(df.loc[tr_idx, "subspecies_multi_hot"].values).astype(np.float32)

        # 5. Concatenate everything for the Smart Teacher
        # New shape: [NDVI, Height, Interaction, Month, Subspecies_0...N]
        teacher_X_combined = np.hstack([teacher_X_meta, interaction, months, subspecies_array])
        teacher_y = np.log1p(df.loc[tr_idx, targets_to_use].values.astype(np.float32))

        teacher_weights = np.array([group_weights[g] for g in df.loc[tr_idx, 'group']])

        # 6. Train with the new feature count
        teacher_model = train_teacher(teacher_X_combined, teacher_y, teacher_weights)

        train_loader = DataLoader(
            GrassRGBDataset(df.loc[tr_idx], IMG_DIR, img_size, train=True, group_weights=group_weights),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            GrassRGBDataset(df.loc[va_idx], IMG_DIR, img_size, train=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        model = DinoHierarchicalRegressor().to(DEVICE)

        backbone = model.backbone.model
        last_layers = list(backbone.layer[-layer_unfreeze:])
        lrs = np.linspace(lr_start, lr_end, len(last_layers))
        param_groups = [{"params": layer.parameters(), "lr": float(lr)} for layer, lr in zip(last_layers, lrs)]
        param_groups.append({
            "params": model.head.parameters(),
            "lr": head_lr
        })
        opt = torch.optim.AdamW(param_groups, weight_decay=0.05)
        # scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.8, patience=5, min_lr=min_lr)
        scheduler = CosineAnnealingLR(opt, T_max=EPOCHS-1, eta_min=min_lr)
        best_wr2 = -1

        # Path to the weights from your previous 0.1 supervisor run
        old_weight_path = f"{BASE_DIR}/with_teacher_mayank_final_{fold+1}_img{img_size}_bestWR2.pth"

        if os.path.exists(old_weight_path):
            model.load_state_dict(torch.load(old_weight_path, map_location=DEVICE))
            # Validate to get the actual baseline WR2 to beat
            _, best_wr2, rmse_best_wr2 = validate(model, val_loader)
            print(f"!!! LOADED SAVED MODEL for FOLD {fold+1} !!!")
            print(f">>> Current Best WR2: {best_wr2:.4f} | RMSE: {rmse_best_wr2:.4f}")
            print(f"Training will now attempt to improve this using SUPERVISE_WEIGHT={SUPERVISE_WEIGHT}")
        else:
            print(f"No previous weights found for Fold {fold+1}. Starting fresh training.")
            best_wr2 = -1

        for e in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, opt, teacher_model)
            r2, wr2, rmse = validate(model, val_loader)
            # scheduler.step(wr2)
            scheduler.step()

            print(f'Epoch {e+1:02d} | Loss {loss:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f} | WR2 {wr2:.4f} | LR: {opt.param_groups[0]["lr"]}')

            if wr2 > best_wr2:
                            print(f"New Best WR2! {wr2:.4f} (Previous best was {best_wr2:.4f}). Saving...")
                            best_wr2 = wr2
                            rmse_best_wr2 = rmse
                            torch.save(model.state_dict(), old_weight_path)

        print(f"\n===== FOLD {fold+1} IMG_SIZE {img_size} BEST WR2: {best_wr2:.4f} RMSE: {rmse_best_wr2:.4f} =====\n")

print("\nALL TRAINING COMPLETE")
#
