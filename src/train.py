import os, random, warnings, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from collections import defaultdict, Counter
from sklearn.model_selection import KFold

from models import TeacherNN, DinoHierarchicalRegressor
from dataset import GrassRGBDataset
from utils import SPECIES_TO_SUBSPECIES, weighted_huber_loss, weighted_r2_score

warnings.simplefilter("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/content/drive/MyDrive/mayank/CSIRO - Image2Biomass Prediction"
CSV_PATH = f"{BASE_DIR}/train.csv"
DINOV3_DIR = f"{BASE_DIR}/dinov3"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_subspecies_vector(sp_list, all_subs, sub_to_idx):
    vec = np.zeros(len(all_subs), dtype=np.float32)
    for sp in sp_list:
        subs = SPECIES_TO_SUBSPECIES.get(sp, [sp])
        for s in subs: vec[sub_to_idx[s]] = 1.0
    return vec

def main():
    seed_everything()
    df_orig = pd.read_csv(CSV_PATH)
    targets_to_use = ["Dry_Total_g", "GDM_g", "Dry_Green_g", "Dry_Clover_g"]

    # Pivot and Clean
    df = df_orig[df_orig["target_name"].isin(targets_to_use)]
    df = df.pivot(index="image_path", columns="target_name", values="target").reset_index()
    df[targets_to_use] = df[targets_to_use].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Metadata Prep
    df_meta = df_orig[['image_path','Pre_GSHH_NDVI','Height_Ave_cm','Sampling_Date','Species']].drop_duplicates()
    df = df.merge(df_meta, on='image_path', how='left')
    df['Month'] = pd.to_datetime(df['Sampling_Date']).dt.month.astype(np.float32)
    df['Species_list'] = df['Species'].str.split('_')

    all_subs = sorted(set(s for sp_list in df['Species_list'] for sp in sp_list for s in SPECIES_TO_SUBSPECIES.get(sp, [sp])))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    df['subspecies_multi_hot'] = df['Species_list'].apply(lambda x: get_subspecies_vector(x, all_subs, sub_to_idx))
    df['group'] = df['Sampling_Date'].astype(str) + '_' + df['Species'].astype(str)

    # Fold logic (Simplified for full script)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold

    processor = AutoImageProcessor.from_pretrained(DINOV3_DIR, local_files_only=True)

    for fold in range(5):
        tr_idx, va_idx = df[df['fold'] != fold].index, df[df['fold'] == fold].index
        targets_tr = np.log1p(df.loc[tr_idx, targets_to_use].values.astype(np.float32))
        t_mean, t_std = targets_tr.mean(axis=0), np.maximum(targets_tr.std(axis=0), 1e-3)
        df.loc[df.index, 'target'] = list((np.log1p(df[targets_to_use].values) - t_mean) / t_std)

        # Teacher Training
        teacher_X = np.hstack([df.loc[tr_idx, ["Pre_GSHH_NDVI","Height_Ave_cm"]].values, (df.loc[tr_idx, "Pre_GSHH_NDVI"] * df.loc[tr_idx, "Height_Ave_cm"]).values.reshape(-1,1), df.loc[tr_idx, ["Month"]].values, np.stack(df.loc[tr_idx, "subspecies_multi_hot"].values)])
        teacher_model = TeacherNN(in_features=teacher_X.shape[1]).to(DEVICE)
        # ... teacher training loop ...

        train_loader = DataLoader(GrassRGBDataset(df.loc[tr_idx], BASE_DIR, processor), batch_size=20, shuffle=True)
        model = DinoHierarchicalRegressor(DINOV3_DIR).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for epoch in range(70):
            model.train()
            for x_rgb, x_crops, y, w, s_vals, sub_vals in train_loader:
                opt.zero_grad()
                p = model(x_rgb.to(DEVICE), x_crops.to(DEVICE))
                loss = weighted_huber_loss(p, y.to(DEVICE), t_std, t_mean, w.to(DEVICE))
                loss.backward()
                opt.step()
            print(f"Fold {fold} Epoch {epoch} complete")

if __name__ == "__main__":
    main()