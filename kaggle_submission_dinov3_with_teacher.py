import os, cv2, warnings, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from collections import defaultdict, Counter

# ------------------------
# CONFIG / DEVICE
# ------------------------
warnings.simplefilter("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fold weights (kept uniform here)
WR2_SCORES = [0.2] * 5
fold_weights = torch.tensor(WR2_SCORES, device=DEVICE).float()
fold_weights = fold_weights / fold_weights.sum()

DATA_DIR = "/kaggle/input/csiro-biomass"
DINOV3_DIR = "/kaggle/input/dinov3-repo-source/dinov3"
CKPT_DIR = "/kaggle/input/with-teacher-mayank-final-img980-bestwr2"

IMG_SIZE = 980
CROP_SIZE = 224
TARGET_COLS = ["Dry_Total_g", "GDM_g", "Dry_Green_g", "Dry_Clover_g"]
N_SPLITS = 5

# ------------------------
# FOLD MEAN / STD (LOG SPACE)
# ------------------------
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_p = (
    df_train[df_train["target_name"].isin(TARGET_COLS)]
    .pivot(index="image_path", columns="target_name", values="target")
    .reset_index()
)

df_meta = df_train[['image_path','Sampling_Date','State','Species']].drop_duplicates()
df_meta['group'] = (
    df_meta['Sampling_Date'].astype(str) + '_' +
    df_meta['State'].astype(str) + '_' +
    df_meta['Species'].astype(str)
)
df_meta['Species_list'] = df_meta['Species'].str.split('_')
df_p = df_p.merge(df_meta[['image_path','group','Species_list']], on='image_path', how='left')

groups = df_p['group'].unique()
group_to_species = {
    g: set(df_p[df_p['group'] == g]['Species_list'].explode().unique())
    for g in groups
}

folds = defaultdict(list)
species_in_fold = [Counter() for _ in range(N_SPLITS)]

for g in sorted(groups, key=lambda x: len(df_p[df_p['group'] == x]), reverse=True):
    best_fold = min(
        range(N_SPLITS),
        key=lambda f: sum(species_in_fold[f][s] for s in group_to_species[g])
    )
    folds[best_fold].append(g)
    for s in group_to_species[g]:
        species_in_fold[best_fold][s] += len(df_p[df_p['group'] == g])

df_p['fold'] = -1
for f, gs in folds.items():
    df_p.loc[df_p['group'].isin(gs), 'fold'] = f

FOLD_STATS = []
for fold in range(N_SPLITS):
    tr_idx = df_p[df_p['fold'] != fold].index
    targets_tr = np.log1p(df_p.loc[tr_idx, TARGET_COLS].values.astype(np.float32))
    mean = torch.tensor(targets_tr.mean(axis=0), device=DEVICE)
    std = torch.tensor(np.maximum(targets_tr.std(axis=0), 1e-3), device=DEVICE)
    FOLD_STATS.append((mean, std))

# ------------------------
# MODEL
# ------------------------
class DinoV3Backbone(nn.Module):
    def __init__(self, model_dir, num_heads=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        self.d_model = self.model.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, num_heads)
        )
        self.embed_dim = self.d_model * 2

    def forward(self, x):
        h = self.model(pixel_values=x).last_hidden_state
        cls = h[:, 0]
        attn = torch.softmax(self.pooler(h[:, 1:]), dim=1)
        patches = torch.matmul(h[:, 1:].transpose(1, 2), attn).mean(dim=-1)
        return torch.cat([cls, patches], dim=1)

class DinoHierarchicalRegressor(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = DinoV3Backbone(DINOV3_DIR)
        d = self.backbone.embed_dim
        self.crop_attention = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x_rgb, x_crops):
        f_global = self.backbone(x_rgb)

        b, n, c, h, w = x_crops.shape
        f_crops = self.backbone(x_crops.view(-1, c, h, w)).view(b, n, -1)

        w_attn = torch.softmax(self.crop_attention(f_crops), dim=1)
        f_crops = (f_crops * w_attn).sum(dim=1)

        return self.head(torch.cat([f_global, f_crops], dim=1))

# ------------------------
# LOAD MODELS
# ------------------------
processor = AutoImageProcessor.from_pretrained(DINOV3_DIR, local_files_only=True)

models = []
for fold in range(1, 6):
    m = DinoHierarchicalRegressor().to(DEVICE)
    ckpt = f"{CKPT_DIR}/with_teacher_mayank_final_{fold}_img{IMG_SIZE}_bestWR2.pth"
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    models.append(m)

# ------------------------
# INFERENCE
# ------------------------
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
unique_paths = test_df["image_path"].unique()
results = {}

with torch.no_grad():
    for p in tqdm(unique_paths):
        img = cv2.cvtColor(
            cv2.imread(os.path.join(DATA_DIR, "test", os.path.basename(p))),
            cv2.COLOR_BGR2RGB
        )
        h, w, _ = img.shape

        img_g = processor(images=img, size=IMG_SIZE, return_tensors="pt")["pixel_values"].to(DEVICE)

        cs = CROP_SIZE
        coords = [(0,0,cs,cs), (h-cs,0,h,cs), (0,w-cs,cs,w), (h-cs,w-cs,h,w)]
        crops = [
            processor(images=img[y1:y2, x1:x2], size=cs, return_tensors="pt")["pixel_values"][0]
            for (y1,x1,y2,x2) in coords
        ]
        img_c = torch.stack(crops).unsqueeze(0).to(DEVICE)

        fold_preds = []
        for i, m in enumerate(models):
            p_out = m(img_g, img_c)
            t_m, t_s = FOLD_STATS[i]

            total = torch.expm1(p_out[:, 0] * t_s[0] + t_m[0])
            r_gdm = torch.softmax(p_out[:, 1:3], dim=1)[:, 0]
            r_gc = torch.softmax(p_out[:, 3:5], dim=1)

            gdm = total * r_gdm
            fold_preds.append(
                torch.stack([total, gdm, gdm * r_gc[:, 0], gdm * r_gc[:, 1]], dim=1)
            )

        avg = (torch.stack(fold_preds) * fold_weights[:, None, None]).sum(0)
        avg = avg.cpu().numpy().flatten()

        out = {TARGET_COLS[i]: avg[i] for i in range(4)}
        out["Dry_Dead_g"] = max(0.0, out["Dry_Total_g"] - out["GDM_g"])
        results[p] = out

# ------------------------
# SUBMISSION
# ------------------------
test_df["target"] = [
    max(0.0, results[r["image_path"]][r["target_name"]])
    for _, r in test_df.iterrows()
]
test_df[["sample_id", "target"]].to_csv("submission.csv", index=False)
print("Complete! submission.csv created.")