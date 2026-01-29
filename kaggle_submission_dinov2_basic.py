import os, cv2, torch, numpy as np, pandas as pd, warnings
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

# ------------------------
# SETTINGS & DEVICE
# ------------------------
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "/kaggle/input/csiro-biomass/train.csv"
TEST_CSV = "/kaggle/input/csiro-biomass/test.csv"
TEST_IMG_DIR = "/kaggle/input/csiro-biomass/test"
DINO_REPO_DIR = "/kaggle/input/dinov2-repo-source/dinov2"
MODEL_WEIGHTS_DIR = "/kaggle/input/main-dino2s14-folds-img980-bestwr2"

IMG_SIZE = 980
targets_to_use = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

# ------------------------------------------------------------
# CV WEIGHTS
# ------------------------------------------------------------
fold_weights = {1: 0.7492, 2: 0.7568, 3: 0.8101, 4: 0.7531, 5:0.7013}  # WR2-based weighting

# ------------------------
# TARGET STATS & DENORM
# ------------------------

df_train = pd.read_csv(TRAIN_CSV)
df_train = df_train[df_train["target_name"].isin(targets_to_use)]
df_pivot = df_train.pivot(index="image_path", columns="target_name", values="target").reset_index()

targets_log = np.log1p(df_pivot[targets_to_use].values.astype(np.float32))  # log-space targets
t_mean = targets_log.mean(axis=0)
t_std = targets_log.std(axis=0) + 1e-8

def denormalize(y):
    # inverse of training-time normalization
    return np.expm1(y * t_std + t_mean)

# ------------------------
# MODEL ARCHITECTURE (ViT-S/14)
# ------------------------
class DinoV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir=DINO_REPO_DIR,
            model="dinov2_vits14",
            source="local",
            pretrained=False
        )
    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]  # CLS token

class DinoHierarchicalRegressor(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = DinoV2Backbone()
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
        return self.head(self.backbone(x))

# ------------------------
# LOAD ENSEMBLE
# ------------------------
ensemble = []
print("Loading ViT-S 5-Fold Ensemble...")
for fold in range(1, 6):
    path = f"{MODEL_WEIGHTS_DIR}/main_dino2_fold{fold}_img{IMG_SIZE}_bestWR2.pth"
    if os.path.exists(path):
        m = DinoHierarchicalRegressor().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        ensemble.append((m, fold_weights[fold]))
        print(f"-> Fold {fold} loaded.")

# ------------------------
# INFERENCE PIPELINE
# ------------------------
test_df = pd.read_csv(TEST_CSV)
unique_paths = test_df['image_path'].unique()
img_results = {}  # per-image cached predictions

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with torch.no_grad():
    for path in tqdm(unique_paths, desc="Processing Test Images"):
        img_name = os.path.basename(path)
        full_path = os.path.join(TEST_IMG_DIR, img_name)

        img_bgr = cv2.imread(full_path)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        weighted_sum = 0
        total_w = 0
        for model, w in ensemble:
            p = model(input_tensor).cpu().numpy()   # normalized log-space
            weighted_sum += (denormalize(p) * w)    # back to grams + weight
            total_w += w

        img_results[path] = (weighted_sum / total_w).flatten()

# ------------------------
# SUBMISSION GENERATION
# ------------------------
submission = []
for _, row in test_df.iterrows():
    p_path = row['image_path']
    t_name = row['target_name']
    t_idx = targets_to_use.index(t_name)

    val = max(0.0, img_results[p_path][t_idx]) if p_path in img_results else 0.0

    s_id = f"{os.path.splitext(os.path.basename(p_path))[0]}__{t_name}"
    submission.append({"sample_id": s_id, "target": val})

pd.DataFrame(submission).to_csv("submission.csv", index=False)
print("Complete! submission.csv created.")