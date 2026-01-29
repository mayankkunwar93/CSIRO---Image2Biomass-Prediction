import torch
import torch.nn as nn

class DinoV2Backbone(nn.Module):
    def __init__(self, weight_path, local_hub_path):
        super().__init__()
        # Load DINOv2 from local path
        self.model = torch.hub.load(
            repo_or_dir=local_hub_path,
            model="dinov2_vits14",
            source="local",
            pretrained=False
        )
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)

        # Freeze all, then unfreeze last 4 blocks
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.blocks[-4:].parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]

class DinoHierarchicalRegressor(nn.Module):
    def __init__(self, weight_path, local_hub_path, out_features=5):
        super().__init__()
        self.backbone = DinoV2Backbone(weight_path, local_hub_path)

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
