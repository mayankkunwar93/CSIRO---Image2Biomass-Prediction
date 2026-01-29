import torch
import torch.nn as nn
from transformers import AutoModel

class TeacherNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)

class DinoV3Backbone(nn.Module):
    def __init__(self, model_dir, layer_unfreeze=5):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        for p in self.model.parameters(): p.requires_grad = False
        if layer_unfreeze > 0:
            for p in self.model.layer[-layer_unfreeze:].parameters(): p.requires_grad = True

        self.d_model = self.model.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 4)
        )
        self.embed_dim = self.d_model * 2

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]
        attn_weights = torch.softmax(self.pooler(patch_tokens), dim=1)
        patch_summary = torch.matmul(patch_tokens.transpose(1, 2), attn_weights).mean(dim=-1)
        return torch.cat([cls_token, patch_summary], dim=1)

class DinoHierarchicalRegressor(nn.Module):
    def __init__(self, model_dir, out_features=5):
        super().__init__()
        self.backbone = DinoV3Backbone(model_dir)
        d = self.backbone.embed_dim
        self.crop_attention = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d + d),
            nn.Linear(d + d, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x_rgb, x_crops):
        f_global = self.backbone(x_rgb)
        b, n_c, c, h, w = x_crops.shape
        f_crops = self.backbone(x_crops.view(-1, c, h, w)).view(b, n_c, -1)
        attn_weights = torch.softmax(self.crop_attention(f_crops), dim=1)
        f_crops_weighted = torch.sum(f_crops * attn_weights, dim=1)
        return self.head(torch.cat([f_global, f_crops_weighted], dim=1))