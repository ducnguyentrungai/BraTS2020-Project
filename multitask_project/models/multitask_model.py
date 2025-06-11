import torch
import torch.nn as nn
from monai.networks.net import SwinUNETR
from collections.abc import Sequence


class AttentionFustion(nn.Module):
    def __init__(self, img_dim:int, tab_dim:int, hidden_dim:int=128):
        super().__init__()
        self.query_layer = nn.Linear(tab_dim, hidden_dim)
        self.key_layer = nn.Linear(img_dim, hidden_dim)
        self.value_layer = nn.Linear(img_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img_feat, tab_feat):
        # img_feat: (B, C_img), tab_feat: (B, C_tab)
        query = self.query_layer(tab_feat).unsqueeze(1)        # (B, 1, H)
        key = self.key_layer(img_feat).unsqueeze(1)            # (B, 1, H)
        value = self.value_layer(img_feat).unsqueeze(1)        # (B, 1, H)

        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)  # (B, 1, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1) + 1e-6     # (B, 1, 1)

        fused = torch.bmm(attn_weights, value).squeeze(1)      # (B, H)
        return self.output_layer(fused)                        # (B, H)


class MutilModalMutilTaskModel(nn.Module):
    def __init__(self,
                 img_size: Sequence[int] | int,
                 in_channels: int,
                 seg_classes: int,
                 cls_classes: int,
                 tabular_dim: int,
                 feature_size: int,
                 hidden_dim: int,
                 norm_name: tuple | str = 'instance',
                 use_v2 = False
                 ) -> None:
        super.__init__()
        
        self.backbone = SwinUNETR(
            img_size = img_size,
            in_channels = in_channels,
            out_channels = seg_classes,
            norm_name= norm_name,
            use_v2 = use_v2,
        )
        # Global Average Pooling layer to extract image features from ViT
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        # Attention-bases funsion
        self.fution = AttentionFustion(img_dim=feature_size * 8, tab_dim=tabular_dim, hidden_dim=hidden_dim)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(feature_size * 8 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cls_classes)
        )

    def forward(self, image, tabular) -> dict:
        # Segmentation output
        seg_out = self.backbone(image)

        # Extract bottleneck features from ViT encoder
        image_feat = self.backbone.swinViT.forward_features(image)["x"]
        image_feat = self.global_pool(image_feat)  # shape: (B, C, 1, 1, 1)
        image_feat = image_feat.view(image_feat.size(0), -1)  # shape: (B, C)

        # Encode tabular data
        tab_feat = self.tabular_encoder(tabular)  # shape: (B, 64)

        # Feature fusion with attention
        fused_feat = self.fution(image_feat, tab_feat)

        # Classification output
        cls_out = self.cls_head(fused_feat)  # shape: (B, num_cls)

        return {
            "seg": seg_out,
            "cls": cls_out
        }
