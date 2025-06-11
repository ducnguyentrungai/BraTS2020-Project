import torch
import torch.nn as nn
from monai.networks.net import SwinUNETR
from collections.abc import Sequence


class MutilModalMutilTaskModel(nn.Module):
    def __init__(self,
                 img_size: Sequence[int] | int,
                 in_channels: int,
                 seg_classes: int,
                 cls_classes: int,
                 tabular_dim: int,
                 feature_size: int,
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

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(feature_size * 8 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cls_classes)
        )

    def forward(self, image, tabular):
        # Segmentation output
        seg_out = self.backbone(image)

        # Extract bottleneck features from ViT encoder
        image_feat = self.backbone.swinViT.forward_features(image)["x"]
        image_feat = self.global_pool(image_feat)  # shape: (B, C, 1, 1, 1)
        image_feat = image_feat.view(image_feat.size(0), -1)  # shape: (B, C)

        # Encode tabular data
        tab_feat = self.tabular_encoder(tabular)  # shape: (B, 64)

        # Feature fusion
        fusion = torch.cat([image_feat, tab_feat], dim=1)  # shape: (B, C+64)

        # Classification output
        cls_out = self.cls_head(fusion)  # shape: (B, num_cls)

        return {
            "seg": seg_out,
            "cls": cls_out
        }
