import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from collections.abc import Sequence


class AttentionFusion(nn.Module):
    def __init__(self, img_dim: int, tab_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query_layer = nn.Linear(tab_dim, hidden_dim)
        self.key_layer = nn.Linear(img_dim, hidden_dim)
        self.value_layer = nn.Linear(img_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img_feat, tab_feat):
        query = self.query_layer(tab_feat).unsqueeze(1)        # (B, 1, H)
        key = self.key_layer(img_feat).unsqueeze(1)            # (B, 1, H)
        value = self.value_layer(img_feat).unsqueeze(1)        # (B, 1, H)

        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)  # (B, 1, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1) + 1e-6                      # (B, 1, 1)

        fused = torch.bmm(attn_weights, value).squeeze(1)      # (B, H)
        return self.output_layer(fused)                        # (B, H)


class MultiModalMultiTaskModel(nn.Module):
    def __init__(self,
                 img_size: Sequence[int] | int,
                 in_channels: int,
                 seg_classes: int,
                 cls_classes: int,
                 tabular_dim: int,
                 feature_size: int,
                 hidden_dim: int,
                 norm_name: tuple | str = 'instance',
                 use_v2: bool = False
                 ) -> None:
        super().__init__()

        # Backbone for segmentation
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=seg_classes,
            feature_size=feature_size,
            norm_name=norm_name,
            use_v2=use_v2,
        )

        # Pool encoder feature
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # 🧪 Tự động suy ra img_dim từ đầu ra của backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            encoder_out = self.backbone.swinViT(dummy_input)
            x = encoder_out[-1]
            dummy_feat = self.global_pool(x).view(1, -1)
            img_dim = dummy_feat.shape[1]  # ví dụ: 768

        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.GELU()
        )

        # Attention fusion
        self.fusion = AttentionFusion(img_dim=img_dim, tab_dim=64, hidden_dim=hidden_dim)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cls_classes)
        )

    def forward(self, image, tabular):
        # Segmentation output
        seg_out = self.backbone(image)  # (B, seg_classes, D, H, W)

        # Trích xuất feature từ SwinViT
        encoder_out = self.backbone.swinViT(image)  # List[Tensor]
        x = encoder_out[-1]  # Lấy feature map sâu nhất
        img_feat = self.global_pool(x).view(x.size(0), -1)  # (B, C)

        # Tabular feature
        tab_feat = self.tabular_encoder(tabular)  # (B, 64)

        # Fusion
        fused_feat = self.fusion(img_feat, tab_feat)

        # Classification
        cls_input = torch.cat([fused_feat, tab_feat], dim=1)
        cls_out = self.cls_head(cls_input)

        return {
            "seg": seg_out,
            "cls": cls_out
        }
