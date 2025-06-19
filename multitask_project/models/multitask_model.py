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


class MultiHeadFusion(nn.Module):
    def __init__(self, img_dim: int, tab_dim: int, fused_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(tab_dim, fused_dim)
        self.key_proj = nn.Linear(img_dim, fused_dim)
        self.value_proj = nn.Linear(img_dim, fused_dim)

        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU()
        )

    def forward(self, img_feat, tab_feat):
        # img_feat: (B, C_img), tab_feat: (B, C_tab)
        query = self.query_proj(tab_feat).unsqueeze(1)  # (B, 1, D)
        key = self.key_proj(img_feat).unsqueeze(1)      # (B, 1, D)
        value = self.value_proj(img_feat).unsqueeze(1)  # (B, 1, D)

        attn_output, _ = self.attn(query, key, value)   # (B, 1, D)
        fused = self.out_proj(attn_output.squeeze(1))   # (B, D)
        return fused


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
            dummy_input = torch.zeros(1, in_channels, * img_size)
            encoder_out = self.backbone.swinViT(dummy_input)
            x = encoder_out[-1]
            dummy_feat = self.global_pool(x).view(1, -1)
            img_dim = dummy_feat.shape[1]

        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),  # Không phụ thuộc batch size
            nn.Linear(64, 64),
            nn.GELU()
        )

        # Attention fusion
        # self.fusion = AttentionFusion(img_dim=img_dim, 
        #                               tab_dim=64, 
        #                               hidden_dim=hidden_dim)
        self.fusion = MultiHeadFusion(
                        img_dim=img_dim,
                        tab_dim=64,
                        fused_dim=hidden_dim,
                        num_heads=4  # bạn có thể chọn 2 hoặc 8 tùy kích thước
                    )


        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cls_classes)
        )

    # def forward(self, image, tabular):
    #     seg_out = self.backbone(image)  # (B, seg_classes, D, H, W)

    #     encoder_out = self.backbone.swinViT(image)
    #     x = encoder_out[-1]
    #     img_feat = self.global_pool(x).view(x.size(0), -1)

    #     tab_feat = self.tabular_encoder(tabular)
    #     fused_feat = self.fusion(img_feat, tab_feat)
    #     cls_input = torch.cat([fused_feat, tab_feat], dim=1)
    #     cls_out = self.cls_head(cls_input)

    #     return {
    #         "seg": seg_out,
    #         "cls": cls_out
    #     }
    
    def forward(self, image, tabular):
        # Segmentation output
        seg_out = self.backbone(image)  # (B, seg_classes, D, H, W)

        # Trích xuất đặc trưng ảnh từ ViT
        encoder_out = self.backbone.swinViT(image)
        x = encoder_out[-1]  # lấy đầu ra cuối cùng
        img_feat = self.global_pool(x).view(x.size(0), -1)  # (B, C_img)

        # Tabular encoding
        tab_feat = self.tabular_encoder(tabular)  # (B, 64)

        # Fusion giữa ảnh và bảng
        fused_feat = self.fusion(img_feat, tab_feat)  # (B, hidden_dim)

        # Kết hợp fused với tab_feat để classification
        cls_input = torch.cat([fused_feat, tab_feat], dim=1)  # (B, hidden_dim + 64)
        cls_out = self.cls_head(cls_input)  # (B, num_classes)

        return {
            "seg": seg_out,
            "cls": cls_out
        }


    def load_pretrained_segmentation(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.backbone."):
                new_k = k.replace("model.backbone.", "")
                new_state_dict[new_k] = v
            elif k.startswith("backbone."):
                new_k = k.replace("backbone.", "")
                new_state_dict[new_k] = v
            elif k.startswith("model.") and not any(sub in k for sub in ["tabular_encoder", "cls_head"]):
                new_k = k.replace("model.", "")
                new_state_dict[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded pretrained SwinUNETR from {ckpt_path}")
        if missing:
            print(f"⚠️ Missing keys: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")
