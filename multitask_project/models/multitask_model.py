import torch
import torch.nn as nn
from typing import Dict, Sequence, Union
from monai.networks.nets import SwinUNETR

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        att = torch.sigmoid(self.conv(x))  # [B,1,D,H,W]
        return x * att

class CrossModalityAttention(nn.Module):
    """
    A simpler dot-product attention over embeddings.
    """
    def __init__(self, img_dim, tab_dim):
        super().__init__()
        self.query = nn.Linear(tab_dim, img_dim)
        self.key = nn.Linear(img_dim, img_dim)
        self.value = nn.Linear(img_dim, img_dim)
        self.scale = torch.sqrt(torch.tensor(img_dim, dtype=torch.float32))
        
    def forward(self, img_feat, tab_feat):
        """
        img_feat: [B, img_dim]
        tab_feat: [B, tab_dim]
        """
        q = self.query(tab_feat)  # [B, img_dim]
        k = self.key(img_feat)    # [B, img_dim]
        v = self.value(img_feat)  # [B, img_dim]

        att_score = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [B,1]
        att_weight = torch.softmax(att_score, dim=1)  # [B,1]
        output = att_weight * v  # [B, img_dim]
        return output

class ImprovedMLPHead(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        conv_channels: Sequence[int] = (256, 512),
        norm_type: str = 'batch',
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_c = in_channels
        
        # Use stride=2 only on first conv to avoid reducing spatial size too much
        for i, c in enumerate(conv_channels):
            stride = 2 if i == 0 else 1
            layers.extend([
                nn.Conv3d(prev_c, c, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm3d(c) if norm_type == 'batch' else nn.InstanceNorm3d(c),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)
            ])
            prev_c = c
        
        self.convs = nn.Sequential(*layers)
        self.attention = SpatialAttention(prev_c)
        self.final_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(prev_c, out_channels)
        
    def forward(self, x):
        x = self.convs(x)
        # Check if spatial size is too small:
        if min(x.shape[2:]) < 1:
            raise ValueError(f"Feature map too small after convs: {x.shape}")
        x = self.attention(x)
        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

class MultiModalMultiTaskModel(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        seg_classes: int,
        cls_classes: int,
        feature_size: int,
        tabular_dim: int,
        img_embedding_dim: int,
        conv_channels: Sequence[int] = (384, 512), 
        norm_name: Union[tuple, str] = 'instance',
        use_v2: bool = False,
        use_checkpoint: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.verbose = verbose

        # Backbone SwinUNETR
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=seg_classes,
            feature_size=feature_size,
            norm_name=norm_name,
            use_v2=use_v2,
            use_checkpoint=use_checkpoint
        )

        # Automatic Channel Detection
        if isinstance(img_size, int):
            dummy_size = (img_size,) * 3
        else:
            dummy_size = tuple(img_size)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *dummy_size)
            encoder_out = self.backbone.swinViT(dummy_input)
            enc_channels = [f.shape[1] for f in encoder_out]
            if self.verbose:
                print(f"Detected encoder output channels: {enc_channels}")
                print(f"Encoder output shapes: {[f.shape for f in encoder_out]}")

        # Handle embedding split to avoid dimension mismatch
        num_scales = len(enc_channels)
        base_dim = img_embedding_dim // num_scales
        remainder = img_embedding_dim - base_dim * num_scales
        split_dims = [base_dim] * num_scales
        split_dims[0] += remainder

        # Multi-scale Image Encoders
        norm_type = "batch" if "batch" in str(norm_name).lower() else "instance"
        self.img_encoders = nn.ModuleList([
            ImprovedMLPHead(
                in_channels=ch,
                out_channels=dim,
                conv_channels=conv_channels,
                norm_type=norm_type,
                dropout=0.2
            )
            for ch, dim in zip(enc_channels, split_dims)
        ])

        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Cross-modality attention
        self.cross_attn = CrossModalityAttention(
            img_dim=img_embedding_dim,
            tab_dim=64
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(img_embedding_dim + 64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, cls_classes)
        )

    def forward(self, image, tabular) -> Dict[str, torch.Tensor]:
        seg_out = self.backbone(image)
        encoder_out = self.backbone.swinViT(image)

        img_feats = []
        for feat, encoder in zip(encoder_out, self.img_encoders):
            # Only apply adaptive pooling if spatial size > 4
            if min(feat.shape[2:]) > 4:
                pooled_feat = nn.functional.adaptive_avg_pool3d(feat, (4, 4, 4))
            else:
                pooled_feat = feat
            img_feats.append(encoder(pooled_feat))

        img_feat = torch.cat(img_feats, dim=1)
        tab_feat = self.tabular_encoder(tabular)
        attn_feat = self.cross_attn(img_feat, tab_feat)

        fusion = torch.cat([attn_feat, tab_feat], dim=1)
        cls_out = self.cls_head(fusion)

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
            elif k.startswith("model.") and not any(sub in k for sub in ["tabular_encoder", "cls_head", "img_encoder"]):
                new_k = k.replace("model.", "")
                new_state_dict[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained SwinUNETR from {ckpt_path}")
        if missing:
            print(f"⚠️ Missing keys: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")
