import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class UNETRMultitaskWithTabular(nn.Module):
    def __init__(self, in_channels:int, out_seg_channels:int, out_cls_classes:int,
                 img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                 norm_name='instance', conv_block=True, res_block=True,
                 dropout_rate=0.0, spatial_dims=3, qkv_bias=False, 
                 tabular_dim=0, classifier_hidden_dim=[256, 128 , 64]):
        super().__init__()
        # Initialize UNETR backbone for 3D segmentation
        self.unetr = UNETR(
            in_channels=in_channels, out_channels=out_seg_channels,
            img_size=img_size, feature_size=feature_size, hidden_size=hidden_size, 
            mlp_dim=mlp_dim, num_heads=num_heads,
            norm_name=norm_name, conv_block=conv_block, res_block=res_block,
            dropout_rate=dropout_rate, spatial_dims=spatial_dims, qkv_bias=qkv_bias
        )
        # Global average pooling to get a single feature vector from segmentation output
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Define classification MLP head
        classifier_input_dim = out_seg_channels + tabular_dim
        # MLP classifier xây dựng từ danh sách hidden layers
        if classifier_hidden_dim is not None and isinstance(classifier_hidden_dim, list):
            layers = []
            input_dim = classifier_input_dim  # ban đầu là out_seg_channels + tabular_dim
            for hidden_dim in classifier_hidden_dim:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p=0.1))
                input_dim = hidden_dim  # cập nhật cho lớp kế tiếp

            layers.append(nn.Linear(input_dim, out_cls_classes))  # output layer
            self.classifier = nn.Sequential(*layers)
            
        else:
            # fallback: 1 lớp nếu không có danh sách
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, out_cls_classes)
            )
            
    def forward(self, image, tabular):
        seg_output = self.unetr(image)  # (B, C, D, H, W)
        pooled_features = self.global_pool(seg_output).view(seg_output.size(0), -1)

        # Đảm bảo tabular có đúng shape, device và dtype
        if tabular.ndim == 1:
            tabular = tabular.unsqueeze(1)
        tabular = tabular.to(device=pooled_features.device, dtype=pooled_features.dtype)

        combined_features = torch.cat((pooled_features, tabular), dim=1)
        cls_output = self.classifier(combined_features)
        return seg_output, cls_output


if __name__ == "__main__":
    model = UNETRMultitaskWithTabular(
        in_channels=1, out_seg_channels=4, out_cls_classes=3,
        img_size=(96, 96, 96), tabular_dim=8
    )

    x = torch.randn(2, 1, 96, 96, 96)  # 2 ảnh
    t = torch.randn(2, 8)  # 2 dòng tabular
    seg, cls = model(x, t)

    print("Seg out:", seg.shape)  # expect: (2, 4, 96, 96, 96)
    print("Cls out:", cls.shape)  # expect: (2, 3)
