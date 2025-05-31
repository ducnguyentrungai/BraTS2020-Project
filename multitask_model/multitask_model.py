import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class UNETRMultitaskWithTabular(nn.Module):
    def __init__(self, in_channels:int, out_seg_channels:int, out_cls_classes:int,
                 img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                 norm_name='instance', conv_block=True, res_block=True,
                 dropout_rate=0.0, spatial_dims=3, qkv_bias=False, 
                 tabular_dim=0, classifier_hidden_dim=[256, 128 , 64, 7]):
        """
        Multi-task model combining a UNETR for 3D image segmentation with a classification head for tabular data.
        It outputs a segmentation map and a classification prediction.
        
        Args:
            in_channels (int): Number of channels in the input image.
            out_seg_channels (int): Number of channels for segmentation output (e.g., number of segmentation classes).
            out_cls_classes (int): Number of classes for the classification output.
            img_size (sequence or int): Input image size. If int, the image is assumed cubic with each dimension equal to img_size.
            feature_size (int): Feature size for the UNETR model (default 16).
            hidden_size (int): Hidden size of the Vision Transformer in UNETR (default 768).
            mlp_dim (int): MLP dimension in the Vision Transformer (default 3072).
            num_heads (int): Number of attention heads in the Vision Transformer (default 12).
            pos_embed (str): Position embedding type for UNETR (default 'conv').
            norm_name (str): Normalization type for UNETR (default 'instance').
            conv_block (bool): Use convolutional blocks in decoder (default True).
            res_block (bool): Use residual blocks in decoder (default True).
            dropout_rate (float): Dropout rate in ViT (default 0.0).
            spatial_dims (int): Number of spatial dimensions of the input (default 3 for 3D).
            qkv_bias (bool): Use bias in QKV linear layers of ViT (default False).
            tabular_dim (int): Dimension of additional tabular data features. If 0, no tabular data is used.
            classifier_hidden_dim (int): Hidden dimension for the classifier MLP. If >0, a two-layer MLP is used; if 0 or None, use single linear layer.
        """
        super().__init__()
        # Initialize UNETR backbone for 3D segmentation
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_seg_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate,
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
        """
        Forward pass for the multi-task model.
        
        Args:
            image (torch.Tensor): 3D image tensor of shape (B, in_channels, D, H, W).
            tabular (torch.Tensor): Tabular data tensor of shape (B, tabular_dim).
        
        Returns:
            torch.Tensor: Segmentation output of shape (B, out_seg_channels, D, H, W).
            torch.Tensor: Classification output of shape (B, out_cls_classes).
        """
        # Segmentation output from UNETR
        seg_output = self.unetr(image)  # (B, out_seg_channels, D, H, W)
        # Global average pooling over the segmentation output (spatial dimensions) to get feature vector
        pooled_features = self.global_pool(seg_output)  # (B, out_seg_channels, 1, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, out_seg_channels)
        # Concatenate pooled image features with tabular data
        combined_features = torch.cat((pooled_features, tabular), dim=1)  # (B, out_seg_channels + tabular_dim)
        # Classification prediction from MLP
        cls_output = self.classifier(combined_features)  # (B, out_cls_classes)
        return seg_output, cls_output
