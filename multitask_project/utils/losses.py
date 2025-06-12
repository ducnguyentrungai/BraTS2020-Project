import torch
import torch.nn as nn
from monai.losses import DiceLoss

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function combining segmentation and classification losses.
    Supports flexible weighting and modular loss components.

    The total loss is computed as:
        total_loss = seg_loss + loss_weight * cls_loss

    where:
        - seg_loss: loss for segmentation task (e.g., DiceCELoss)
        - cls_loss: loss for classification task (e.g., CrossEntropyLoss)
        - loss_weight: balancing coefficient for classification loss
    """
    def __init__(self,
                 loss_seg: nn.Module = None,
                 loss_cls: nn.Module = None,
                 loss_weight: float = 1.0):
        super().__init__()

        # Default segmentation loss: Dice + CrossEntropy (for multi-class seg)
        self.loss_seg = loss_seg if loss_seg is not None else DiceLoss(to_onehot_y=True, softmax=True)

        # Default classification loss: CrossEntropy
        self.loss_cls = loss_cls if loss_cls is not None else nn.CrossEntropyLoss()

        # Weight to balance between tasks
        self.loss_weight = loss_weight

    def forward(self,
                seg_pred: torch.Tensor,
                seg_target: torch.Tensor,
                cls_pred: torch.Tensor,
                cls_target: torch.Tensor) -> dict:
        """
        Compute multi-task loss.

        Args:
            seg_pred: (B, C, D, H, W) predicted segmentation logits
            seg_target: (B, D, H, W) ground truth segmentation labels
            cls_pred: (B, num_cls) predicted classification logits
            cls_target: (B,) ground truth class indices

        Returns:
            A dictionary with keys: 'loss', 'loss_seg', 'loss_cls'
        """
        loss_seg = self.loss_seg(seg_pred, seg_target)
        loss_cls = self.loss_cls(cls_pred, cls_target)
        total_loss = loss_seg + self.loss_weight * loss_cls

        return {
            "loss": total_loss,
            "loss_seg": loss_seg,
            "loss_cls": loss_cls
        }
