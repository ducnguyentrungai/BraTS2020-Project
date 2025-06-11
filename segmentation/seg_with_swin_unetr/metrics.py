import torch
from typing import Optional

class Metric:
    def __init__(self, num_classes: int, threshold: float = 0.5, include_background: bool = True):
        self.num_classes = num_classes
        self.threshold = threshold
        self.include_background = include_background
    
    def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.ndim != 5:
            raise ValueError(f"Expected preds to be 5D (B, C, D, H, W), got {preds.shape}")
        if self.num_classes == 1:
            return (preds > self.threshold).long().view(preds.shape[0], *preds.shape[2:])
        return torch.argmax(preds, dim=1)

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 5:
            if targets.shape[1] > 1:  # one-hot encoded
                targets = torch.argmax(targets, dim=1)
            elif targets.shape[1] == 1:
                targets = targets.squeeze(1)
        return targets.long()

    def _compute_confusion(self, preds: torch.Tensor, targets: torch.Tensor, cls: int):
        preds = self._prepare_preds(preds)
        targets = self._prepare_targets(targets).to(preds.device)

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        TP = (pred_cls * target_cls).sum()
        FP = (pred_cls * (1 - target_cls)).sum()
        FN = ((1 - pred_cls) * target_cls).sum()
        TN = ((1 - pred_cls) * (1 - target_cls)).sum()
        return TP, FP, FN, TN

    def _loop_classes(self, preds, targets, compute_fn):
        start_cls = 0 if self.include_background else 1
        values = []
        for cls in range(start_cls, self.num_classes):
            TP, FP, FN, TN = self._compute_confusion(preds, targets, cls)
            denom = TP + FP + FN + TN
            if denom > 0:  # tránh chia 0 vô nghĩa
                val = compute_fn(TP, FP, FN, TN)
                values.append(val.detach().cpu().item())
        return sum(values) / len(values) if values else 0.0


    # def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
    #     if preds.ndim != 5:
    #         raise ValueError(f"Expected preds to be 5D (B, C, D, H, W), got {preds.shape}")
    #     if self.num_classes == 1:
    #         return (preds > self.threshold).long().squeeze(1)
    #     return torch.argmax(preds, dim=1)

    # def _compute_confusion(self, preds: torch.Tensor, targets: torch.Tensor, cls: int):
    #     preds = self._prepare_preds(preds)
    #     targets = targets.to(preds.device).long()

    #     if targets.ndim == 5 and targets.shape[1] == 1:
    #         targets = targets.squeeze(1)

    #     pred_cls = (preds == cls).float()
    #     target_cls = (targets == cls).float()

    #     TP = (pred_cls * target_cls).sum()
    #     FP = (pred_cls * (1 - target_cls)).sum()
    #     FN = ((1 - pred_cls) * target_cls).sum()
    #     TN = ((1 - pred_cls) * (1 - target_cls)).sum()
    #     return TP, FP, FN, TN

    # def _loop_classes(self, preds, targets, compute_fn):
    #     start_cls = 0 if self.include_background else 1
    #     values = []
    #     for cls in range(start_cls, self.num_classes):
    #         TP, FP, FN, TN = self._compute_confusion(preds, targets, cls)
    #         val = compute_fn(TP, FP, FN, TN)
    #         values.append(val.item())
    #     return sum(values) / len(values) if values else 0.0

    def IoU(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return self._loop_classes(preds, targets, lambda TP, FP, FN, _: TP / (TP + FP + FN + 1e-6))

    def Dice(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return self._loop_classes(preds, targets, lambda TP, FP, FN, _: 2 * TP / (2 * TP + FP + FN + 1e-6))

    def Accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return self._loop_classes(preds, targets, lambda TP, FP, FN, TN: (TP + TN) / (TP + TN + FP + FN + 1e-6))

    def Sensitivity(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return self._loop_classes(preds, targets, lambda TP, _, FN, __: TP / (TP + FN + 1e-6))

    def Specificity(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return self._loop_classes(preds, targets, lambda _, FP, __, TN: TN / (TN + FP + 1e-6))
