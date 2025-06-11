import torch

class Metric:
    def __init__(self, num_classes: int, threshold: float = 0.5, include_background: bool = True):
        self.num_classes = num_classes
        self.threshold = threshold
        self.include_background = include_background

    def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.ndim != 5:
            raise ValueError(f"Expected preds to be 5D (B, C, D, H, W), got {preds.shape}")
        if self.num_classes == 1:
            return (preds > self.threshold).long().squeeze(1)
        return torch.argmax(preds, dim=1)

    def _compute_confusion(self, preds: torch.Tensor, targets: torch.Tensor, cls: int):
        preds = self._prepare_preds(preds)
        targets = targets.to(preds.device).long()

        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

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
            val = compute_fn(TP, FP, FN, TN)
            values.append(val.item())
        return sum(values) / len(values) if values else 0.0

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



# from typing import List, Optional
# import torch

# class Metric:
#     def __init__(self, num_classes: int, threshold: float = 0.5, ignore_labels: Optional[int] = None):
#         self.num_classes = num_classes
#         self.threshold = threshold
#         self.ignore_index = ignore_labels

#     def _to_device(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
#         return tensor.to(reference.device)

#     def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
#         # (B, C, D, H, W) -> (B, D, H, W)
#         if preds.ndim != 5:
#             raise ValueError(f"Expected preds to be 5D, got {preds.shape}")
#         if self.num_classes == 1:
#             return (preds > self.threshold).long().squeeze(1)
#         return torch.argmax(preds, dim=1)

#     def _compute_confusion(self, preds: torch.Tensor, targets: torch.Tensor, cls: int):
#         preds = self._prepare_preds(preds)
#         targets = self._to_device(targets.long(), preds)

#         if targets.ndim == 5 and targets.shape[1] == 1:
#             targets = targets.squeeze(1)

#         if self.ignore_index is not None:
#             mask = targets != self.ignore_index
#             preds = preds[mask]
#             targets = targets[mask]

#         pred_cls = (preds == cls).float()
#         target_cls = (targets == cls).float()

#         TP = (pred_cls * target_cls).sum()
#         FP = (pred_cls * (1 - target_cls)).sum()
#         FN = ((1 - pred_cls) * target_cls).sum()
#         TN = ((1 - pred_cls) * (1 - target_cls)).sum()
#         return TP, FP, FN, TN

#     def _filter_classes(self, values: List[float], selected_classes: List[int]) -> float:
#         if selected_classes:
#             filtered = [value for idx, value in enumerate(values) if idx in selected_classes]
#             return sum(filtered) / len(filtered) if filtered else 0.0
#         return sum(values) / len(values) if values else 0.0

#     def IoU(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
#         ious = []
#         for cls in range(self.num_classes):
#             if cls == self.ignore_index:
#                 continue
#             TP, FP, FN, _ = self._compute_confusion(preds, targets, cls)
#             iou = TP / (TP + FP + FN + 1e-6)
#             ious.append(iou.item())
#         return self._filter_classes(ious, selected_classes)

#     def Dice(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
#         dices = []
#         for cls in range(self.num_classes):
#             if cls == self.ignore_index:
#                 continue
#             TP, FP, FN, _ = self._compute_confusion(preds, targets, cls)
#             dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
#             dices.append(dice.item())
#         return self._filter_classes(dices, selected_classes)

#     def Accuracy(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
#         accs = []
#         for cls in range(self.num_classes):
#             if cls == self.ignore_index:
#                 continue
#             TP, FP, FN, TN = self._compute_confusion(preds, targets, cls)
#             acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
#             accs.append(acc.item())
#         return self._filter_classes(accs, selected_classes)

#     def Sensitivity(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
#         recalls = []
#         for cls in range(self.num_classes):
#             if cls == self.ignore_index:
#                 continue
#             TP, _, FN, _ = self._compute_confusion(preds, targets, cls)
#             recall = TP / (TP + FN + 1e-6)
#             recalls.append(recall.item())
#         return self._filter_classes(recalls, selected_classes)

#     def Specificity(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
#         specs = []
#         for cls in range(self.num_classes):
#             if cls == self.ignore_index:
#                 continue
#             _, FP, _, TN = self._compute_confusion(preds, targets, cls)
#             spec = TN / (TN + FP + 1e-6)
#             specs.append(spec.item())
#         return self._filter_classes(specs, selected_classes)