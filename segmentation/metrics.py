from typing import List, Optional
import torch

class Metric:
    def __init__(self, num_classes: int, threshold: float = 0.5, ignore_index: Optional[int] = None):
        """
        Initialize the metric class for multi-class segmentation evaluation.

        Args:
            num_classes (int): Number of segmentation classes.
            threshold (float): Threshold for binary segmentation (used when num_classes = 1).
            ignore_index (Optional[int]): Class label to be ignored in metric calculations.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.ignore_index = ignore_index

    def _to_device(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the same device as a reference tensor."""
        return tensor.to(reference.device)

    def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Convert raw model outputs to discrete label predictions.

        For binary: apply threshold.
        For multi-class: apply argmax across channels.

        Args:
            preds (Tensor): Raw output tensor from the model, shape (B, C, D, H, W).

        Returns:
            Tensor: Predicted class labels, shape (B, D, H, W).
        """
        if preds.ndim != 5:
            raise ValueError(f"Expected preds to be 5D, got {preds.shape}")
        if self.num_classes == 1:
            return (preds > self.threshold).long().squeeze(1)
        return torch.argmax(preds, dim=1)

    def _compute_confusion(self, preds: torch.Tensor, targets: torch.Tensor, cls: int):
        """
        Compute TP, FP, FN, TN for a specific class.

        Args:
            preds (Tensor): Raw predictions or logits.
            targets (Tensor): Ground truth labels.
            cls (int): Class index to compute metrics for.

        Returns:
            Tuple of (TP, FP, FN, TN) as torch tensors.
        """
        preds = self._prepare_preds(preds)
        targets = self._to_device(targets.long(), preds)

        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        TP = (pred_cls * target_cls).sum()
        FP = (pred_cls * (1 - target_cls)).sum()
        FN = ((1 - pred_cls) * target_cls).sum()
        TN = ((1 - pred_cls) * (1 - target_cls)).sum()
        return TP, FP, FN, TN

    def _filter_classes(self, values: List[float], selected_classes: List[int]) -> float:
        """
        Filter and average metric values across selected classes.

        Args:
            values (List[float]): Metric values for all classes.
            selected_classes (List[int]): Class indices to include.

        Returns:
            float: Averaged metric over selected classes.
        """
        if selected_classes:
            filtered = [value for idx, value in enumerate(values) if idx in selected_classes]
            return sum(filtered) / len(filtered) if filtered else 0.0
        return sum(values) / len(values) if values else 0.0

    def IoU(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
        """
        Compute mean Intersection over Union (IoU) over classes.

        Returns:
            float: Mean IoU.
        """
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            TP, FP, FN, _ = self._compute_confusion(preds, targets, cls)
            iou = TP / (TP + FP + FN + 1e-6)
            ious.append(iou.item())
        return self._filter_classes(ious, selected_classes)

    def Dice(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
        """
        Compute mean Dice coefficient over classes.

        Returns:
            float: Mean Dice score.
        """
        dices = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            TP, FP, FN, _ = self._compute_confusion(preds, targets, cls)
            dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
            dices.append(dice.item())
        return self._filter_classes(dices, selected_classes)

    def Accuracy(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
        """
        Compute mean accuracy over classes.

        Returns:
            float: Mean accuracy.
        """
        accs = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            TP, FP, FN, TN = self._compute_confusion(preds, targets, cls)
            acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
            accs.append(acc.item())
        return self._filter_classes(accs, selected_classes)

    def Sensitivity(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
        """
        Compute mean sensitivity (recall) over classes.

        Returns:
            float: Mean sensitivity.
        """
        recalls = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            TP, _, FN, _ = self._compute_confusion(preds, targets, cls)
            recall = TP / (TP + FN + 1e-6)
            recalls.append(recall.item())
        return self._filter_classes(recalls, selected_classes)

    def Specificity(self, preds: torch.Tensor, targets: torch.Tensor, selected_classes: List[int] = []) -> float:
        """
        Compute mean specificity over classes.

        Returns:
            float: Mean specificity.
        """
        specs = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            _, FP, _, TN = self._compute_confusion(preds, targets, cls)
            spec = TN / (TN + FP + 1e-6)
            specs.append(spec.item())
        return self._filter_classes(specs, selected_classes)
