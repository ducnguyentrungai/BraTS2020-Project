from metrics import Metric
import torch
from tqdm import tqdm
from typing import List, Optional
from monai.data import DataLoader

def Evaluate_Model(model, num_classes: int, data_loader: DataLoader, loss_fn, device,
                   include_background: bool = True, classification_loss_fn=None):
    model.eval()
    metric = Metric(num_classes=num_classes, include_background=include_background)

    total_loss = total_seg_loss = total_cls_loss = 0.0
    total_iou = total_dice = total_sensi = total_speci = total_seg_acc = 0.0
    total_cls_acc = 0.0

    num_batches = len(data_loader)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating', colour='blue')
        for batch in pbar:
            images = batch['image'].to(device).float()               # [B, C, D, H, W]
            seg_labels = batch['label'].to(device).long()            # [B, D, H, W]
            cls_labels = batch['cls_label'].to(device).long()        # [B] or [B, 1]

            # === Dự đoán ===
            seg_output, cls_output = model(images)                   # Tuple: ([B, C, D, H, W], [B, num_cls])

            # === Loss ===
            seg_loss = loss_fn(seg_output, seg_labels)
            cls_loss = classification_loss_fn(cls_output, cls_labels)
            loss = seg_loss + cls_loss  # Tổng loss nếu không dùng alpha/beta

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()

            # === Metric segmentation ===
            total_iou += metric.IoU(seg_output, seg_labels)
            total_dice += metric.Dice(seg_output, seg_labels)
            total_sensi += metric.Sensitivity(seg_output, seg_labels)
            total_speci += metric.Specificity(seg_output, seg_labels)
            total_seg_acc += metric.Accuracy(seg_output, seg_labels)

            # === Metric classification ===
            preds_cls = torch.argmax(cls_output, dim=1)
            acc_cls = (preds_cls == cls_labels).float().mean().item()
            total_cls_acc += acc_cls

            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Dice': f'{total_dice / (pbar.n+1):.3f}',
                'Cls_Acc': f'{acc_cls:.3f}'
            })

    # Trung bình các metric
    avg_metrics = {
        'total_loss': total_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'sensitivity': total_sensi / num_batches,
        'specificity': total_speci / num_batches,
        'seg_accuracy': total_seg_acc / num_batches,
        'cls_accuracy': total_cls_acc / num_batches,
    }

    return avg_metrics
