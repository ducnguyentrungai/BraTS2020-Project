from metrics import Metric
import torch
from tqdm import tqdm


def Evaluate_Model(model, data_loader, loss_fn, device, selected_classes:list=[]):
    model.eval()
    metric = Metric(num_classes=4)  # Cập nhật đúng số lớp segmentation
    total_loss = total_iou = total_dice = total_sensi = total_speci = total_acc = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating', colour='blue')
        for batch in pbar:
            images = batch['image'].to(device).float()          # [B, C, D, H, W]
            labels = batch['label'].to(device).long()            # [B, D, H, W]

            outputs = model(images)                             # [B, C, D, H, W]
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            

            total_iou   += metric.IoU(outputs, labels, selected_classes)
            total_dice  += metric.Dice(outputs, labels, selected_classes)
            total_sensi += metric.Sensitivity(outputs, labels, selected_classes)
            total_speci += metric.Specificity(outputs, labels, selected_classes)
            total_acc   += metric.Accuracy(outputs, labels, selected_classes)

            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',           
                'IoU': f'{metric.IoU(outputs, labels, selected_classes):.3f}',
                'Dice': f'{metric.Dice(outputs, labels, selected_classes):.3f}'
            })

    # Trung bình kết quả
    avg_loss  = total_loss / num_batches
    avg_iou   = total_iou / num_batches
    avg_dice  = total_dice / num_batches
    avg_sensi = total_sensi / num_batches
    avg_speci = total_speci / num_batches
    avg_acc   = total_acc / num_batches

    return avg_loss, avg_iou, avg_dice, avg_sensi, avg_speci, avg_acc