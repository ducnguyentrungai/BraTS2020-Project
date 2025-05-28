from metrics import Metric
import torch
from tqdm import tqdm


def Evaluate_Model(model, data_loader, criterion1, criterion2, device, selected_classes:list=[]):
    model.eval()
    num_data_batch = len(data_loader)
    metric = Metric()
    with torch.no_grad():
        run_loss = 0.0
        run_iou = 0.0 
        run_dice = 0.0
        run_recal = 0.0
        run_spec = 0.0
        run_acc = 0.0
        pbar = tqdm(data_loader, desc='testing')
        for batch  in pbar:
            image = batch['image'].to(device).float()          # (B, 3, D, H, W)
            mask = batch['mask'].squeeze(1).to(device).long()  # (B, D, H, W)

            mask_hat = model(image)
            loss1 = criterion1(mask_hat, mask)
            loss2 = criterion2(mask_hat, mask)
            loss = loss1 + loss2
            run_loss += loss.item()

            # compute iou and dice
            iou = metric.IoU(mask_hat, mask, selected_classes)
            dice = metric.Dice(mask_hat, mask, selected_classes)
            recal = metric.Sensitivity(mask_hat, mask, selected_classes)
            spec = metric.Specificity(mask_hat, mask, selected_classes)
            acc = metric.Accuracy(mask_hat, mask, selected_classes)
            run_iou += iou
            run_dice += dice
            run_recal += recal
            run_spec += spec
            run_acc += acc
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'iou': f'{iou:.3f}',
                'dice': f'{dice:.3f}',
            })
            
        avg_run_loss = run_loss / num_data_batch
        avg_run_iou = run_iou / num_data_batch
        avg_run_dice = run_dice / num_data_batch
        avg_run_recal = run_recal / num_data_batch
        avg_run_spec = run_spec / num_data_batch
        avg_run_acc = run_acc / num_data_batch
        
    return avg_run_loss, avg_run_iou, avg_run_dice, avg_run_recal, avg_run_spec, avg_run_acc