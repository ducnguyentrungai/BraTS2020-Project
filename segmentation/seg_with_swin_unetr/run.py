import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from torch.optim import Adam
from lightning_module import LitSegSwinUNETR
from metrics import Metric
from dataset import *
from pynvml import *
import os

def auto_select_gpus(n=3, threshold_mem_mb=800, threshold_util=10):
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_info = nvmlDeviceGetUtilizationRates(handle)

            mem_used_mb = mem_info.used // 1024**2
            gpu_util = util_info.gpu

            print(f"GPU {i}: Used {mem_used_mb}MB, Util {gpu_util}%")

            if mem_used_mb < threshold_mem_mb and gpu_util < threshold_util:
                gpus.append((i, mem_used_mb, gpu_util))

        if not gpus:
            print("⚠️ No suitable GPU found. Using default CUDA_VISIBLE_DEVICES.")
            return None

        # Sắp xếp theo bộ tiêu chí (RAM dùng, %util thấp)
        gpus.sort(key=lambda x: (x[1], x[2]))
        selected_ids = [g[0] for g in gpus[:n]]
        print(f"✅ Auto-selected GPUs: {selected_ids}")
        return selected_ids

    except Exception as e:
        print(f"⚠️ GPU auto-selection failed: {e}")
        return None
    finally:
        try:
            nvmlShutdown()
        except:
            pass

def train():
    # ==== Config ====
    data_dir = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2'
    batch_size = 2
    spatial_size = (128, 128, 128)
    num_classes = 4 
    in_channels = 4 
    log_dir = "logs"
    ckpt_dir = "checkpoints"

    # ==== Data ====
    train_dir, val_dir = extract_data_dicts(data_dir, train_percent=0.9)
    train_loader = get_dataloader(train_dir, batch_size=batch_size, is_train=True, spatial_size=spatial_size, num_workers=2)
    val_loader   = get_dataloader(val_dir, batch_size=batch_size, is_train=False, spatial_size=spatial_size, num_workers=2)

    # ==== Model ====
    model = SwinUNETR(
        img_size=spatial_size,
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=48,
        norm_name = 'batch',
        use_checkpoint=True
    )

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)

    optimizer = Adam

    lightning_model = LitSegSwinUNETR(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        lr=2e-4,
        num_classes=num_classes,
        include_background=False,
        log_dir=log_dir
    )

    # ==== Checkpoint Callback ====
    checkpoint_cb = ModelCheckpoint(
        monitor='val_dice',
        dirpath=ckpt_dir,
        filename='best_model-{epoch:02d}-{val_dice:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    # ==== Auto GPU selection ====
    selected_gpus = auto_select_gpus(n=4)
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        accelerator = "gpu"
        devices = len(selected_gpus)
        strategy = "ddp" if devices > 1 else "auto"
        print(f"Using GPUs: {selected_gpus}")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        print("Fallback to CPU")

    # ==== Trainer ====
    trainer = Trainer(
        max_epochs=500,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=16,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
        default_root_dir=log_dir
    )

    # ==== Training ====
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    train()
