from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss, DiceFocalLoss
from torch.optim import Adam, AdamW
from lightning_module import LitSegSwinUNETR
from dataset import *
from pynvml import *
import os
import torch.distributed as dist

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)

def auto_select_gpus(n=3, threshold_mem_mib=10000, threshold_util=55):
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_info = nvmlDeviceGetUtilizationRates(handle)
            mem_used_mib = mem_info.used // (1024 ** 2) 
            gpu_util = int(util_info.gpu)

            rank_zero_print(f"GPU {i}: Used {mem_used_mib} MiB, Util {gpu_util}%")

            if mem_used_mib < threshold_mem_mib and gpu_util < threshold_util:
                gpus.append((i, mem_used_mib, gpu_util))

        if not gpus:
            rank_zero_print("⚠️ No suitable GPU found.")
            return None

        gpus.sort(key=lambda x: (x[1], x[2]))
        selected = [g[0] for g in gpus[:n]]

        if len(selected) < n:
            rank_zero_print(f"⚠️ Only found {len(selected)} / {n} suitable GPUs.")

        rank_zero_print(f"✅ Auto-selected GPUs: {selected}")
        return selected

    except Exception as e:
        rank_zero_print(f"⚠️ GPU auto-selection failed: {e}")
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
    root_dir = "swin_unetr_v2"

    ckpt_dir = os.path.join(root_dir, "checkpoints")
    log_dir = os.path.join(root_dir, "logs")

    # ==== Auto GPU selection ====
    selected_gpus = auto_select_gpus(n=4, threshold_mem_mib=1000, threshold_util=20)
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        accelerator = "gpu"
        devices = len(selected_gpus)
        strategy = "ddp" if devices > 1 else "auto"
        rank_zero_print(f"\u2705 Using GPUs: {selected_gpus}")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        rank_zero_print("\u26A0\uFE0F Fallback to CPU")

    # ==== DataModule ====
    datamodule = BratsDataModule(
        data_dir=data_dir,
        spatial_size=spatial_size,
        batch_size=batch_size,
        num_workers=2,
        cache_num=282,
        cache_num_val_all=True,
        train_percent=0.9
    )

    # ==== Model ====
    model = SwinUNETR(
        img_size=spatial_size,
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=48,
        norm_name='batch',
        use_checkpoint=True,
        use_v2=True
    )

    # ==== Loss and Optimizer ====
    loss_fn = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_focal=1.0
    )
    optimizer = AdamW

    lightning_model = LitSegSwinUNETR(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        lr=2e-5,
        num_classes=num_classes,
        include_background=False
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

    # ==== CSV Logger ====
    csv_logger = CSVLogger(save_dir=log_dir, name="swin_unetr_logs")

    # ==== Trainer ====
    trainer = Trainer(
        max_epochs=500,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=16,
        callbacks=[checkpoint_cb],
        logger=csv_logger,
        log_every_n_steps=10,
        default_root_dir=root_dir
    )

    # ==== Train ====
    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    train()