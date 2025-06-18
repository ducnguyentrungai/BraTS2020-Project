from data_modules.brats_datamodule import BratsDataModule
from models.multitask_model import MultiModalMultiTaskModel
from utils.losses import MultiTaskLoss
from lightning.lit_multitask_module import LitMultiTaskModule 

import os
import sys
import torch
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pynvml import *
from transforms.transform_data import get_multitask_transforms, compute_minmax_stats

sys.path.append(os.path.dirname(__file__))

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)

def auto_select_gpus(n=2, threshold_mem_mib=5000, threshold_util=55):
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

            if mem_used_mib < threshold_mem_mib and gpu_util < threshold_util:
                gpus.append((i, mem_used_mib, gpu_util))

        if not gpus:
            return None

        gpus.sort(key=lambda x: (x[1], x[2]))
        return [g[0] for g in gpus[:n]]

    except:
        return None
    finally:
        try:
            nvmlShutdown()
        except:
            pass

def train():
    image_dir = '/work/cuc.buithi/brats_challenge/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    table_path = "/work/cuc.buithi/brats_challenge/code/multitask_project/data/suvivaldays_info.csv"

    batch_size = 2
    spatial_size = (128, 128, 128)
    num_seg_classes = 4
    num_cls_classes = 3
    in_channels = 4
    root_dir = 'multitask_logs'
    ckpt_dir = os.path.join(root_dir, 'checkpoints')
    logs_dir = os.path.join(root_dir, 'logs')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    selected_gpus = auto_select_gpus(n=2)
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        accelerator = "gpu"
        devices = len(selected_gpus)
    else:
        accelerator = "cpu"
        devices = 1

    strategy = DDPStrategy(find_unused_parameters=False) if accelerator == "gpu" and devices > 1 else "auto"

    # === DataModule ===
    
    # Tính thống kê min-max cho tabular features
    tabular_stats = compute_minmax_stats(table_path)

    # Truyền transform đúng chuẩn multitask
    transform_fn = lambda is_train: get_multitask_transforms(
        spatial_size=(128, 128, 128),
        is_train=is_train,
        tabular_stats=tabular_stats
    )

    # Khởi tạo DataModule với transform đúng
    data_module = BratsDataModule(
        data_dir=image_dir,
        table_path=table_path,
        batch_size=batch_size,
        num_workers=2,
        transform_fn=transform_fn,
    )

    # === Model ===
    model = MultiModalMultiTaskModel(
        img_size=spatial_size,
        in_channels=in_channels,
        seg_classes=num_seg_classes,
        cls_classes=num_cls_classes,
        tabular_dim=10,
        feature_size=48,
        hidden_dim=128,
        use_v2=True,
        norm_name='instance'
    )
    
    lit_model = LitMultiTaskModule(
        model=model,
        loss_fn=MultiTaskLoss(loss_weight=1.0),
        lr=1e-4,
        num_seg_classes=num_seg_classes,
        include_background=False              
    )

    # === Logger + Checkpoint ===
    logger = CSVLogger(save_dir=logs_dir, name="multitask")
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch}-valdice={val/dice:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # === Trainer ===
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=500,
        logger=logger,
        callbacks=[checkpoint, lr_monitor],
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        deterministic=True,
    )

    trainer.fit(lit_model, datamodule=data_module)

if __name__ == "__main__":
    train()
