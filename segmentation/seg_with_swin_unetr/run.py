import os
import sys
import torch
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss
from torch.optim import AdamW
from lightning_module import LitSegSwinUNETR
from my_dataset import BratsDataModule
from my_transform import get_transforms
from pynvml import *

# ==== Đảm bảo thư viện phụ thuộc nằm đúng trong sys.path ====
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
    data_dir = '/work/cuc.buithi/brats_challenge/BraTS2021'
    batch_size = 4
    spatial_size = (128, 128, 128)
    num_classes = 4
    in_channels = 4
    root_dir = "swin_unetr_v2_new_batch4"
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    log_dir = os.path.join(root_dir, "logs")

    # ==== Auto GPU selection ====
    selected_gpus = auto_select_gpus(n=2, threshold_mem_mib=5000, threshold_util=55)
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        accelerator = "gpu"
        devices = len(selected_gpus)
    else:
        accelerator = "cpu"
        devices = 1

    strategy = DDPStrategy(find_unused_parameters=False) if accelerator == "gpu" and devices > 1 else "auto"

    # ==== DataModule ====
    datamodule = BratsDataModule(
        data_dir=data_dir,
        spatial_size=spatial_size,
        batch_size=batch_size,
        num_workers=4,
        train_percent=0.9,
        transform_fn=lambda is_train: get_transforms(spatial_size=spatial_size, is_train=is_train)
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

    # ==== Loss & Lightning Module ====
    loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_focal=1.0)

    lightning_model = LitSegSwinUNETR(
        model=model,
        loss_fn=loss_fn,
        optim_class=AdamW,
        lr=2e-4,
        num_classes=num_classes,
        include_background=False,
        out_path=log_dir
    )

    # ==== Callbacks ====
    checkpoint_cb = ModelCheckpoint(
        monitor='val_dice',
        dirpath=ckpt_dir,
        filename='best_model-{epoch:02d}-{val_dice:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    csv_logger = CSVLogger(save_dir=log_dir, name="swin_unetr_v2_logs")

    # ==== Trainer ====
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",  # Mixed precision
        accumulate_grad_batches=4,
        callbacks=[checkpoint_cb],
        logger=csv_logger,
        log_every_n_steps=10,
        default_root_dir=root_dir
    )

    # ==== Load weights từ checkpoint như pretrain ====
    checkpoint_path = "/work/cuc.buithi/brats_challenge/code/segmentation/seg_with_swin_unetr/swin_unetr_v2_new/checkpoints/best_model-epoch=116-val_dice=0.8749.ckpt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    missing, unexpected = lightning_model.load_state_dict(state_dict, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)

    # ==== Training ====
    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    train()
