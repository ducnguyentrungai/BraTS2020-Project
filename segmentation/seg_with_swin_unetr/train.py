from lightning_module import SwinUNETRModule
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, CacheDataset
from dataset import get_dataloader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

def auto_select_gpus(n=3, threshold_mem_mb=800, threshold_util=10):
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_info = nvmlDeviceGetUtilizationRates(handle)

            mem_used_mb = int(mem_info.used // 1024**2)
            gpu_util = util_info.gpu

            if mem_used_mb < threshold_mem_mb and gpu_util < threshold_util:
                gpus.append((i, mem_used_mb, gpu_util))

        if not gpus:
            return None

        gpus.sort(key=lambda x: (x[1], x[2]))
        selected_ids = [g[0] for g in gpus[:n]]
        return selected_ids

    except Exception as e:
        print(f"GPU auto-selection failed: {e}")
        return None
    finally:
        try:
            nvmlShutdown()
        except:
            pass


if __name__ == "__main__":
    train_loader = get_dataloader("data/train_flair_t1_t1ce_t2", is_train=True, batch_size=2)
    val_loader = get_dataloader("data/train_flair_t1_t1ce_t2", is_train=False, batch_size=1)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_dice:.4f}"
    )

    model = SwinUNETRModule()

    trainer = Trainer(
        max_epochs=200,
        accelerator="gpu", devices=1,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
