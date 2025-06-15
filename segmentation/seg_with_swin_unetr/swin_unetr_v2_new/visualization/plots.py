import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("/work/cuc.buithi/brats_challenge/code/segmentation/seg_with_swin_unetr/swin_unetr_v2_new/logs/swin_unetr_v2_logs/version_0/metrics.csv")
df_valid = df[df["train_loss_step"].notna()]

# Tính trung bình theo mỗi epoch
df_epoch_mean = df_valid.groupby("epoch").agg({
    "train_loss_step": "mean",
    "train_dice_step": "mean"
}).reset_index()

# Vẽ biểu đồ line của giá trị trung bình loss và dice theo epoch
plt.figure(figsize=(12, 5))

# Biểu đồ loss trung bình
plt.subplot(1, 2, 1)
plt.plot(df_epoch_mean["epoch"], df_epoch_mean["train_loss_step"], label="Mean Train Loss", color="blue", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mean Train Loss per Epoch")
plt.grid(True)
plt.legend()

# Biểu đồ dice trung bình
plt.subplot(1, 2, 2)
plt.plot(df_epoch_mean["epoch"], df_epoch_mean["train_dice_step"], label="Mean Train Dice", color="green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Mean Train Dice per Epoch")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('LOSS-DICE TRAINING')
plt.show()