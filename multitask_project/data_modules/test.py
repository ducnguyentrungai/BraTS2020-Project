
import torch
from monai.networks.nets import SwinUNETR

a = torch.randn(4, 1, 160, 160, 160)  # Batch size = 4, 1 channel, 160³ volume

model = SwinUNETR(
    img_size=(160, 160, 160),
    in_channels=1,
    out_channels=2
)

b = model(a)
print(b.shape)
