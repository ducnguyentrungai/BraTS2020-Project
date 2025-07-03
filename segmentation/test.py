import torch
from monai.networks.nets import SwinUNETR

if __name__ == "__main__":
    model = SwinUNETR(
        img_size=(128,128,128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
    )

    x = torch.randn(1,4,128,128,128)

    outputs = model.swinViT(x)

    print("outputs type:", type(outputs))
    print("outputs len:", len(outputs))
    for i, o in enumerate(outputs):
        if isinstance(o, torch.Tensor):
            print(f"outputs[{i}] shape: {o.shape}")
        else:
            print(f"outputs[{i}] type: {type(o)}")

