from models_custom import SwinUNETR
import torch

if __name__ == "__main__":
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_global_attn=True,
        depths=(2, 4, 2, 2),
        use_v2=True
    )

    input = torch.randn(1, 4, 128, 128, 128)
    out = model(input)
    print(input.shape)
    print(out.shape)