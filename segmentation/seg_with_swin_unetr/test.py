import os
import glob
from typing import List

def create_data_dicts(data_dirs: List[str], modalities: List[str]) -> List[dict]:
    """
    Tạo danh sách dict chuẩn MONAI: {'image': [...], 'label': ...}
    """
    data_dicts = []
    for case_dir in data_dirs:
        base = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
        label = os.path.join(case_dir, f"{base}_seg.nii.gz")
        if all(os.path.exists(p) for p in images) and os.path.exists(label):
            data_dicts.append({"image": images, "label": label})
            
    return data_dicts


if __name__ == "__main__":
    path = '/work/cuc.buithi/brats_challenge/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    all_cases = sorted(glob.glob(os.path.join(path, "BraTS2020_Training_*")))
    print(all_cases)