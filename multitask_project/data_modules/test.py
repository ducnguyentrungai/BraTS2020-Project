import os
import glob
import pandas as pd
from typing import List

def create_data_dicts_from_bratsid(root_dir: str,
                                   table_path: str,
                                   modalities: List[str]) -> List[dict]:

    def find_file_by_keyword(folder: str, keyword: str) -> str:
        nii_files = glob.glob(os.path.join(folder, "*.nii*"))  # Match cả .nii và .nii.gz
        for f in nii_files:
            if keyword.lower() in os.path.basename(f).lower():
                return f
        return None

    tab_data = pd.read_csv(table_path)
    tab_data.set_index("Brats20ID", inplace=True)

    data_dicts = []
    for brats_id in tab_data.index:
        matched_dirs = glob.glob(os.path.join(root_dir, f"*{brats_id}*"))
        if not matched_dirs:
            print(f"❌ Không tìm thấy thư mục chứa {brats_id}")
            continue
        case_dir = matched_dirs[0]

        # Tìm ảnh modalities
        images = []
        for mod in modalities:
            img_path = find_file_by_keyword(case_dir, mod)
            if img_path:
                images.append(img_path)
            else:
                print(f"❌ Thiếu ảnh modality '{mod}' trong {brats_id}")
                break
        if len(images) != len(modalities):
            continue

        # Tìm segmentation
        label_path = find_file_by_keyword(case_dir, "seg")
        if not label_path:
            print(f"❌ Không tìm thấy segmentation trong {brats_id}")
            continue

        row = tab_data.loc[brats_id]
        tabular_features = row.drop("Survival_Class").to_dict()
        survival_class = int(row["Survival_Class"])

        data_dicts.append({
            "image": images,
            "label": label_path,
            "tabular": tabular_features,
            "label_class": survival_class
        })

    print(f"✅ Tạo được {len(data_dicts)} sample hợp lệ.")
    return data_dicts

if __name__ == "__main__":
    root = "/work/cuc.buithi/brats_challenge/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    tab_path = "/work/cuc.buithi/brats_challenge/code/multitask_project/data/suvivaldays_info.csv"
    modalities = ["flair", "t1", "t1ce", "t2"]

    data = create_data_dicts_from_bratsid(root_dir=root, table_path=tab_path, modalities=modalities)

    if data:
        from pprint import pprint
        print(data[0])

