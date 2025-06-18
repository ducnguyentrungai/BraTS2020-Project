import os
import glob
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from monai.transforms import MapTransform
import torch

# Custom transform to convert tabular dict → vector tensor
def compute_minmax_stats(csv_path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "Age", "tumor_volume", "ncr_net_volume", "ed_volume", "et_volume",
        "tumor_pct", "ncr_net_pct", "ed_pct", "et_pct"
    ]
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max()
        }
    return stats


def minmax_scale_tabular(tab: dict, stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
    def scale(val, vmin, vmax):
        val = float(val)
        x = (val - vmin) / (vmax - vmin + 1e-8)  # scale về [0, 1]
        return x * 10.0 - 5.0  # scale về [-5, 5]

    resection_str = tab.get("Extent_of_Resection", "NA")
    resection_code = (
        5.0 if resection_str == "GTR" else
        -5.0 if resection_str == "STR" else
        0.0
    )

    features = [
        scale(tab["Age"], *stats["Age"].values()),
        resection_code,
        scale(tab["tumor_volume"], *stats["tumor_volume"].values()),
        scale(tab["ncr_net_volume"], *stats["ncr_net_volume"].values()),
        scale(tab["ed_volume"], *stats["ed_volume"].values()),
        scale(tab["et_volume"], *stats["et_volume"].values()),
        scale(tab["tumor_pct"], *stats["tumor_pct"].values()),
        scale(tab["ncr_net_pct"], *stats["ncr_net_pct"].values()),
        scale(tab["ed_pct"], *stats["ed_pct"].values()),
        scale(tab["et_pct"], *stats["et_pct"].values()),
    ]

    return torch.tensor(features, dtype=torch.float)

class TabularToTensor(MapTransform):
    def __init__(self, keys, stats: Dict[str, Dict[str, float]]):
        super().__init__(keys)
        self.stats = stats

    def __call__(self, data):
        d = dict(data)
        d["tabular"] = minmax_scale_tabular(d["tabular"], self.stats)
        d["label_class"] = torch.tensor(d["label_class"], dtype=torch.long)
        return d

def create_data_dicts_from_bratsid(root_dir: str,
                                   table_path: str,
                                   modalities: List[str],
                                   train_percent: float = 0.8, 
                                   shuffle: bool = True
                                   ) -> Tuple[List[dict], List[dict]]:
    tab_data = pd.read_csv(table_path)
    tab_data.set_index("Brats20ID", inplace=True)

    data_dicts = []

    for brats_id in tab_data.index:
        matched_dirs = glob.glob(os.path.join(root_dir, f"*{brats_id}*"))
        if not matched_dirs:
            print(f"❌ Không tìm thấy thư mục cho {brats_id}")
            continue
        case_dir = matched_dirs[0]

        all_files = glob.glob(os.path.join(case_dir, "*.nii"))

        image_paths = []
        for mod in modalities:
            mod_matches = [
                f for f in all_files
                if os.path.basename(f).endswith(f"_{mod}.nii")
            ]
            if len(mod_matches) != 1:
                print(f"❌ Không tìm thấy đúng 1 ảnh modality '{mod}' cho {brats_id}: {mod_matches}")
                break
            image_paths.append(mod_matches[0])

        if len(image_paths) != len(modalities):
            continue

        seg_matches = [
            f for f in all_files
            if "seg" in os.path.basename(f).lower()
        ]
        if len(seg_matches) != 1:
            print(f"❌ Không tìm thấy đúng 1 file segmentation cho {brats_id}")
            continue
        label_path = seg_matches[0]

        row = tab_data.loc[brats_id]
        tabular_features = row.drop("Survival_Class").to_dict()
        survival_class = int(row["Survival_Class"])

        data_dicts.append({
            "image": image_paths,
            "label": label_path,
            "tabular": tabular_features,
            "label_class": survival_class
        })

    print(f"✅ Tạo được {len(data_dicts)} sample hợp lệ.")

    # Split train/test theo tỉ lệ
    train_data, test_data = train_test_split(data_dicts, train_size=train_percent, random_state=42, shuffle=shuffle)
    print(f"📊 Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data
