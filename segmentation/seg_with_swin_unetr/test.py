import os
import glob
from typing import List

import os
import glob
import json
from typing import List

def create_data_dicts(data_dirs: List[str], modalities: List[str], error_json_path: str = "missing_cases.json") -> List[dict]:
    """
    Tạo danh sách dict chuẩn MONAI: {'image': [...], 'label': ...}
    Ghi log lỗi (các case thiếu file) ra file JSON.
    """
    data_dicts = []
    error_cases = []

    for case_dir in data_dirs:
        base = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{base}_{mod}.nii.gz") for mod in modalities]
        label = os.path.join(case_dir, f"{base}_seg.nii.gz")

        missing_files = [p for p in images if not os.path.exists(p)]
        if not os.path.exists(label):
            missing_files.append(label)

        if missing_files:
            print(f"❌ Case {base} bị thiếu {len(missing_files)} file:")
            for p in missing_files:
                print(f"   - {p}")
            error_cases.append({
                "case_id": base,
                "missing_files": missing_files
            })
        else:
            data_dicts.append({
                "image": images,
                "label": label,
                "case_id": base
            })

    # Ghi file lỗi
    if error_cases:
        with open(error_json_path, "w") as f:
            json.dump(error_cases, f, indent=2)
        print(f"⚠️ Đã ghi {len(error_cases)} case lỗi vào '{error_json_path}'")
    else:
        print("✅ Không có case nào lỗi.")

    print(f"✅ Tổng số case hợp lệ: {len(data_dicts)} / {len(data_dirs)}")
    return data_dicts



if __name__ == "__main__":
    path = '/work/cuc.buithi/brats_challenge/BraTS2021'
    all_cases = sorted(glob.glob(os.path.join(path, "BraTS2021_*")))
    data = create_data_dicts(all_cases, modalities=['t1', 't1ce', 't2', 'flair'], error_json_path="brats_missing_cases.json")
    from pprint import pprint
    pprint(data[100])
