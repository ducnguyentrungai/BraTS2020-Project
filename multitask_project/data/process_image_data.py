import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import re
import csv

def normalize_to_255(img, mask=True):
    img = np.nan_to_num(img)
    if mask:
        mask = img > 0
        img[mask] = (img[mask] - np.min(img[mask])) / (np.max(img[mask]) - np.min(img[mask]) + 1e-8) * 255
        return img.astype(np.uint8)
    else:
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255
        return img.astype(np.float32)

def get_max_existing_index(images_path):
    max_index = -1
    if os.path.exists(images_path):
        for fname in os.listdir(images_path):
            match = re.search(r'image_(\d+)\.nii\.gz', fname)
            if match:
                idx = int(match.group(1))
                if idx > max_index:
                    max_index = idx
    return max_index + 1


def crop_to_brain_region(image: np.ndarray, axis:int):
    """
    Crop ảnh theo vùng non-zero (não bộ).
    Áp dụng cho ảnh 3D hoặc ảnh 4D với các kênh ở dim=-1.
    """
    if image.ndim == 4:
        # Duyệt qua các kênh để lấy vùng chứa thông tin
        brain_mask = np.any(image > 0, axis=axis)
    else:
        brain_mask = image > 0

    non_zero = np.where(brain_mask)
    if len(non_zero[0]) == 0:
        raise ValueError("No brain region found (all zeros).")

    x_min, x_max = non_zero[0].min(), non_zero[0].max()
    y_min, y_max = non_zero[1].min(), non_zero[1].max()
    z_min, z_max = non_zero[2].min(), non_zero[2].max()

    return (x_min, x_max + 1, y_min, y_max + 1, z_min, z_max + 1)

def prepare_data(root_paths, out_path, images_stack, dim=3, scales=True,
                 hw_crop=None, d_crop=None, train=True):

    valid_modalities = ['t1', 't1ce', 't2', 'flair']
    if len(images_stack) == 0 or any(mod.lower() not in valid_modalities for mod in images_stack):
        raise ValueError(f"Invalid images_stack: must contain only {valid_modalities}")

    # Output folders
    names = '_' + '_'.join(images_stack)
    data_path = os.path.join(out_path, ('train' if train else 'test') + names)
    images_path = os.path.join(data_path, 'imageTr')
    labels_path = os.path.join(data_path, 'labelTr')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # CSV log path
    log_path = os.path.join(data_path, 'log.csv')
    log_entries = []

    # Gather all folders
    all_folders = []
    for root_path in root_paths:
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"Folder not found: {root_path}")
        all_folders += [os.path.join(root_path, f) for f in os.listdir(root_path)
                        if os.path.isdir(os.path.join(root_path, f))]

    start_idx = get_max_existing_index(images_path)

    for idx, folder_path in enumerate(tqdm(sorted(all_folders), desc="Processing")):
        folder_name = os.path.basename(folder_path)
        rel_path = os.path.relpath(folder_path, start=os.path.commonpath(root_paths))  # e.g., BraTS2021/Patient_001
        patient_id = f"{start_idx + idx:05d}"

        modality_images = {}
        label = None
        affine = None

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            file_lower = file.lower()
            if 'seg' in file_lower:
                label = nib.load(file_path).get_fdata().astype(np.uint8)

            for modality in images_stack:
                if any(file_lower.endswith(f"_{modality}{ext}") for ext in [".nii.gz", ".nii"]):
                    img = nib.load(file_path)
                    modality_images[modality] = img.get_fdata().astype(np.float32)
                    if affine is None:
                        affine = img.affine

        if len(modality_images) != len(images_stack):
            print(f"⚠️ Missing modalities in {folder_name}. Found {list(modality_images.keys())}. Skipped.")
            continue

        stacked = np.stack([modality_images[m] for m in images_stack], axis=dim)

        if hw_crop and d_crop:
            stacked = stacked[
                hw_crop[0]:hw_crop[1],
                hw_crop[0]:hw_crop[1],
                d_crop[0]:d_crop[1],
                ...
            ]
            if label is not None:
                label = label[
                    hw_crop[0]:hw_crop[1],
                    hw_crop[0]:hw_crop[1],
                    d_crop[0]:d_crop[1]
                ]

        if scales:
            stacked = normalize_to_255(stacked, mask=False)

        nib.save(nib.Nifti1Image(stacked, affine),
                 os.path.join(images_path, f"image_{patient_id}.nii.gz"))

        if label is not None:
            label[label == 4] = 3
            nib.save(nib.Nifti1Image(label, affine),
                     os.path.join(labels_path, f"label_{patient_id}.nii.gz"))

        # Save log entry
        log_entries.append([patient_id, rel_path])

    # Write CSV
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'source_folder'])
        writer.writerows(log_entries)



if __name__ == "__main__":
    prepare_data(
    root_paths=[
        "/work/cuc.buithi/brats_challenge/BraTS2021"
    ],
    out_path="/work/cuc.buithi/brats_challenge/data",
    images_stack=["flair", "t1", "t1ce", "t2"],
    dim=-1,
    scales=True,
    hw_crop=[25, 215],
    d_crop=[13, 141],
    train=True
    )
