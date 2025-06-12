import os
import re
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from typing import List, Tuple, Union, Optional, Literal

def normalize_zscore(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img).astype(np.float32)
    mask = img > 0
    if np.any(mask):
        mean = img[mask].mean()
        std = img[mask].std()
        img[mask] = (img[mask] - mean) / (std + 1e-8)
    return img

def normalize_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Scale ảnh về khoảng [0, 1], chỉ tính trên vùng ảnh > 0 (vùng não).
    Nếu không có giá trị dương, trả về ảnh gốc.

    Args:
        img: ảnh 3D (H, W, D)
        eps: để tránh chia cho 0

    Returns:
        Ảnh đã được scale về [0, 1]
    """
    img = np.nan_to_num(img).astype(np.float32)
    mask = img > 0
    if np.any(mask):
        min_val = img[mask].min()
        max_val = img[mask].max()
        img[mask] = (img[mask] - min_val) / (max_val - min_val + eps)
    return img


def smart_crop_brats(image: np.ndarray, label: Union[np.ndarray, None], margin: int = 10, min_size=(128, 128, 128)) -> Tuple[np.ndarray, Union[np.ndarray, None], Tuple[int]]:
    assert image.ndim == 4, "Expected image shape (H, W, D, C)"
    h, w, d, _ = image.shape

    # Thay vì chỉ cần nonzero ở 1 modality → dùng max(image) > 0 để lấy toàn bộ não (vì các voxel não thường có giá trị > 0 ở ít nhất 1 modality)
    image_brain_mask = np.max(image, axis=-1) > 0
    if label is not None:
        tumor_mask = label > 0
        image_brain_mask = image_brain_mask | tumor_mask
    # === END ===

    mask = binary_fill_holes(image_brain_mask)
    nonzero = np.where(mask)
    x_min, x_max = nonzero[0].min(), nonzero[0].max() + 1
    y_min, y_max = nonzero[1].min(), nonzero[1].max() + 1
    z_min, z_max = nonzero[2].min(), nonzero[2].max() + 1

    x_min = max(0, x_min - margin)
    x_max = min(h, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(w, y_max + margin)
    z_min = max(0, z_min - margin)
    z_max = min(d, z_max + margin)

    def ensure_min_size(min_val, max_val, max_dim, min_size):
        if (max_val - min_val) < min_size:
            center = (min_val + max_val) // 2
            min_val = max(0, center - min_size // 2)
            max_val = min(max_dim, min_val + min_size)
        return min_val, max_val

    x_min, x_max = ensure_min_size(x_min, x_max, h, min_size[0])
    y_min, y_max = ensure_min_size(y_min, y_max, w, min_size[1])
    z_min, z_max = ensure_min_size(z_min, z_max, d, min_size[2])

    cropped_img = image[x_min:x_max, y_min:y_max, z_min:z_max, :]
    cropped_label = label[x_min:x_max, y_min:y_max, z_min:z_max] if label is not None else None
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)

    return cropped_img, cropped_label, bbox


def pad_to_shape(img: np.ndarray, target_shape: Tuple[int, int, int], label: np.ndarray = None) -> np.ndarray:
    """
    Pad hoặc crop ảnh về target_shape theo từng chiều.
    Nếu cần crop, sẽ crop xung quanh khối u (dựa trên label).
    Hỗ trợ ảnh 3D (H, W, D) hoặc 4D (H, W, D, C).
    """
    def crop_along_axis(arr, start, end, axis):
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(start, end)
        return arr[tuple(slicer)]

    def pad_along_axis(arr, pad_before, pad_after, axis):
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad_before, pad_after)
        return np.pad(arr, pad_width, mode='constant')

    h, w, d = img.shape[:3]
    th, tw, td = target_shape
    img_out = img.copy()

    # Nếu có label chứa tumor thì crop xung quanh tumor, ngược lại crop ở giữa
    if label is not None and np.any(label > 0):
        tumor_coords = np.array(np.where(label > 0))
        center = tumor_coords.mean(axis=1).astype(int)
    else:
        center = [h // 2, w // 2, d // 2]

    for axis, (size, t_size, c) in enumerate(zip((h, w, d), target_shape, center)):
        if size > t_size:
            # Crop centered
            start = max(0, c - t_size // 2)
            end = start + t_size
            if end > size:
                end = size
                start = end - t_size
            img_out = crop_along_axis(img_out, start, end, axis)
        elif size < t_size:
            # Pad centered
            pad_total = t_size - size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            img_out = pad_along_axis(img_out, pad_before, pad_after, axis)

    return img_out


def get_max_existing_index(images_path: str) -> int:
    max_index = -1
    if os.path.exists(images_path):
        for fname in os.listdir(images_path):
            match = re.search(r'image_(\d+)\.nii\.gz', fname)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1

def prepare_data(
    root_paths: List[str],
    out_path: str,
    images_stack: List[str],
    target_shape: Union[Tuple[int, int, int], None] = None,
    dim: int = -1,
    scale: Optional[Literal['z-score', 'min-max']] = 'min-max',
    margin:int= 32,
    train: bool = True
):
    valid_modalities = {'t1', 't1ce', 't2', 'flair'}
    if not all(mod.lower() in valid_modalities for mod in images_stack):
        raise ValueError(f"Invalid images_stack: expected subset of {valid_modalities}, got {images_stack}")

    mode = 'train' if train else 'test'
    subfolder = f"{mode}_{'_'.join(images_stack)}"
    images_path = os.path.join(out_path, subfolder, 'imageTr')
    labels_path = os.path.join(out_path, subfolder, 'labelTr')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    log_path = os.path.join(out_path, subfolder, 'log.csv')

    all_folders = []
    for root_path in root_paths:
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"Invalid root path: {root_path}")
        all_folders += [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    start_idx = get_max_existing_index(images_path)
    log_entries = []

    for idx, folder_path in enumerate(tqdm(sorted(all_folders), desc=f"Preparing {mode} data")):
        folder_name = os.path.basename(folder_path)
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
                if re.search(fr"_{modality}\.nii(\.gz)?$", file_lower):
                    img = nib.load(file_path)
                    modality_images[modality] = img.get_fdata().astype(np.float32)
                    if affine is None:
                        affine = img.affine

        if len(modality_images) != len(images_stack):
            print(f"[Warning] Missing modalities in {folder_name}: found {list(modality_images.keys())}")
            continue

        stacked = np.stack([modality_images[m] for m in images_stack], axis=dim)
        
        if scale not in ['min-max', 'z-score']:
            raise ValueError("scale must be 'min-max', 'z-score' or None")
        elif scale.lower() == 'z-score':
            for c in range(stacked.shape[-1]):
                stacked[..., c] = normalize_zscore(stacked[..., c])
        elif scale.lower() == 'min-max':
            for c in range(stacked.shape[-1]):
                stacked[..., c] = normalize_minmax(stacked[..., c])
        else: 
            pass

        stacked, label, bbox = smart_crop_brats(stacked, label, margin=margin)
        
        if label is not None:
                label[label == 4] = 3
        
        if target_shape is not None: 
            stacked = pad_to_shape(stacked, target_shape, label)
            if label is not None:
                label = pad_to_shape(label, target_shape, label)
            

        nib.save(nib.Nifti1Image(stacked, affine), os.path.join(images_path, f"image_{patient_id}.nii.gz"))
        if label is not None:
            nib.save(nib.Nifti1Image(label, affine), os.path.join(labels_path, f"label_{patient_id}.nii.gz"))

        log_entries.append([
            patient_id,
            folder_name,
            str(stacked.shape),
            str(label.shape if label is not None else None),
            str(bbox)
        ])

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'folder', 'image_shape', 'label_shape', 'crop_bbox'])
        writer.writerows(log_entries)

if __name__ == "__main__":
    prepare_data(
        root_paths=["/work/cuc.buithi/brats_challenge/BraTS2021"],
        out_path="/work/cuc.buithi/brats_challenge/data/temp",
        images_stack=['t1', 't1ce', 't2', 'flair'],
        target_shape=(128, 128, 128),
        dim=-1,
        train=True,
        scale='min-max',
        margin=5,
    )
