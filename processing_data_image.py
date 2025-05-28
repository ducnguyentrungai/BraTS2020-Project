import os
import nibabel as nib
import numpy as np
from  tqdm import tqdm
import pandas as pd

# Normalize image to range [0, 255]
def normalize_to_255(image: np.ndarray, mask: bool = False) -> np.ndarray:
    image = image.astype(np.float64)
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    if mask:
        return (norm_image * 255).astype(np.float64)
    return (norm_image * 255).astype(np.uint8)

# Prepare data function
def prepare_data(root_path, out_path, images_stack: list, dim: int, scales: bool = True,
                 hw_crop: list = None, d_crop: list = None, train: bool = True):

    valid_modalities = ['t1', 't1ce', 't2', 'flair']

    if not os.path.isdir(root_path):
        raise FileNotFoundError(f'Directory not found: {root_path} !')

    if len(images_stack) == 0 or any(mod.lower() not in valid_modalities for mod in images_stack):
        raise ValueError(f"Invalid images_stack: must contain only values from {valid_modalities}")

    # Output paths
    names = '_' + '_'.join(images_stack)
    data_path = os.path.join(out_path, ('train' if train else 'test') + names)
    images_path = os.path.join(data_path, 'imageTr')
    labels_path = os.path.join(data_path, 'labelTr')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Iterate over patient folders
    for folder in tqdm(sorted(os.listdir(root_path)), desc="Processing data"):
        folder_path = os.path.join(root_path, folder)
        if not os.path.isdir(folder_path):
            continue

        patient_id = folder.split('_')[-1]
        modality_images = {}
        label = None
        flair_affine = None

        # Load images
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if 'seg' in file.lower():
                label = nib.load(file_path).get_fdata().astype(np.uint8)

            for modality in images_stack:
                if modality.lower() in file.lower():
                    img = nib.load(file_path)
                    modality_images[modality.lower()] = img.get_fdata().astype(np.float32)
                    if flair_affine is None:
                        flair_affine = img.affine

        # Ensure all required modalities are present
        if len(modality_images) != len(images_stack):
            raise ValueError(f"Missing modalities in {folder}. Found: {list(modality_images.keys())}, expected: {images_stack}")

        # Stack images in user-specified order
        image_stack = np.stack([modality_images[m.lower()] for m in images_stack], axis=dim)

        # Crop image if specified
        if hw_crop and len(hw_crop) == 2:
            image_stack = image_stack[
                hw_crop[0]: hw_crop[1],
                hw_crop[0]: hw_crop[1],
                d_crop[0]: d_crop[1],
                ...
            ]

        # Normalize if required
        if scales:
            image_stack = normalize_to_255(image_stack, mask=False)

        # Save stacked image
        nib.save(nib.Nifti1Image(image_stack, affine=flair_affine),
                 os.path.join(images_path, f"image_{patient_id}.nii.gz"))

        # Optionally crop and save label
        if label is not None:
            if hw_crop and len(hw_crop) == 2:
                label = label[
                    hw_crop[0]:hw_crop[1],
                    hw_crop[0]:hw_crop[1],
                    d_crop[0]:d_crop[1]
                ]
                label[label == 4] = 3
            nib.save(nib.Nifti1Image(label, affine=flair_affine),
                     os.path.join(labels_path, f"label_{patient_id}.nii.gz"))
                
if __name__ == "__main__":
    
    out_path = "/work/cuc.buithi/brats_challenge/data"
    root_path = "/work/cuc.buithi/brats_challenge/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"\
    
    
    # [t1ce]
    # [t2, flair]
    # [t1ce, t1]
    # [t1, t1ce, t2, flair]
    
    prepare_data(root_path=root_path, 
                 out_path=out_path, 
                 images_stack=['t1', 't1ce', 't2', 'flair'], 
                 scales=True, 
                 hw_crop=[24, 216], 
                 d_crop=[13, 141] ,
                 dim=-1, train=True)
    
    # img = nib.load('/work/cuc.buithi/brats_challenge/data/train_t1_t2/imageTr/image_001.nii.gz').get_fdata()
    # print(img.shape)
    
    
    