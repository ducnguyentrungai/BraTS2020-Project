import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd

def compute_tumor_statistics(root_path: str):
    list_out = []

    # Check if the provided path exists and is a directory
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f'Directory not found: {root_path}')

    # Iterate over each patient folder
    for fol in tqdm(sorted(os.listdir(root_path)), desc='Processing'):
        fol_path = os.path.join(root_path, fol)
        if not os.path.isdir(fol_path):
            continue  # Skip files that are not directories

        out = {'Brats20ID': fol}
        brain_volume = None
        seg_volume = None
        seg_ncr = seg_ed = seg_et = 0.0

        # Loop through all files in the patient folder
        for file in os.listdir(fol_path):
            file_path = os.path.join(fol_path, file)

            # Use T1-weighted image (excluding T1CE) to estimate brain volume
            if 't1' in file.lower() and 't1ce' not in file.lower():
                try:
                    image = nib.load(file_path)
                    voxel_volume = np.prod(image.header.get_zooms())  # volume of a single voxel
                    image_data = image.get_fdata()
                    brain_voxel = np.sum(image_data > 0)  # count non-zero voxels (brain area)
                    brain_volume = brain_voxel * voxel_volume
                    out['brain_volume'] = brain_volume
                except Exception as e:
                    print(f"[ERROR] Failed to load T1 image in {file_path}: {e}")

            # Use segmentation image to calculate tumor volumes
            elif 'seg' in file.lower():
                try:
                    seg = nib.load(file_path)
                    voxel_volume = np.prod(seg.header.get_zooms())
                    seg_data = seg.get_fdata()

                    seg_voxel = np.sum(seg_data > 0) * voxel_volume        # Whole tumor volume
                    seg_ncr = np.sum(seg_data == 1) * voxel_volume         # Necrotic tumor core
                    seg_ed = np.sum(seg_data == 2) * voxel_volume          # Edema region
                    seg_et = np.sum(seg_data == 4) * voxel_volume          # Enhancing tumor

                    seg_volume = seg_voxel
                    out['tumor_volume'] = seg_volume
                    out['ncr_net_volume'] = seg_ncr
                    out['ed_volume'] = seg_ed
                    out['et_volume'] = seg_et
                except Exception as e:
                    print(f"[ERROR] Failed to load segmentation file in {file_path}: {e}")

        # Perform percentage calculations only if required volumes are available
        if brain_volume and seg_volume:
            try:
                out['tumor_pct'] = (seg_volume / brain_volume) * 100
                out['ncr_net_pct'] = (seg_ncr / brain_volume) * 100 if brain_volume > 0 else 0
                out['ed_pct'] = (seg_ed / brain_volume) * 100 if brain_volume > 0 else 0
                out['et_pct'] = (seg_et / brain_volume) * 100 if brain_volume > 0 else 0
            except ZeroDivisionError:
                out['tumor_pct'] = out['ncr_net_pct'] = out['ed_pct'] = out['et_pct'] = 0
        else:
            print(f"[WARNING] Missing brain or tumor volume in {fol}. Skipping percentage calculations.")
            out['tumor_pct'] = out['ncr_net_pct'] = out['ed_pct'] = out['et_pct'] = None

        list_out.append(out)

    return pd.DataFrame(list_out).round(4)