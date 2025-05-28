import torch
import shutil
import nibabel  as nib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    label = nib.load("/work/cuc.buithi/brats_challenge/subdata/Training/labelsTr/image_001.nii.gz").get_fdata()
    # slice_data = label[:, :, 69]
    # plt.imshow(slice_data)
    # plt.axis('off')
    # plt.savefig("slice_69.png", bbox_inches='tight')
    print(label.shape)
    
    