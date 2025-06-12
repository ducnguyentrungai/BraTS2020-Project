import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib


if __name__ == "__main__":
    image_path = "/work/cuc.buithi/brats_challenge/data/temp/train_t1_t1ce_t2_flair/imageTr/image_00001.nii.gz"
    image_data = nib.load(image_path).get_fdata()
    label_path = "/work/cuc.buithi/brats_challenge/data/temp/train_t1_t1ce_t2_flair/labelTr/label_00001.nii.gz"
    lable_data = nib.load(label_path).get_fdata()
    image__scale = img = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

    print(lable_data.min())
    print(lable_data.max())
    print(image_data.shape)
    print(image_data.max())
    print(image_data.min())
    plt.subplot(1, 2, 1)
    plt.imshow(image_data[:, :, 69, -1], vmin=image_data.min(),vmax=image_data.max(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(lable_data[:, :, 69], vmin=lable_data.min(), vmax=lable_data.max())
    plt.savefig('img_lab1.png')
    plt.show()
