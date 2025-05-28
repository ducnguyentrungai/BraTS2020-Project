import nibabel as nib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    image_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr/image_001.nii.gz"
    label_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr/label_001.nii.gz"
    image = nib.load(image_path)
    image_data = image.get_fdata()
    
    label = nib.load(label_path)
    label_data = label.get_fdata()
    
    print(image_data.shape)
    print(label_data.shape)
    
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_data[:, :, 69, -1])
    plt.subplot(1, 2, 2)
    plt.imshow(label_data[:, :, 69])
    # plt.savefig('showtest.png', bbox_inches='tight')