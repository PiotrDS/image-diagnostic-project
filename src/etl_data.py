import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A

def load_img_and_mask(img_path, mask_path, des_size):

    img = sitk.ReadImage(img_path)
    img = sitk.DICOMOrient(img, "RAS")
    img = sitk.GetArrayFromImage(img)

    mask = sitk.ReadImage(mask_path)
    mask = sitk.DICOMOrient(mask, "RAS")
    mask = sitk.GetArrayFromImage(mask)
                
    img = cv2.resize(img, dsize =des_size, interpolation=cv2.INTER_CUBIC )
    mask = cv2.resize(mask, dsize =des_size, interpolation=cv2.INTER_CUBIC )

    z = img.shape[2] // 2

    img = img[:,:,z]
    mask = mask[:,:,z]

    img = standardize(img)
    mask = standardize(mask)

    return img, mask

def standardize(img, eps=1e-8):

    pixels = img[img > 0]

    mean = pixels.mean()
    std = pixels.std()

    img = (img - mean) / (std + eps)
    return img

class spine_dataset(Dataset):
    
    def __init__(self):
        self.samples_img = []
        self.samples_mask = []
        self.augmented = []

        des_size = (256,256)

        path_1_images = f"..{os.sep}data{os.sep}data_1{os.sep}images"
        path_1_masks = f"..{os.sep}data{os.sep}data_1{os.sep}masks"

        transform = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
                        A.GaussNoise(p=0.2),
                        A.ElasticTransform(alpha=1, sigma=50, p=0.3)
        ], additional_targets={'mask': 'mask'})

        for image, mask in zip(sorted(os.listdir(path_1_images)), sorted(os.listdir(path_1_masks))):
            full_image_path = os.path.join(path_1_images, image)
            full_mask_path = os.path.join(path_1_masks, mask)

            try:

                img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size)
                self.samples_img.append(img)
                self.samples_mask.append(mask)
                self.augmented.append(False)

                for _ in range(8):
                    try:
                        augmented_img_mask = transform(image=img, mask=mask)
                    except Exception as e:
                        continue
                        
                    aug_img = augmented_img_mask["image"]
                    aug_mask = augmented_img_mask["mask"]
                        
                    self.samples_img.append(aug_img)
                    self.samples_mask.append(aug_mask)
                    self.augmented.append(True)

            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path)
            
        path_2 = f"..{os.sep}data{os.sep}data_2"

        for el in sorted(os.listdir(path_2)):
            full_path = os.path.join(path_2, el)

            full_image_path = os.path.join(full_path, f"{el}_sagittal_image.nii.gz")
            full_mask_path = os.path.join(full_path, f"{el}_sagittal_label.nii.gz")

            try:

                img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size)

                self.samples_img.append(img)
                self.samples_mask.append(mask)
                self.augmented.append(False)

                for _ in range(5):
                    try:
                        augmented_img_mask = transform(image=img, mask=mask)
                    except Exception as e:
                        continue

                    aug_img = augmented_img_mask["image"]
                    aug_mask = augmented_img_mask["mask"]
                        
                    self.samples_img.append(aug_img)
                    self.samples_mask.append(aug_mask)
                    self.augmented.append(True)

            except Exception as e:
                
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path)           
    
    def __len__(self):
        return len(self.samples_img)

    def __getitem__(self, idx):
        
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        
        # (H, W) -> (1, H, W)
        X = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0)
        
        return X, y

    def visualize(self, idx):
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        ori = self.augmented[idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        fig.suptitle(f"Visualization of idx={idx} original = {not(ori)}")

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        # maska
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()
