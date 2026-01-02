import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2

class spine_dataset(Dataset):
    
    def __init__(self):
        self.samples_img = []
        self.samples_mask = []

        des_size = (256,256)

        path_1_images = f"..{os.sep}data{os.sep}data_1{os.sep}images"
        path_1_masks = f"..{os.sep}data{os.sep}data_1{os.sep}masks"

        for image, mask in zip(sorted(os.listdir(path_1_images)), sorted(os.listdir(path_1_masks))):
            full_image_path = os.path.join(path_1_images, image)
            full_mask_path = os.path.join(path_1_masks, mask)

            try:

                img = sitk.GetArrayFromImage(sitk.ReadImage(full_image_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(full_mask_path))
                
                img = cv2.resize(img, dsize =des_size, interpolation=cv2.INTER_CUBIC )
                mask = cv2.resize(mask, dsize =des_size, interpolation=cv2.INTER_CUBIC )

                z = img.shape[2] // 2

                img = img[:,:,z]
                mask = mask[:,:,z]

                self.samples_img.append(img)
                self.samples_mask.append(mask)
            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path)
            
        path_2 = f"..{os.sep}data{os.sep}data_2"

        for el in sorted(os.listdir(path_2)):
            full_path = os.path.join(path_2, el)

            full_image_path = os.path.join(full_path, f"{el}_sagittal_image.nii.gz")
            full_mask_path = os.path.join(full_path, f"{el}_sagittal_label.nii.gz")

            try:

                img = sitk.GetArrayFromImage(sitk.ReadImage(full_image_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(full_mask_path))
                
                img = np.moveaxis(img, 0, -1)
                mask = np.moveaxis(mask, 0,-1)
                
                img = cv2.resize(img, dsize =des_size, interpolation=cv2.INTER_CUBIC )
                mask = cv2.resize(mask, dsize =des_size, interpolation=cv2.INTER_CUBIC )

                z = img.shape[2] // 2

                img = img[:,:,z]
                mask = mask[:,:,z]

                self.samples_img.append(img)
                self.samples_mask.append(mask)
            
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

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        # maska
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()
