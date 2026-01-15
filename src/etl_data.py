import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from torch.utils.data import random_split

def load_test_data():

    images = []

    des_size = (512,512)

    path_images = f"..{os.sep}data{os.sep}data_3"
    
    if os.path.exists(path_images):
        for path1 in sorted(os.listdir(path_images)):
            new_path = path_images + os.sep + path1
            if os.path.isdir(new_path):
                for path2 in sorted(os.listdir(new_path)):
                    next_path = new_path+os.sep+path2
                    if os.path.isdir(next_path):
                        for path3 in sorted(os.listdir(next_path)):
                            if path3.startswith("T1_TSE_SAG"):
                                next_next_path = next_path + os.sep + path3
                                for path4 in sorted(os.listdir(next_next_path)):
                                    if path4.endswith(("002.ima","005.ima","007.ima","010.ima")):
                                        final_path = next_next_path + os.sep + path4

                                        img_itk = sitk.ReadImage(final_path)
                                        img_itk = sitk.DICOMOrient(img_itk, "RAS")
                                        img_3d = sitk.GetArrayFromImage(img_itk)
                                        img = img_3d[:,:,0]

                                        img = np.flipud(img)


                                        img = resize_with_pad(img, des_size, interpolation=cv2.INTER_LINEAR)

                                        img = standardize(img)
                                        
                                        images.append(img)
    return images


def load_data():

    samples_img = []
    samples_mask = []
    types = []

    des_size = (512, 512)

    # load first dataset
    path_1_images = f"..{os.sep}data{os.sep}data_1{os.sep}images"
    path_1_masks = f"..{os.sep}data{os.sep}data_1{os.sep}masks"

    seen = []

    if os.path.exists(path_1_images):
        for image, mask in zip(sorted(os.listdir(path_1_images)), sorted(os.listdir(path_1_masks))):

            image_number = image.split('_')[0]

            
            if image_number in seen:
                continue
                
            seen.append(image_number)

            full_image_path = os.path.join(path_1_images, image)
            full_mask_path = os.path.join(path_1_masks, mask)

            try:
                imgs, masks = load_img_and_mask(full_image_path, full_mask_path, des_size, type=1)
                
                samples_img.append(imgs)
                samples_mask.append(masks)

                types.append([1 for _ in range(len(imgs))])

    
            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path)
            
        # # load second dataset
        
    path_2 = f"..{os.sep}data{os.sep}data_2"
    if os.path.exists(path_2):
        for el in sorted(os.listdir(path_2)):
            full_path = os.path.join(path_2, el)

            full_image_path = os.path.join(full_path, f"{el}_sagittal_image.nii.gz")
            full_mask_path = os.path.join(full_path, f"{el}_sagittal_label.nii.gz")
    
            try:
                imgs, masks = load_img_and_mask(full_image_path, full_mask_path, des_size, type=2)
                samples_img.append(imgs)
                samples_mask.append(masks)               

                types.append([2 for _ in range(len(imgs))])
                    
            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path) 

    return samples_img, samples_mask, types

def split_data(imgs, masks, types ,train_frac=0.7, val_frac=0.2, seed=42):


    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac > 1.0:
        raise ValueError("Sum of train_frac and val_frac must be <= 1.0")

    total_len = len(imgs)
    test_frac = 1.0 - train_frac - val_frac

    torch.manual_seed(seed)
    indices = torch.randperm(total_len)

    train_len = int(train_frac * total_len)
    val_len = int(val_frac * total_len)

    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    train_imgs = [imgs[i] for i in train_idx]
    train_imgs = [el for sub_list in train_imgs for el in sub_list]
    train_masks = [masks[i] for i in train_idx]
    train_masks = [el for sub_list in train_masks for el in sub_list]
    train_types = [types[i] for i in train_idx]
    train_types = [el for sub_list in train_types for el in sub_list]

    val_imgs = [imgs[i] for i in val_idx]
    val_imgs = [el for sub_list in val_imgs for el in sub_list]
    val_masks = [masks[i] for i in val_idx]
    val_masks = [el for sub_list in val_masks for el in sub_list]
    val_types = [types[i] for i in val_idx]
    val_types = [el for sub_list in val_types for el in sub_list]

    test_imgs = [imgs[i] for i in test_idx]
    test_imgs = [el for sub_list in test_imgs for el in sub_list]
    test_masks = [masks[i] for i in test_idx]
    test_masks = [el for sub_list in test_masks for el in sub_list]
    test_types = [types[i] for i in test_idx]
    test_types = [el for sub_list in test_types for el in sub_list]

    return train_imgs, train_masks, train_types, val_imgs, val_masks, val_types, test_imgs, test_masks, test_types

def resize_with_pad(image, target_size, interpolation):

    h, w = image.shape
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    
    return canvas
def map_labels(mask, type):
    """
    Maps raw labels to semantic classes.
    Target mapping:
    Background: stays at 0
    Vertebrae: from labels greater than 200 -> 1
    Discs: from labels 1 to 19 -> 2
    Sacrum: from label 100 -> 0 (to background)
    """

    new_mask = np.zeros_like(mask)
    if type==1:

        # 1. Map: Vertebrae -> Class 1
        new_mask[(mask >= 1) & (mask < 100)] = 1
        
        # 2. Map: Discs -> Class 2
        new_mask[mask >= 200] = 2
        

    elif type==2:

        # 1. Map: Vertebrae -> Class 1
        new_mask[mask == 1] = 1

        # 2. Map: Discs -> Class 2
        new_mask[mask == 3] = 2

    return new_mask

def load_img_and_mask(img_path, mask_path, des_size, type):

    imgs = []
    masks = []

    # Load dicom files
    img_itk = sitk.ReadImage(img_path)
    img_itk = sitk.DICOMOrient(img_itk, "RAS")
    img_3d = sitk.GetArrayFromImage(img_itk)

    mask_itk = sitk.ReadImage(mask_path)
    mask_itk = sitk.DICOMOrient(mask_itk, "RAS")
    mask_3d = sitk.GetArrayFromImage(mask_itk)

    for sagittal_slice_idx in range(img_3d.shape[2]):
        
        img = img_3d[:, :, sagittal_slice_idx]
        mask = mask_3d[:, :, sagittal_slice_idx]
        
        img = np.flipud(img)
        mask = np.flipud(mask)

        img = resize_with_pad(img, des_size, interpolation=cv2.INTER_LINEAR)
        mask = resize_with_pad(mask, des_size, interpolation=cv2.INTER_NEAREST)

        mask = map_labels(mask, type)
        
        img = standardize(img)
        mask = mask.astype(np.int64)

        imgs.append(img)
        masks.append(mask)

    return imgs, masks


def standardize(img, eps=1e-8):
    if img.max() == 0: return img 
    pixels = img[img > 0]
    if pixels.size == 0: return img
    
    mean = pixels.mean()
    std = pixels.std()
    img = (img - mean) / (std + eps)
    return img

class spine_dataset(Dataset):
    
    def __init__(self, imgs, masks, types, train=True):
        self.samples_img = imgs
        self.samples_mask = masks
        self.types = types
        self.train = train
        
    def __len__(self):
        return len(self.samples_img)

    def __getitem__(self, idx):
        
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        
        if self.train:
            transform = A.Compose([
                            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.9),
                            A.GaussNoise(p=0.2),
                            A.ElasticTransform(alpha=1, sigma=50, p=0.3)
            ], additional_targets={'mask': 'mask'})

            try:
                augmented_img_mask = transform(image=img, mask=mask)
                img = augmented_img_mask["image"]
                mask = augmented_img_mask["mask"]
            except Exception as e:
                pass # if augmentation fail just process with original image and mask

        # Image: (H, W) -> (1, H, W), float32
        X = torch.from_numpy(img).unsqueeze(0).float()
        
        # Mask: (H, W), int64 (Long). 
        y = torch.from_numpy(mask).long()
        

        return X, y

    def visualize(self, idx):
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        type = self.types[idx]

        print(f"Number of unique class in mask: {np.unique(mask)}, source type: {type}") 

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()


class spine_test_dataset(Dataset):

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return(len(self.imgs))
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        X = torch.from_numpy(img).unsqueeze(0).float()
        return X