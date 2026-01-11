import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from torch.utils.data import random_split

def load_data():

    samples_img = []
    samples_mask = []
    type = []

    des_size = (512, 512)

    # --- Wczytywanie DATA 1 ---
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
                img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size, type=1)
                samples_img.append(img)
                samples_mask.append(mask)
                type.append(1)

    
            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path)
            
        # --- Wczytywanie DATA 2 ---
        
    path_2 = f"..{os.sep}data{os.sep}data_2"
    if os.path.exists(path_2):
        for el in sorted(os.listdir(path_2)):
            full_path = os.path.join(path_2, el)

            full_image_path = os.path.join(full_path, f"{el}_sagittal_image.nii.gz")
            full_mask_path = os.path.join(full_path, f"{el}_sagittal_label.nii.gz")
            type.append(2)

            try:
                img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size, type=2)

                samples_img.append(img)
                samples_mask.append(mask)
                    
            except Exception as e:
                print(e)
                print("couldn't load ", full_image_path, " or ", full_mask_path) 

    return samples_img, samples_mask, type

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
    train_masks = [masks[i] for i in train_idx]
    train_types = [types[i] for i in train_idx]

    val_imgs = [imgs[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]
    val_types = [types[i] for i in val_idx]

    test_imgs = [imgs[i] for i in test_idx]
    test_masks = [masks[i] for i in test_idx]
    test_types = [types[i] for i in test_idx]

    return train_imgs, train_masks, train_types, val_imgs, val_masks, val_types, test_imgs, test_masks, test_types

def resize_with_pad(image, target_size, interpolation):
    """
    Skaluje obraz zachowując proporcje i dodaje padding (czarne pasy),
    aby wynikowy obraz miał wymiar target_size (H, W).
    """
    h, w = image.shape
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Tworzenie nowego płótna
    canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    
    # Wyliczanie pozycji wklejenia (centrowanie)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    
    return canvas
def map_labels(mask):
    """
    Mapuje surowe etykiety (instance labels) na klasy semantyczne zgodne z artykułem.
    Docelowo:
    0 - Tło
    1 - Kręgi (Vertebrae) -> z numerów > 200
    2 - Dyski (Discs) -> z numerów 1-19
    3 - Kość krzyżowa (Sacrum) -> z numeru 100 (zgaduję, że to 100, bo odstaje)
    """
    new_mask = np.zeros_like(mask)
    
    # 1. Mapowanie Kręgów (Vertebrae) -> Klasa 1
    # Zakładam, że kręgi to wartości powyżej 200 (widzę 201-208 w Twoich logach)
    new_mask[mask >= 200] = 3#1
    
    # 2. Mapowanie Dysków (Discs) -> Klasa 2
    # Zakładam, że dyski to małe numery (widzę 1-8 w Twoich logach)
    # Zazwyczaj dyski są numerowane sekwencyjnie.
    new_mask[(mask >= 1) & (mask < 100)] = 1#2
    
    # 3. Mapowanie Kości Krzyżowej (Sacrum) -> Klasa 3
    # W logach widzę "100". Często okrągłe liczby oznaczają inne struktury.
    # Jeśli Sacrum to nie 100, to prawdopodobnie jeden z tych > 200.
    # Ale spróbujmy tak:
    new_mask[mask == 100] = 2#3
    
    return new_mask

def load_img_and_mask(img_path, mask_path, des_size, type):
    # Wczytanie
    img_itk = sitk.ReadImage(img_path)
    img_itk = sitk.DICOMOrient(img_itk, "RAS")
    img = sitk.GetArrayFromImage(img_itk)

    mask_itk = sitk.ReadImage(mask_path)
    mask_itk = sitk.DICOMOrient(mask_itk, "RAS")
    mask = sitk.GetArrayFromImage(mask_itk)

    # Wybór płaszczyzny Sagittal
    if img.ndim == 3:
        # Zakładam, że oś X (indeks 2) to płaszczyzna strzałkowa w RAS
        sagittal_slice_idx = img.shape[2] // 2
        img = img[:, :, sagittal_slice_idx]
        mask = mask[:, :, sagittal_slice_idx]
        
        # Obrót, żeby głowa była u góry
        img = np.flipud(img)
        mask = np.flipud(mask)

    # Resize z paddingiem (zachowanie proporcji)
    img = resize_with_pad(img, des_size, interpolation=cv2.INTER_LINEAR)
    mask = resize_with_pad(mask, des_size, interpolation=cv2.INTER_NEAREST)

    # --- NOWOŚĆ: MAPOWANIE KLAS ---
    # Musimy to zrobić przed konwersją na typy danych
    if type == 1:
        mask = map_labels(mask)

    # Preprocessing
    img = standardize(img)
    mask = mask.astype(np.int64)

    return img, mask

# Twoja funkcja standardize (bez zmian, jest OK)
def standardize(img, eps=1e-8):
    if img.max() == 0: return img # Zabezpieczenie przed pustym obrazem
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
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
                            A.GaussNoise(p=0.2),
                            A.ElasticTransform(alpha=1, sigma=50, p=0.3)
            ], additional_targets={'mask': 'mask'})

            try:
                augmented_img_mask = transform(image=img, mask=mask)
                img = augmented_img_mask["image"]
                mask = augmented_img_mask["mask"]
            except Exception as e:
                pass # if augmentation fail just process with original image and mask

        # Obraz: (H, W) -> (1, H, W), float32
        X = torch.from_numpy(img).unsqueeze(0).float()
        
        # Maska: (H, W), int64 (Long). 
        # UWAGA: Nie dodajemy unsqueeze(0)! CrossEntropyLoss chce (Batch, H, W) a nie (Batch, 1, H, W)
        y = torch.from_numpy(mask).long()
        
        return X, y

    def visualize(self, idx):
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        type = self.types[idx]

        print(f"Klasy w masce: {np.unique(mask)}, typ obserwacji: {type}") # Debugowanie

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        # maska - wyświetlamy z kolorami dla klas
        axes[1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()
