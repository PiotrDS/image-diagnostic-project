import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A

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
    new_mask[mask >= 200] = 1
    
    # 2. Mapowanie Dysków (Discs) -> Klasa 2
    # Zakładam, że dyski to małe numery (widzę 1-8 w Twoich logach)
    # Zazwyczaj dyski są numerowane sekwencyjnie.
    new_mask[(mask >= 1) & (mask < 100)] = 2
    
    # 3. Mapowanie Kości Krzyżowej (Sacrum) -> Klasa 3
    # W logach widzę "100". Często okrągłe liczby oznaczają inne struktury.
    # Jeśli Sacrum to nie 100, to prawdopodobnie jeden z tych > 200.
    # Ale spróbujmy tak:
    new_mask[mask == 100] = 3
    
    return new_mask

def load_img_and_mask(img_path, mask_path, des_size):
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
    
    def __init__(self, if_trasform=True):
        self.samples_img = []
        self.samples_mask = []
        self.augmented = []

        # ZMIANA 5: Nowy rozmiar z artykułu
        des_size = (512, 512)

        path_1_images = f"..{os.sep}data{os.sep}data_1{os.sep}images"
        path_1_masks = f"..{os.sep}data{os.sep}data_1{os.sep}masks"

        if if_trasform:
            transform = A.Compose([
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
                            A.GaussNoise(p=0.2),
                            A.ElasticTransform(alpha=1, sigma=50, p=0.3)
            ], additional_targets={'mask': 'mask'})

        # --- Wczytywanie DATA 1 ---
        if os.path.exists(path_1_images):
            for image, mask in zip(sorted(os.listdir(path_1_images)), sorted(os.listdir(path_1_masks))):
                full_image_path = os.path.join(path_1_images, image)
                full_mask_path = os.path.join(path_1_masks, mask)

                try:
                    img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size)
                    self.samples_img.append(img)
                    self.samples_mask.append(mask)
                    self.augmented.append(False)

                    if if_trasform:
                        for _ in range(8):
                            try:
                                # Albumentations obsłuży maskę poprawnie (nie interpoluje liczb całkowitych w zły sposób)
                                augmented_img_mask = transform(image=img, mask=mask)
                                aug_img = augmented_img_mask["image"]
                                aug_mask = augmented_img_mask["mask"]
                                    
                                self.samples_img.append(aug_img)
                                self.samples_mask.append(aug_mask)
                                self.augmented.append(True)
                            except Exception as e:
                                continue

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

                try:
                    img, mask = load_img_and_mask(full_image_path, full_mask_path, des_size)

                    self.samples_img.append(img)
                    self.samples_mask.append(mask)
                    self.augmented.append(False)

                    if if_trasform:
                        for _ in range(5):
                            try:
                                augmented_img_mask = transform(image=img, mask=mask)
                                aug_img = augmented_img_mask["image"]
                                aug_mask = augmented_img_mask["mask"]
                                
                                self.samples_img.append(aug_img)
                                self.samples_mask.append(aug_mask)
                                self.augmented.append(True)
                            except Exception as e:
                                continue

                except Exception as e:
                    print(e)
                    print("couldn't load ", full_image_path, " or ", full_mask_path)           
    
    def __len__(self):
        return len(self.samples_img)

    def __getitem__(self, idx):
        
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        
        # Obraz: (H, W) -> (1, H, W), float32
        X = torch.from_numpy(img).unsqueeze(0).float()
        
        # Maska: (H, W), int64 (Long). 
        # UWAGA: Nie dodajemy unsqueeze(0)! CrossEntropyLoss chce (Batch, H, W) a nie (Batch, 1, H, W)
        y = torch.from_numpy(mask).long()
        
        return X, y

    def visualize(self, idx):
        img = self.samples_img[idx]
        mask = self.samples_mask[idx]
        ori = self.augmented[idx]

        print(f"Klasy w masce: {np.unique(mask)}") # Debugowanie

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        fig.suptitle(f"Visualization of idx={idx} original = {not(ori)}")

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        # maska - wyświetlamy z kolorami dla klas
        axes[1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        plt.show()