import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import rotate

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, mode, resolution, scenario='1'):
        """
        Args:
            data_dir (string): Directory with all the images.
            mode (string): One of 'train', 'val', or 'test'.
            resolution (int): Resolution of images, either 64 or 128.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.resolution = resolution
        self.scenario = scenario

        # Construct paths based on resolution and mode
        if scenario not in ['1', '2', '3', '4']:
            raise ValueError("Invalid scenario. Choose from '1', '2', '3', or '4'.")
        if self.scenario == '2':
            scenario = '1'

        if self.mode == 'train':
            self.cta_path = os.path.join(data_dir, f"training_{scenario}_{resolution}", "cta")
            self.annotation_path = os.path.join(data_dir, f"training_{scenario}_{resolution}", "annotation")

        elif self.mode == 'val':
            self.cta_path = os.path.join(data_dir, f"validation_{resolution}", "cta")
            self.annotation_path = os.path.join(data_dir, f"validation_{resolution}", "annotation")
        else:
            self.cta_path = os.path.join(data_dir, f"test_{resolution}", "cta")
            self.annotation_path = os.path.join(data_dir, f"test_{resolution}", "annotation")

        # List all files in the directory
        self.samples = [f for f in os.listdir(self.cta_path) if f.endswith('.img.nii.gz')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the filename
        img_name = self.samples[idx]
        label_name = img_name.replace('.img.nii.gz', '.label.nii.gz')

        # Load image and label
        img_path = os.path.join(self.cta_path, img_name)
        label_path = os.path.join(self.annotation_path, label_name)

        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        affine = nib.load(img_path).affine

        # Convert to channel-first (C, D, H, W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        
        # Normalize image to [0, 1]
        min_val = image.min()
        max_val = image.max()

        if max_val != min_val:
            image[image > max_val] = max_val
            image[image < min_val] = min_val
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        # Scale to [-1, 1]
        image = 2 * image - 1

        # Apply transformations
        if self.scenario == '2':
            image, label = self.augment_image(image, label)
            
            image = image.copy()
            label = label.copy()
        return {'image': torch.tensor(image, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.long),
                'affine': affine}

    def augment_image(self, image, label):
        """
        Apply random augmentations to the image and label.

        Args:
            image (numpy array): The input image data (C, D, H, W).
            label (numpy array): The corresponding label data (C, D, H, W).

        Returns:
            tuple: Augmented image and label.
        """
        # Random flip along each axis
        if random.random() > 0.3:
            image = np.flip(image, axis=1)  # Flip depth
            label = np.flip(label, axis=1)
        if random.random() > 0.3:
            image = np.flip(image, axis=2)  # Flip height
            label = np.flip(label, axis=2)
        if random.random() > 0.3:
            image = np.flip(image, axis=3)  # Flip width
            label = np.flip(label, axis=3)

        # Random rotation
        if random.random() > 0.3:
            # Rotate the image slightly (e.g., by ±10 degrees)
            angle = random.uniform(-10, 10)
            image = self.rotate_volume(image, angle)
            label = self.rotate_volume(label, angle, is_label=True)

        # Random intensity adjustment
        if random.random() > 0.3:
            factor = random.uniform(0.9, 1.1)
            image *= factor

        return image, label

    def rotate_volume(self, volume, angle, is_label=False):
        """
        Rotate a 3D volume around the Z-axis.

        Args:
            volume (numpy array): The input volume (C, D, H, W).
            angle (float): The angle to rotate by in degrees.
            is_label (bool): Whether the volume is a label map (nearest neighbor interpolation).

        Returns:
            numpy array: The rotated volume.
        """

        # Use nearest neighbor interpolation for labels to avoid interpolation artifacts
        order = 0 if is_label else 1
        rotated = rotate(volume, angle, axes=(2, 3), reshape=False, order=order, mode='nearest')
        return rotated

def get_dataloaders(data_dir, resolution, train_batch_size=1, val_batch_size=1, test_batch_size=1, scenario='1'):
    
    # Create datasets
    train_dataset = MedicalImageDataset(data_dir=data_dir, mode='train', resolution=resolution, scenario=scenario)
    val_dataset = MedicalImageDataset(data_dir=data_dir, mode='val', resolution=resolution, scenario=scenario)
    test_dataset = MedicalImageDataset(data_dir=data_dir, mode='test', resolution=resolution, scenario=scenario)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
