import os
import glob
import numpy as np
from PIL import Image


class YourCustomDataset:
    """
    Custom dataset class for your dataset structure:
    your_dataset/
    ├── train/
    │   ├── t0/
    │   ├── t1/
    │   └── mask/
    └── test/
        ├── t0/
        ├── t1/
        └── mask/
    """

    def __init__(self, root, mode="train"):
        
        assert mode in {"train", "test"}
        
        self.mode = mode
        self.root = root
        
        # Set paths for the dataset
        self.dataset_path = os.path.join(self.root, mode)
        self._t0 = os.path.join(self.dataset_path, "t0")
        self._t1 = os.path.join(self.dataset_path, "t1")
        self._mask = os.path.join(self.dataset_path, "mask")
        
        # Get all filenames from t0 folder
        filenames = glob.glob(os.path.join(self._t0, "*"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)
        
        self._filenames = np.array(filenames)
        
        # Verify that all folders have the same files
        self._verify_dataset_consistency()

    def _verify_dataset_consistency(self):
        """Verify that t0, t1, and mask folders have the same files"""
        
        t0_files = set(os.listdir(self._t0))
        t1_files = set(os.listdir(self._t1))
        mask_files = set(os.listdir(self._mask))
        
        if not (t0_files == t1_files == mask_files):
            raise ValueError("Files in t0, t1, and mask folders don't match!")
        
        print(f"Dataset loaded: {len(self._filenames)} images in {self.mode} set")

    @property
    def filenames(self):
        return self._filenames

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        
        filename = self._filenames[idx]
        
        t0_path = os.path.join(self._t0, filename)
        t1_path = os.path.join(self._t1, filename)
        mask_path = os.path.join(self._mask, filename)
        
        # Load images
        t0_image = Image.open(t0_path).convert("RGB")
        t1_image = Image.open(t1_path).convert("RGB")
        mask_image = Image.open(mask_path)
        
        # Convert to numpy arrays and normalize
        t0_image = np.array(t0_image) / 255.0
        t1_image = np.array(t1_image) / 255.0
        mask_image = np.array(mask_image) / 255.0
        
        # Convert mask to binary
        mask_image = mask_image > 0.0
        
        # Convert to float32
        t0_image = t0_image.astype(np.float32)
        t1_image = t1_image.astype(np.float32)
        mask_image = mask_image.astype(np.float32)
        
        return t0_image, t1_image, mask_image

    @property
    def figsize(self):
        # Return 504x504 as you mentioned you want to use this size
        return np.array([504, 504])