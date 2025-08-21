import glob
import os
import copy
import numpy as np
from PIL import Image


class CustomChangeDetectionDataset:
    """
    Custom dataset class for change detection with structure:
    dataset_root/
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
        """
        Args:
            root: Path to your dataset root directory
            mode: 'train' or 'test'
        """
        assert mode in {"train", "test"}, f"Mode must be 'train' or 'test', got {mode}"
        
        self.mode = mode
        self.root = os.path.join(root, mode)
        
        # Define paths
        self.t0_path = os.path.join(self.root, "t0")
        self.t1_path = os.path.join(self.root, "t1")
        self.mask_path = os.path.join(self.root, "mask")
        
        # Verify directories exist
        for path in [self.t0_path, self.t1_path, self.mask_path]:
            if not os.path.exists(path):
                raise ValueError(f"Directory not found: {path}")
        
        # Get all filenames from t0 directory
        filenames = glob.glob(os.path.join(self.t0_path, "*.png"))
        filenames = [os.path.basename(f) for f in filenames]
        filenames = sorted(filenames)
        
        # Verify that all files exist in all directories
        missing_files = []
        for filename in filenames:
            if not os.path.exists(os.path.join(self.t1_path, filename)):
                missing_files.append(f"t1/{filename}")
            if not os.path.exists(os.path.join(self.mask_path, filename)):
                missing_files.append(f"mask/{filename}")
        
        if missing_files:
            raise ValueError(f"Missing files: {missing_files}")
        
        self._filenames = np.array(filenames)
        print(f"Loaded {len(self._filenames)} samples for {mode} mode")

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        """Get a sample (t0_image, t1_image, mask_image)"""
        t0_image = self.get_img_0(idx)
        t1_image = self.get_img_1(idx)
        mask_image = self.get_mask(idx)
        return t0_image, t1_image, mask_image

    def get_img_0(self, idx):
        """Load t0 image"""
        filename = self._filenames[idx]
        path = os.path.join(self.t0_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_img_1(self, idx):
        """Load t1 image"""
        filename = self._filenames[idx]
        path = os.path.join(self.t1_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_mask(self, idx):
        """Load mask image"""
        filename = self._filenames[idx]
        path = os.path.join(self.mask_path, filename)
        mask_image = Image.open(path)
        # Convert to grayscale if needed
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')
        mask_image = np.array(mask_image) / 255.0
        mask_image = mask_image > 0.0  # Convert to binary
        mask_image = mask_image.astype(np.float32)
        return mask_image

    def loc(self, key):
        """Create a subset of the dataset"""
        other = copy.copy(self)
        other._filenames = self._filenames[key]
        return other

    @property
    def filenames(self):
        return self._filenames

    @property
    def figsize(self):
        """Return the figure size - you may need to adjust this based on your images"""
        # Get the size of the first image to determine figsize
        if len(self._filenames) > 0:
            sample_path = os.path.join(self.t0_path, self._filenames[0])
            with Image.open(sample_path) as img:
                height, width = img.size[1], img.size[0]  # PIL returns (width, height)
                return np.array([height, width])
        return np.array([512, 512])  # Default fallback