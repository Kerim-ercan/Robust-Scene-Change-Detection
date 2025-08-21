import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


import robust_scene_change_detect.models as models

from py_utils import utils_img, utils_torch


# Load your fine-tuned model
# Step 1: Load base model architecture
my_model = models.get_model_from_pretrained("dino_2Cross_DiffCMU")

# Step 2: Load your fine-tuned checkpoint
checkpoint_path = "/home/divit/projects/Robust-Scene/Robust-Scene-Change-Detection/output/2025-08-21.10-52-58/best.val.pth"
checkpoint = torch.load(checkpoint_path, map_location='cuda')

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys if present (DataParallel training)"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

# Step 3: Apply fine-tuned weights to the model
try:
    # Try different checkpoint structure possibilities
    if 'model_state_dict' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        missing_keys, unexpected_keys = my_model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded from 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['state_dict'])
        missing_keys, unexpected_keys = my_model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded from 'state_dict'")
    elif 'model' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['model'])
        missing_keys, unexpected_keys = my_model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded from 'model'")
    else:
        # If checkpoint is directly the state dict
        state_dict = remove_module_prefix(checkpoint)
        missing_keys, unexpected_keys = my_model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded direct state dict")
    
    # Report any missing or unexpected keys
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} (these will use pretrained weights)")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (these will be ignored)")
        
    my_model = my_model.eval()
    my_model = my_model.cuda()
    print("🎉 Fine-tuned model loaded successfully on top of dino_2Cross_DiffCMU!")
    
    # Show training info if available
    if 'epoch' in checkpoint:
        print(f"📊 Trained for {checkpoint['epoch']} epochs")
    if 'best_metric' in checkpoint:
        print(f"📈 Best validation metric: {checkpoint['best_metric']:.4f}")
        
except Exception as e:
    print(f"❌ Error loading fine-tuned weights: {e}")
    print("\n🔍 Checkpoint structure:")
    print("Available keys:", list(checkpoint.keys()))
    
    # Fallback to pretrained model
    print("🔄 Using pretrained model instead...")
    my_model = my_model.eval().cuda()


# Load your own images (without ROI masking)
# Replace these paths with your own image paths
my_image_root = "/home/divit/projects/photos/504"  # Change this to your image directory

# Load your image pair
my_t0 = plt.imread(os.path.join(my_image_root, "in.jpg"))[..., :3]  # Replace with your image names
my_t1 = plt.imread(os.path.join(my_image_root, "in.jpg"))[..., :3]  # Replace with your image names

# Display your images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(my_t0)
ax[1].imshow(my_t1)
ax[0].set_title("Your Image T0")
ax[1].set_title("Your Image T1")

# remove axis
for a in ax.ravel():
    a.axis("off")

plt.tight_layout()

# Check image dimensions (should be divisible by 14 for DinoV2)
H, W = my_t0.shape[:2]
print(f"Image dimensions: {H}x{W}")
print(f"H % 14: {H % 14}, W % 14: {W % 14}")


# Inference on your images (no ROI masking)
# Update model upsampling size for your images
my_model.module.upsample.size = (H, W)

# Convert to tensors
my_tensor_0 = torch.from_numpy(my_t0).permute(2, 0, 1).unsqueeze(0)
my_tensor_0 = my_tensor_0.float().cuda()

my_tensor_1 = torch.from_numpy(my_t1).permute(2, 0, 1).unsqueeze(0)
my_tensor_1 = my_tensor_1.float().cuda()

# Run inference
with torch.no_grad():
    my_pred_0 = my_model(my_tensor_0, my_tensor_1)  # 1, H, W, 2
    my_pred_1 = my_model(my_tensor_1, my_tensor_0)  # 1, H, W, 2

    my_pred_0 = torch.argmax(my_pred_0, dim=-1).squeeze().cpu().numpy()
    my_pred_1 = torch.argmax(my_pred_1, dim=-1).squeeze().cpu().numpy()

# Visualize results
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(my_t0)
ax[0, 1].imshow(my_t1)
ax[1, 0].imshow(utils_img.overlay_image(my_t0, [1, 0, 0], mask=my_pred_0))
ax[1, 1].imshow(utils_img.overlay_image(my_t1, [1, 0, 0], mask=my_pred_1))

ax[0, 0].set_title("Your Image T0")
ax[0, 1].set_title("Your Image T1")
ax[1, 0].set_title("Change map on T0")
ax[1, 1].set_title("Change map on T1")

# remove axis
for a in ax.ravel():
    a.axis("off")

plt.tight_layout()