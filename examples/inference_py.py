import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import robust_scene_change_detect.models as models
from py_utils import utils_img, utils_torch

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys if present (DataParallel training)"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_fine_tuned_model(checkpoint_path, base_model_name="dino_2Cross_DiffCMU"):
    """
    Load fine-tuned model with proper error handling and debugging
    """
    print(f"🔄 Loading base model: {base_model_name}")
    
    # Step 1: Load base model architecture
    try:
        my_model = models.get_model_from_pretrained(base_model_name)
        print(f"✓ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return None
    
    # Step 2: Load checkpoint
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        # Try different device mappings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # Debug: Print checkpoint structure
    print("\n🔍 Checkpoint structure:")
    if isinstance(checkpoint, dict):
        print("Available keys:", list(checkpoint.keys()))
        
        # Check for common checkpoint keys
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                print(f"  - Found '{key}' with {len(checkpoint[key])} parameters")
    else:
        print("Checkpoint is directly a state_dict")
    
    # Step 3: Apply fine-tuned weights to the model
    try:
        # Try different checkpoint structure possibilities
        state_dict = None
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📦 Using 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("📦 Using 'state_dict'")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("📦 Using 'model'")
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint
                print("📦 Using entire checkpoint as state_dict")
        else:
            state_dict = checkpoint
            print("📦 Using direct state_dict")
        
        # Remove module prefix if present
        state_dict = remove_module_prefix(state_dict)
        
        # Load state dict with detailed reporting
        missing_keys, unexpected_keys = my_model.load_state_dict(state_dict, strict=False)
        
        print("✓ Fine-tuned weights loaded successfully!")
        
        # Report any issues
        if missing_keys:
            print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
            print("   These parameters will use pretrained weights")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
            print("   These parameters will be ignored")
        
        # Show training info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"📊 Training completed at epoch: {checkpoint['epoch']}")
            if 'best_metric' in checkpoint:
                print(f"📈 Best validation metric: {checkpoint['best_metric']:.4f}")
            if 'loss' in checkpoint:
                print(f"📉 Final loss: {checkpoint['loss']:.4f}")
        
        # Set model to evaluation mode and move to device
        my_model.eval()
        if torch.cuda.is_available():
            my_model = my_model.cuda()
            print("🚀 Model moved to CUDA")
        
        return my_model
        
    except Exception as e:
        print(f"❌ Error loading fine-tuned weights: {e}")
        print("🔄 Falling back to pretrained model...")
        
        # Fallback to pretrained model
        my_model.eval()
        if torch.cuda.is_available():
            my_model = my_model.cuda()
        return my_model

def prepare_images(image_path1, image_path2):
    """
    Load and prepare image pairs for inference
    """
    if not os.path.exists(image_path1):
        print(f"❌ Image not found: {image_path1}")
        return None, None
    if not os.path.exists(image_path2):
        print(f"❌ Image not found: {image_path2}")
        return None, None
    
    # Load images
    img_t0 = plt.imread(image_path1)[..., :3]
    img_t1 = plt.imread(image_path2)[..., :3]
    
    print(f"✓ Images loaded: {img_t0.shape}, {img_t1.shape}")
    
    # Check if images have the same dimensions
    if img_t0.shape != img_t1.shape:
        print("⚠️  Images have different dimensions, this might cause issues")
    
    # Check dimensions (should be divisible by 14 for DinoV2)
    H, W = img_t0.shape[:2]
    print(f"📐 Image dimensions: {H}x{W}")
    if H % 14 != 0 or W % 14 != 0:
        print(f"⚠️  Warning: Dimensions not divisible by 14 (H%14={H%14}, W%14={W%14})")
        print("   This might affect model performance")
    
    return img_t0, img_t1

def run_inference(model, img_t0, img_t1):
    """
    Run inference on image pair
    """
    H, W = img_t0.shape[:2]
    
    # Update model upsampling size for your images
    try:
        if hasattr(model, 'module') and hasattr(model.module, 'upsample'):
            model.module.upsample.size = (H, W)
        elif hasattr(model, 'upsample'):
            model.upsample.size = (H, W)
        else:
            print("⚠️  Could not find upsample layer to update size")
    except Exception as e:
        print(f"⚠️  Warning: Could not update upsample size: {e}")
    
    # Convert to tensors and normalize to [0, 1] if needed
    def prepare_tensor(img):
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        # Normalize to [0, 1] if values are in [0, 255]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.cuda() if torch.cuda.is_available() else tensor
    
    tensor_t0 = prepare_tensor(img_t0)
    tensor_t1 = prepare_tensor(img_t1)
    
    print(f"📊 Input tensor shapes: {tensor_t0.shape}")
    print(f"📊 Input tensor ranges: T0=[{tensor_t0.min():.3f}, {tensor_t0.max():.3f}], T1=[{tensor_t1.min():.3f}, {tensor_t1.max():.3f}]")
    
    # Run inference
    with torch.no_grad():
        try:
            pred_0 = model(tensor_t0, tensor_t1)  # Changes from T0 to T1
            pred_1 = model(tensor_t1, tensor_t0)  # Changes from T1 to T0
            
            print(f"✓ Inference completed. Output shapes: {pred_0.shape}")
            
            # Convert to binary masks
            pred_0 = torch.argmax(pred_0, dim=-1).squeeze().cpu().numpy()
            pred_1 = torch.argmax(pred_1, dim=-1).squeeze().cpu().numpy()
            
            return pred_0, pred_1
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None, None

def visualize_results(img_t0, img_t1, pred_0, pred_1):
    """
    Visualize the results
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    ax[0, 0].imshow(img_t0)
    ax[0, 1].imshow(img_t1)
    
    # Overlay change maps
    try:
        ax[1, 0].imshow(utils_img.overlay_image(img_t0, [1, 0, 0], mask=pred_0))
        ax[1, 1].imshow(utils_img.overlay_image(img_t1, [1, 0, 0], mask=pred_1))
    except:
        # Fallback visualization if utils_img.overlay_image is not available
        ax[1, 0].imshow(pred_0, cmap='Reds', alpha=0.7)
        ax[1, 1].imshow(pred_1, cmap='Reds', alpha=0.7)
    
    ax[0, 0].set_title("Image T0")
    ax[0, 1].set_title("Image T1") 
    ax[1, 0].set_title("Change map on T0")
    ax[1, 1].set_title("Change map on T1")
    
    # Remove axis
    for a in ax.ravel():
        a.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"📊 Change detection statistics:")
    print(f"   T0->T1 changes: {np.sum(pred_0)} pixels ({np.mean(pred_0)*100:.2f}%)")
    print(f"   T1->T0 changes: {np.sum(pred_1)} pixels ({np.mean(pred_1)*100:.2f}%)")

# Main execution
if __name__ == "__main__":
    # Configuration
    checkpoint_path = "/home/divit/projects/Robust-Scene/Robust-Scene-Change-Detection/output/2025-08-21.10-52-58/best.val.pth"
    image_root = "/home/divit/projects/photos/504"
    image_name_t0 = "in.jpg"  # Change this to your actual image names
    image_name_t1 = "in.jpg"  # Change this to your actual image names
    
    # Full image paths
    image_path_t0 = os.path.join(image_root, image_name_t0)
    image_path_t1 = os.path.join(image_root, image_name_t1)
    
    print("🚀 Starting inference with fine-tuned model...")
    print("=" * 60)
    
    # Step 1: Load model
    model = load_fine_tuned_model(checkpoint_path)
    if model is None:
        print("❌ Failed to load model, exiting...")
        exit(1)
    
    print("\n" + "=" * 60)
    
    # Step 2: Prepare images
    img_t0, img_t1 = prepare_images(image_path_t0, image_path_t1)
    if img_t0 is None or img_t1 is None:
        print("❌ Failed to load images, exiting...")
        exit(1)
    
    # Step 3: Run inference
    print("\n🔄 Running inference...")
    pred_0, pred_1 = run_inference(model, img_t0, img_t1)
    
    if pred_0 is not None and pred_1 is not None:
        print("✓ Inference completed successfully!")
        
        # Step 4: Visualize results
        visualize_results(img_t0, img_t1, pred_0, pred_1)
    else:
        print("❌ Inference failed")