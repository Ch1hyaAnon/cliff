#!/usr/bin/env python3
"""
CLIFF Model Outputs Demonstration Script

This script demonstrates how to understand and work with the three main outputs
from the CLIFF model: pred_rotmat, pred_betas, and pred_cam_crop.

The script shows:
1. The shapes and meanings of each output
2. How to extract specific information from each output
3. How the outputs are used in the SMPL model
4. How camera parameters are converted from crop to full image coordinates

此脚本演示如何理解和使用CLIFF模型的三个主要输出：
pred_rotmat、pred_betas和pred_cam_crop。
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import CLIFF modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.cliff_res50.cliff import CLIFF as cliff_res50
    from models.cliff_hr48.cliff import CLIFF as cliff_hr48
    from common.utils import cam_crop2full
    from common.imutils import rot6d_to_rotmat
    from common import constants
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you are running this script from the CLIFF repository root.")
    sys.exit(1)


def demonstrate_model_outputs():
    """
    Demonstrate the CLIFF model outputs with dummy data
    演示CLIFF模型输出（使用虚拟数据）
    """
    print("=" * 80)
    print("CLIFF Model Outputs Demonstration")
    print("CLIFF模型输出演示")
    print("=" * 80)
    
    # Create a dummy model instance (we'll use ResNet50 version)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Note: This requires the SMPL mean parameters file
        if not os.path.exists(constants.SMPL_MEAN_PARAMS):
            print(f"Warning: SMPL mean parameters file not found at {constants.SMPL_MEAN_PARAMS}")
            print("Creating dummy outputs for demonstration purposes...")
            demonstrate_with_dummy_outputs()
            return
            
        model = cliff_res50(constants.SMPL_MEAN_PARAMS).to(device)
        model.eval()
        
        # Create dummy input data
        batch_size = 2
        img_height, img_width = constants.CROP_IMG_HEIGHT, constants.CROP_IMG_WIDTH
        
        # Dummy normalized image tensor
        norm_img = torch.randn(batch_size, 3, img_height, img_width).to(device)
        
        # Dummy bounding box info (normalized)
        bbox_info = torch.randn(batch_size, 3).to(device)
        
        print(f"\nInput shapes:")
        print(f"  norm_img: {norm_img.shape} - Normalized input image")
        print(f"  bbox_info: {bbox_info.shape} - Bounding box information [cx, cy, scale]")
        
        # Get model predictions
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info)
            
        analyze_outputs(pred_rotmat, pred_betas, pred_cam_crop, batch_size)
        
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Demonstrating with dummy outputs instead...")
        demonstrate_with_dummy_outputs()


def demonstrate_with_dummy_outputs():
    """
    Demonstrate outputs using dummy tensors when model cannot be loaded
    当模型无法加载时使用虚拟张量演示输出
    """
    batch_size = 2
    
    # Create dummy outputs with correct shapes
    pred_rotmat = torch.randn(batch_size, 24, 3, 3)
    pred_betas = torch.randn(batch_size, 10)
    pred_cam_crop = torch.randn(batch_size, 3)
    
    print(f"\nUsing dummy outputs for demonstration (batch_size = {batch_size}):")
    analyze_outputs(pred_rotmat, pred_betas, pred_cam_crop, batch_size)


def analyze_outputs(pred_rotmat, pred_betas, pred_cam_crop, batch_size):
    """
    Analyze and explain the model outputs
    分析并解释模型输出
    """
    print(f"\n" + "=" * 60)
    print("OUTPUT ANALYSIS / 输出分析")
    print("=" * 60)
    
    # 1. Analyze pred_rotmat
    print(f"\n1. pred_rotmat (Rotation Matrices / 旋转矩阵):")
    print(f"   Shape: {pred_rotmat.shape}")
    print(f"   Description: SMPL pose parameters as rotation matrices")
    print(f"   描述: SMPL姿态参数，以旋转矩阵形式表示")
    print(f"   - Batch size: {pred_rotmat.shape[0]}")
    print(f"   - Number of joints: {pred_rotmat.shape[1]} (24 SMPL joints)")
    print(f"   - Each joint: {pred_rotmat.shape[2]}x{pred_rotmat.shape[3]} rotation matrix")
    
    # Show global orientation vs body pose
    global_orient = pred_rotmat[:, [0]]  # First joint (root)
    body_pose = pred_rotmat[:, 1:]       # Remaining 23 joints
    print(f"   - Global orientation (全局方向): {global_orient.shape}")
    print(f"   - Body pose (身体姿态): {body_pose.shape}")
    
    # 2. Analyze pred_betas
    print(f"\n2. pred_betas (Shape Parameters / 形状参数):")
    print(f"   Shape: {pred_betas.shape}")
    print(f"   Description: SMPL shape parameters (PCA coefficients)")
    print(f"   描述: SMPL形状参数（PCA系数）")
    print(f"   - Batch size: {pred_betas.shape[0]}")
    print(f"   - Shape coefficients: {pred_betas.shape[1]} (10 PCA components)")
    print(f"   - Controls: body height, weight, proportions")
    print(f"   - 控制: 身高、体重、身体比例")
    
    # Show some example values
    if batch_size > 0:
        print(f"   Example values for first person:")
        print(f"   第一个人的示例值: {pred_betas[0, :5].detach().cpu().numpy()}")
    
    # 3. Analyze pred_cam_crop
    print(f"\n3. pred_cam_crop (Cropped Camera Parameters / 裁剪相机参数):")
    print(f"   Shape: {pred_cam_crop.shape}")
    print(f"   Description: Weak perspective camera in crop coordinates")
    print(f"   描述: 裁剪坐标系中的弱透视相机参数")
    print(f"   - Batch size: {pred_cam_crop.shape[0]}")
    print(f"   - Parameters: [scale, tx, ty]")
    print(f"   - 参数: [缩放, x平移, y平移]")
    
    if batch_size > 0:
        s, tx, ty = pred_cam_crop[0, 0], pred_cam_crop[0, 1], pred_cam_crop[0, 2]
        print(f"   Example for first person / 第一个人的示例:")
        print(f"     - Scale (缩放): {s.item():.4f}")
        print(f"     - Translation X (X平移): {tx.item():.4f}")
        print(f"     - Translation Y (Y平移): {ty.item():.4f}")
    
    # 4. Show how to use outputs
    print(f"\n" + "=" * 60)
    print("USAGE EXAMPLES / 使用示例")
    print("=" * 60)
    
    print(f"\n4. How to use with SMPL model / 如何与SMPL模型一起使用:")
    print(f"   ```python")
    print(f"   # Convert crop camera to full camera coordinates")
    print(f"   # 将裁剪相机坐标转换为完整相机坐标")
    print(f"   pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, ")
    print(f"                                 full_img_shape, focal_length)")
    print(f"   ")
    print(f"   # Use with SMPL model / 与SMPL模型一起使用")
    print(f"   pred_output = smpl_model(")
    print(f"       betas=pred_betas,                    # Shape parameters")
    print(f"       body_pose=pred_rotmat[:, 1:],        # Body joint rotations")
    print(f"       global_orient=pred_rotmat[:, [0]],   # Root rotation")
    print(f"       pose2rot=False,                      # Already rotation matrices")
    print(f"       transl=pred_cam_full                 # 3D translation")
    print(f"   )")
    print(f"   ```")
    
    print(f"\n5. Output tensor properties / 输出张量属性:")
    print(f"   - Device: {pred_rotmat.device}")
    print(f"   - Data type: {pred_rotmat.dtype}")
    print(f"   - Requires gradient: {pred_rotmat.requires_grad}")


def demonstrate_camera_conversion():
    """
    Demonstrate camera parameter conversion from crop to full coordinates
    演示相机参数从裁剪坐标到完整坐标的转换
    """
    print(f"\n" + "=" * 60)
    print("CAMERA CONVERSION EXAMPLE / 相机转换示例")
    print("=" * 60)
    
    # Create dummy camera conversion example
    batch_size = 1
    pred_cam_crop = torch.tensor([[0.5, 0.1, -0.05]])  # [scale, tx, ty]
    center = torch.tensor([[128, 256]])  # bbox center [cx, cy]
    scale = torch.tensor([1.2])  # bbox scale
    full_img_shape = torch.tensor([[512, 512]])  # [height, width]
    focal_length = torch.tensor([724.0])  # estimated focal length
    
    print(f"Input parameters / 输入参数:")
    print(f"  pred_cam_crop: {pred_cam_crop[0]} [scale, tx, ty]")
    print(f"  center: {center[0]} [cx, cy] - bbox center in full image")
    print(f"  scale: {scale[0]} - bbox scale factor") 
    print(f"  full_img_shape: {full_img_shape[0]} [height, width]")
    print(f"  focal_length: {focal_length[0]}")
    
    # Convert to full camera coordinates
    pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, 
                                  full_img_shape, focal_length)
    
    print(f"\nOutput / 输出:")
    print(f"  pred_cam_full: {pred_cam_full[0]} [tx, ty, tz]")
    print(f"  - tx, ty: 2D translation in full image coordinates")
    print(f"  - tz: depth (distance from camera)")
    print(f"  - tx, ty: 完整图像坐标中的2D平移")
    print(f"  - tz: 深度（距离相机的距离）")


if __name__ == "__main__":
    print("CLIFF Model Outputs Explanation")
    print("CLIFF模型输出说明")
    print("\nThis script demonstrates the three main outputs from CLIFF model:")
    print("此脚本演示CLIFF模型的三个主要输出：")
    print("1. pred_rotmat - SMPL pose as rotation matrices")
    print("2. pred_betas - SMPL shape parameters") 
    print("3. pred_cam_crop - Camera parameters in crop coordinates")
    print("\n1. pred_rotmat - SMPL姿态（旋转矩阵形式）")
    print("2. pred_betas - SMPL形状参数")
    print("3. pred_cam_crop - 裁剪坐标系中的相机参数")
    
    demonstrate_model_outputs()
    demonstrate_camera_conversion()
    
    print(f"\n" + "=" * 80)
    print("For more details, see CLIFF_MODEL_OUTPUTS.md")
    print("更多详情请参见 CLIFF_MODEL_OUTPUTS.md")
    print("=" * 80)