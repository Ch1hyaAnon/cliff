#!/usr/bin/env python3
"""
Simple CLIFF Model Outputs Demonstration

This script provides a simple explanation of the CLIFF model outputs
without requiring all dependencies to be installed.

此脚本提供CLIFF模型输出的简单说明，无需安装所有依赖项。
"""

import torch
import numpy as np


def demonstrate_cliff_outputs():
    """
    Demonstrate the CLIFF model outputs with explanations
    演示CLIFF模型输出及其说明
    """
    print("=" * 80)
    print("CLIFF Model Outputs Demonstration")
    print("CLIFF模型输出演示")
    print("=" * 80)
    
    # Simulate the three outputs from CLIFF model
    batch_size = 2
    print(f"Demonstrating with batch_size = {batch_size}")
    print(f"演示批次大小 = {batch_size}")
    
    # Create dummy outputs with the correct shapes
    pred_rotmat = torch.randn(batch_size, 24, 3, 3)
    pred_betas = torch.randn(batch_size, 10) 
    pred_cam_crop = torch.randn(batch_size, 3)
    
    print(f"\nModel outputs / 模型输出:")
    print(f"pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)")
    
    print(f"\n" + "=" * 60)
    print("OUTPUT ANALYSIS / 输出分析")
    print("=" * 60)
    
    # 1. pred_rotmat analysis
    print(f"\n1. pred_rotmat (旋转矩阵):")
    print(f"   Shape: {pred_rotmat.shape}")
    print(f"   Meaning: SMPL pose parameters as rotation matrices")
    print(f"   含义: SMPL姿态参数，以旋转矩阵形式表示")
    print(f"   ")
    print(f"   Structure breakdown / 结构分解:")
    print(f"   - Dimension 0: Batch size ({pred_rotmat.shape[0]} people)")
    print(f"   - Dimension 1: SMPL joints ({pred_rotmat.shape[1]} joints)")
    print(f"   - Dimensions 2,3: 3x3 rotation matrix for each joint")
    print(f"   - 维度0: 批次大小 ({pred_rotmat.shape[0]} 个人)")
    print(f"   - 维度1: SMPL关节 ({pred_rotmat.shape[1]} 个关节)")
    print(f"   - 维度2,3: 每个关节的3x3旋转矩阵")
    
    print(f"   ")
    print(f"   Joint mapping / 关节映射:")
    print(f"   - pred_rotmat[:, 0]    -> Global orientation (root joint)")
    print(f"   - pred_rotmat[:, 1:23] -> Body pose (23 body joints)")
    print(f"   - pred_rotmat[:, 0]    -> 全局方向（根关节）")
    print(f"   - pred_rotmat[:, 1:23] -> 身体姿态（23个身体关节）")
    
    # Show example values for first person's root joint
    print(f"   ")
    print(f"   Example - First person's root joint rotation matrix:")
    print(f"   示例 - 第一个人的根关节旋转矩阵:")
    root_rotation = pred_rotmat[0, 0].detach().cpu().numpy()
    for i, row in enumerate(root_rotation):
        print(f"   [{row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f}]")
    
    # 2. pred_betas analysis  
    print(f"\n2. pred_betas (形状参数):")
    print(f"   Shape: {pred_betas.shape}")
    print(f"   Meaning: SMPL shape parameters (body shape coefficients)")
    print(f"   含义: SMPL形状参数（身体形状系数）")
    print(f"   ")
    print(f"   Details / 详细信息:")
    print(f"   - 10 PCA coefficients that define human body shape")
    print(f"   - Controls height, weight, body proportions")
    print(f"   - Values typically range from -3 to +3 standard deviations")
    print(f"   - 10个PCA系数，定义人体形状")
    print(f"   - 控制身高、体重、身体比例")
    print(f"   - 数值通常在-3到+3标准差范围内")
    
    print(f"   ")
    print(f"   Example - First person's shape parameters:")
    print(f"   示例 - 第一个人的形状参数:")
    betas_example = pred_betas[0].detach().cpu().numpy()
    print(f"   {betas_example}")
    
    # 3. pred_cam_crop analysis
    print(f"\n3. pred_cam_crop (裁剪相机参数):")
    print(f"   Shape: {pred_cam_crop.shape}")
    print(f"   Meaning: Weak perspective camera parameters in crop coordinates")
    print(f"   含义: 裁剪坐标系中的弱透视相机参数")
    print(f"   ")
    print(f"   Format: [scale, tx, ty] / 格式: [缩放, x平移, y平移]")
    print(f"   - scale (s): Controls the apparent size of the person")
    print(f"   - tx: Horizontal translation in crop coordinates")
    print(f"   - ty: Vertical translation in crop coordinates")
    print(f"   - 缩放 (s): 控制人物的视觉大小")
    print(f"   - tx: 裁剪坐标系中的水平平移") 
    print(f"   - ty: 裁剪坐标系中的垂直平移")
    
    # Show example values
    for i in range(batch_size):
        s, tx, ty = pred_cam_crop[i]
        print(f"   ")
        print(f"   Person {i+1} camera parameters / 第{i+1}个人的相机参数:")
        print(f"   - Scale: {s.item():.4f}")
        print(f"   - TX: {tx.item():.4f}")  
        print(f"   - TY: {ty.item():.4f}")
    
    print(f"\n" + "=" * 60)
    print("USAGE IN SMPL MODEL / SMPL模型中的使用")
    print("=" * 60)
    
    print(f"\nTo use these outputs with SMPL model:")
    print(f"要在SMPL模型中使用这些输出:")
    print(f"""
# Step 1: Convert camera from crop to full coordinates
# 步骤1: 将相机从裁剪坐标转换为完整坐标
pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, 
                              full_img_shape, focal_length)

# Step 2: Use with SMPL model
# 步骤2: 与SMPL模型一起使用
pred_output = smpl_model(
    betas=pred_betas,                    # Shape parameters / 形状参数
    body_pose=pred_rotmat[:, 1:],        # Body joint rotations / 身体关节旋转
    global_orient=pred_rotmat[:, [0]],   # Root rotation / 根旋转  
    pose2rot=False,                      # Already rotation matrices / 已是旋转矩阵
    transl=pred_cam_full                 # 3D translation / 3D平移
)

# Step 3: Get 3D vertices and joints
# 步骤3: 获取3D顶点和关节
vertices = pred_output.vertices   # 3D mesh vertices / 3D网格顶点
joints = pred_output.joints       # 3D joint positions / 3D关节位置
""")

    print(f"\n" + "=" * 60)
    print("COORDINATE SYSTEM NOTES / 坐标系说明")
    print("=" * 60)
    
    print(f"""
Important: Camera parameter conversion / 重要：相机参数转换

pred_cam_crop is in CROPPED image coordinates:
pred_cam_crop处于裁剪图像坐标系中:
- Origin at center of crop region / 原点在裁剪区域中心
- Normalized to crop size / 归一化到裁剪大小

pred_cam_full is in FULL image coordinates:  
pred_cam_full处于完整图像坐标系中:
- Origin at full image center / 原点在完整图像中心
- Accounts for crop position and scale / 考虑裁剪位置和缩放

The conversion is essential for proper 3D reconstruction!
转换对于正确的3D重建至关重要！
""")


def demonstrate_tensor_properties():
    """Show tensor properties and operations"""
    print(f"\n" + "=" * 60)
    print("TENSOR PROPERTIES / 张量属性")
    print("=" * 60)
    
    # Create example tensors
    pred_rotmat = torch.randn(2, 24, 3, 3)
    pred_betas = torch.randn(2, 10)
    pred_cam_crop = torch.randn(2, 3)
    
    print(f"\nTensor properties / 张量属性:")
    print(f"pred_rotmat:")
    print(f"  - dtype: {pred_rotmat.dtype}")
    print(f"  - device: {pred_rotmat.device}")  
    print(f"  - requires_grad: {pred_rotmat.requires_grad}")
    print(f"  - memory: {pred_rotmat.nelement() * pred_rotmat.element_size()} bytes")
    
    print(f"\nCommon operations / 常用操作:")
    print(f"# Extract global orientation / 提取全局方向")
    print(f"global_orient = pred_rotmat[:, [0]]  # Shape: (batch, 1, 3, 3)")
    
    print(f"\n# Extract body pose / 提取身体姿态")  
    print(f"body_pose = pred_rotmat[:, 1:]       # Shape: (batch, 23, 3, 3)")
    
    print(f"\n# Access individual parameters / 访问单个参数")
    print(f"scale = pred_cam_crop[:, 0]          # Scale for each person")
    print(f"tx = pred_cam_crop[:, 1]             # X translation")
    print(f"ty = pred_cam_crop[:, 2]             # Y translation")


if __name__ == "__main__":
    print("CLIFF Model Outputs Simple Demonstration")
    print("CLIFF模型输出简单演示\n")
    
    try:
        demonstrate_cliff_outputs()
        demonstrate_tensor_properties()
        
        print(f"\n" + "=" * 80)
        print("SUCCESS: Demonstration completed!")
        print("成功: 演示完成!")
        print("For more details, see CLIFF_MODEL_OUTPUTS.md")
        print("更多详情请参见 CLIFF_MODEL_OUTPUTS.md")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print(f"演示过程中出错: {e}")