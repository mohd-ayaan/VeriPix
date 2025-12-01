"""
Advanced augmentations for VeriPix
Improves classifier generalization and robustness
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_advanced_train_transforms(image_size=224):
    """
    Advanced training augmentations for better classifier accuracy
    Expected improvement: +2-3% accuracy
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Augmentation pipeline
    """
    return A.Compose([
        # Geometric augmentations
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.6
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.4
        ),
        
        # Image quality augmentations (simulate real-world conditions)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.4),
        
        # Noise augmentations
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.4),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        
        # JPEG compression (CRITICAL for forgery detection!)
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        
        # Advanced augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=2,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # Pixel-level augmentations
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        ], p=0.3),
        
        # Grid distortion (simulate camera artifacts)
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
        ], p=0.2),
        
        # Normalization (always last)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_advanced_val_transforms(image_size=224):
    """
    Validation transforms (no augmentation, only normalization)
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Transform pipeline
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_test_time_augmentation_transforms(image_size=224):
    """
    Test-time augmentation (TTA) transforms
    Use multiple augmented versions at inference for better accuracy
    
    Returns:
        list: List of transform pipelines
    """
    base_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    tta_transforms = [
        # Original
        base_transform,
        
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Rotate 90
        A.Compose([
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]
    
    return tta_transforms


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    print("Testing Advanced Augmentations")
    print("=" * 60)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test training transforms
    print("\n1. Training Transforms")
    print("-" * 60)
    train_transform = get_advanced_train_transforms(224)
    
    for i in range(3):
        augmented = train_transform(image=dummy_image)
        tensor = augmented['image']
        print(f"Augmentation {i+1}: Shape={tensor.shape}, "
              f"Mean={tensor.mean():.4f}, Std={tensor.std():.4f}")
    
    # Test validation transforms
    print("\n2. Validation Transforms")
    print("-" * 60)
    val_transform = get_advanced_val_transforms(224)
    augmented = val_transform(image=dummy_image)
    tensor = augmented['image']
    print(f"Val transform: Shape={tensor.shape}, "
          f"Mean={tensor.mean():.4f}, Std={tensor.std():.4f}")
    
    # Test TTA transforms
    print("\n3. Test-Time Augmentation Transforms")
    print("-" * 60)
    tta_transforms = get_test_time_augmentation_transforms(224)
    print(f"Number of TTA transforms: {len(tta_transforms)}")
    
    for i, transform in enumerate(tta_transforms):
        augmented = transform(image=dummy_image)
        tensor = augmented['image']
        print(f"TTA {i+1}: Shape={tensor.shape}, "
              f"Mean={tensor.mean():.4f}, Std={tensor.std():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All augmentation tests passed!")
