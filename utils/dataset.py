"""
PyTorch Dataset classes for VeriPix
Handles both classification and localization tasks
Supports multiple image formats: jpg, png, tif, bmp
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.ela import ELAProcessor


class ForgeryDataset(Dataset):
    """
    Unified dataset for image forgery detection
    Handles both classification and localization
    """
    
    def __init__(
        self,
        data_dir,
        transform=None,
        use_ela=True,
        ela_quality=90,
        ela_scale=10,
        return_mask=True,
        image_size=224
    ):
        """
        Args:
            data_dir (str or Path): Path to processed data split (e.g., data/processed/train)
            transform (albumentations.Compose): Augmentation pipeline
            use_ela (bool): Apply ELA preprocessing
            ela_quality (int): JPEG quality for ELA
            ela_scale (int): Amplification for ELA
            return_mask (bool): Return segmentation masks
            image_size (int): Image dimension (assumes square images)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_ela = use_ela
        self.return_mask = return_mask
        self.image_size = image_size
        
        # Initialize ELA processor
        if use_ela:
            self.ela_processor = ELAProcessor(quality=ela_quality, scale=ela_scale)
        
        # Get image paths
        self.authentic_dir = self.data_dir / "authentic"
        self.tampered_dir = self.data_dir / "tampered"
        self.mask_dir = self.data_dir / "masks"
        
        # Collect authentic images (all formats)
        self.authentic_images = []
        if self.authentic_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                self.authentic_images.extend(list(self.authentic_dir.glob(ext)))
        
        # Collect tampered images (all formats)
        self.tampered_images = []
        if self.tampered_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                self.tampered_images.extend(list(self.tampered_dir.glob(ext)))
        
        # Combine all images
        self.all_images = self.authentic_images + self.tampered_images
        
        # Create labels (0 = authentic, 1 = tampered)
        self.labels = [0] * len(self.authentic_images) + [1] * len(self.tampered_images)
        
        print(f"Dataset initialized from {self.data_dir}")
        print(f"  - Authentic: {len(self.authentic_images)}")
        print(f"  - Tampered: {len(self.tampered_images)}")
        print(f"  - Total: {len(self.all_images)}")
    
    def __len__(self):
        return len(self.all_images)
    
    def _load_image(self, img_path):
        """Load and preprocess image (handles jpg, png, tif, bmp)"""
        if self.use_ela:
            # Apply ELA (works with all formats via PIL)
            img = self.ela_processor.apply_ela(img_path, output_size=(self.image_size, self.image_size))
        else:
            # Load normally - use PIL for TIF support
            try:
                # Try PIL first (handles TIF better)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                img = np.array(img)
            except Exception as e:
                # Fallback to OpenCV
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Could not load image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size, self.image_size))
        
        return img
    
    def _load_mask(self, img_path, label):
        """Load segmentation mask or create blank mask"""
        if label == 0:
            # Authentic image - no mask (all black)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            # Tampered image - try to load mask
            # Try different extensions since mask might be .png but image is .tif
            possible_mask_names = [
                img_path.stem + ".png",
                img_path.stem + ".jpg",
                img_path.stem + ".tif",
                img_path.stem + ".tiff",
            ]
            
            mask = None
            for mask_name in possible_mask_names:
                mask_path = self.mask_dir / mask_name
                if mask_path.exists():
                    try:
                        # Use PIL for better format support
                        mask_img = Image.open(mask_path).convert('L')
                        mask = np.array(mask_img, dtype=np.float32)
                        mask = cv2.resize(mask, (self.image_size, self.image_size))
                        # Normalize to [0, 1] range (FIXED)
                        mask = mask / 255.0
                        # Binarize mask (threshold at 0.5)
                        mask = (mask > 0.5).astype(np.float32)
                        break
                    except Exception as e:
                        # Try next extension
                        continue
            
            if mask is None:
                # Mask not found - create black mask
                mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
        return mask

    
    def __getitem__(self, idx):
        """
        Returns:
            dict containing:
                - image: Tensor (C, H, W)
                - mask: Tensor (H, W) if return_mask=True
                - label: int (0=authentic, 1=tampered)
                - has_mask: bool (whether ground truth mask exists)
        """
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load mask
        mask = self._load_mask(img_path, label)
        has_mask = mask.max() > 0  # Check if mask is not all-black
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            mask = torch.from_numpy(mask).float() / 255.0
        
        result = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
            'path': str(img_path)
        }
        
        if self.return_mask:
            result['mask'] = mask
        
        return result


def get_train_transforms(image_size=224):
    """Training augmentations using Albumentations"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    """Validation/test transforms (no augmentation)"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=4,
    use_ela=True,
    image_size=224
):
    """
    Create train, val, test dataloaders
    
    Args:
        data_dir (str or Path): Base processed data directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        use_ela (bool): Apply ELA preprocessing
        image_size (int): Image size
        
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = ForgeryDataset(
        data_dir / "train",
        transform=get_train_transforms(image_size),
        use_ela=use_ela,
        image_size=image_size
    )
    
    val_dataset = ForgeryDataset(
        data_dir / "val",
        transform=get_val_transforms(image_size),
        use_ela=use_ela,
        image_size=image_size
    )
    
    test_dataset = ForgeryDataset(
        data_dir / "test",
        transform=get_val_transforms(image_size),
        use_ela=use_ela,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing ForgeryDataset")
    print("=" * 60)
    
    # Create dataset
    dataset = ForgeryDataset(
        data_dir="data/processed/train",
        transform=get_train_transforms(),
        use_ela=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading samples with different formats
    print("\nTesting sample loading (first 5 samples):")
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            img_path = Path(sample['path'])
            print(f"  [{i}] {img_path.name[:40]:40s} | ext={img_path.suffix:5s} | "
                  f"label={sample['label'].item()} | has_mask={sample['has_mask'].item()}")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
    
    # Test dataloader
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders("data/processed", batch_size=8, num_workers=0)
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Test batches: {len(dataloaders['test'])}")
    
    # Test one batch
    print("\nTesting train batch loading...")
    batch = next(iter(dataloaders['train']))
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Masks: {batch['mask'].shape}")
    print(f"  Label distribution: {batch['label'].sum().item()}/{len(batch['label'])} tampered")
    print(f"  Has mask count: {batch['has_mask'].sum().item()}/{len(batch['has_mask'])}")
    
    print("\n✓ All dataset tests passed!")
