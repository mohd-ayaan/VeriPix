"""
Data Preparation Script for VeriPix v3.0
Split-level mask organization for better portability
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import re


class DatasetOrganizer:
    def __init__(self, base_dir="data", include_transformations=True):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.include_transformations = include_transformations
        
        # Dataset paths
        self.casia_dir = self.raw_dir / "CASIA2"
        self.comofod_dir = self.raw_dir / "CoMoFoD_small_v2"
        
        # CoMoFoD naming patterns
        self.comofod_pattern = re.compile(r'(\d{3})_(O|F|B|M)(?:_([A-Z]{2}\d))?\.png')
        
    def parse_comofod_filename(self, filename):
        """
        Parse CoMoFoD filename to extract set number, type, and transformation
        
        Returns:
            dict with keys: set_num, type, transformation, is_base
        """
        match = self.comofod_pattern.match(filename)
        if not match:
            return None
        
        set_num, file_type, transformation = match.groups()
        
        return {
            'set_num': set_num,
            'type': file_type,  # O, F, B, M
            'transformation': transformation,  # BC1, JC5, etc. (None if base)
            'is_base': transformation is None,
            'filename': filename
        }
    
    def verify_raw_datasets(self):
        """Verify that raw datasets exist"""
        print("=" * 60)
        print("Verifying Raw Datasets")
        print("=" * 60)
        
        issues = []
        
        # Check CASIA2
        if not self.casia_dir.exists():
            issues.append(f"CASIA2 not found at {self.casia_dir}")
        else:
            au_path = self.casia_dir / "Au"
            tp_path = self.casia_dir / "Tp"
            gt_path = self.casia_dir / "CASIA 2 Groundtruth"
            
            if au_path.exists():
                au_count = len(list(au_path.glob("*.*")))
                print(f"✓ CASIA2 Authentic: {au_count} images")
            else:
                issues.append("CASIA2 Au/ folder not found")
                
            if tp_path.exists():
                tp_count = len(list(tp_path.glob("*.*")))
                print(f"✓ CASIA2 Tampered: {tp_count} images")
            else:
                issues.append("CASIA2 Tp/ folder not found")
                
            if gt_path.exists():
                gt_count = len(list(gt_path.glob("*.*")))
                print(f"✓ CASIA2 Ground Truth: {gt_count} masks")
            else:
                print("⚠ CASIA2 Ground Truth folder not found (optional)")
        
        # Check CoMoFoD
        if not self.comofod_dir.exists():
            issues.append(f"CoMoFoD_small_v2 not found at {self.comofod_dir}")
        else:
            all_files = list(self.comofod_dir.glob("*.png"))
            print(f"✓ CoMoFoD_small_v2: {len(all_files)} total files")
            
            # Parse and categorize
            parsed_files = [self.parse_comofod_filename(f.name) for f in all_files]
            parsed_files = [p for p in parsed_files if p is not None]
            
            originals = len([p for p in parsed_files if p['type'] == 'O'])
            forged = len([p for p in parsed_files if p['type'] == 'F'])
            binary_masks = len([p for p in parsed_files if p['type'] == 'B'])
            colored_masks = len([p for p in parsed_files if p['type'] == 'M'])
            
            base_forged = len([p for p in parsed_files if p['type'] == 'F' and p['is_base']])
            transformed = len([p for p in parsed_files if p['type'] == 'F' and not p['is_base']])
            
            print(f"  - Original images (O): {originals}")
            print(f"  - Forged images (F): {forged}")
            print(f"    • Base forgeries: {base_forged}")
            print(f"    • Transformed forgeries: {transformed}")
            print(f"  - Binary masks (B): {binary_masks}")
            print(f"  - Colored masks (M): {colored_masks}")
        
        if issues:
            print("\n⚠ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✓ All datasets verified!")
            return True
    
    def organize_casia2(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """Organize CASIA2 dataset into train/val/test splits"""
        print("\n" + "=" * 60)
        print("Organizing CASIA2 Dataset")
        print("=" * 60)
        
        au_path = self.casia_dir / "Au"
        tp_path = self.casia_dir / "Tp"
        gt_path = self.casia_dir / "CASIA 2 Groundtruth"
        
        # Get all files (including various extensions)
        authentic_files = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.bmp']:
            authentic_files.extend(list(au_path.glob(ext)))
        authentic_files = sorted(authentic_files)
        
        tampered_files = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.bmp']:
            tampered_files.extend(list(tp_path.glob(ext)))
        tampered_files = sorted(tampered_files)
        
        print(f"Found {len(authentic_files)} authentic images")
        print(f"Found {len(tampered_files)} tampered images")
        
        # Create splits
        random.seed(42)
        random.shuffle(authentic_files)
        random.shuffle(tampered_files)
        
        def split_files(files, train_r, val_r, test_r):
            n = len(files)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)
            return {
                'train': files[:train_end],
                'val': files[train_end:val_end],
                'test': files[val_end:]
            }
        
        auth_splits = split_files(authentic_files, train_ratio, val_ratio, test_ratio)
        tamp_splits = split_files(tampered_files, train_ratio, val_ratio, test_ratio)
        
        # Copy files to organized structure
        dataset_info = defaultdict(lambda: {'authentic': 0, 'tampered': 0, 'masks': 0})
        
        for split in ['train', 'val', 'test']:
            # Create directories - SPLIT-LEVEL MASKS
            (self.processed_dir / split / "authentic").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / "tampered").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)
            
            # Copy authentic images (NO MASKS)
            print(f"\nProcessing {split} split - authentic images...")
            for img_path in tqdm(auth_splits[split], desc=f"CASIA2 {split} authentic"):
                dest = self.processed_dir / split / "authentic" / f"casia_{img_path.name}"
                shutil.copy2(img_path, dest)
                dataset_info[split]['authentic'] += 1
            
            # Copy tampered images WITH masks
            print(f"Processing {split} split - tampered images...")
            for img_path in tqdm(tamp_splits[split], desc=f"CASIA2 {split} tampered"):
                # Copy tampered image
                dest = self.processed_dir / split / "tampered" / f"casia_{img_path.name}"
                shutil.copy2(img_path, dest)
                dataset_info[split]['tampered'] += 1
                
                # Copy mask to SPLIT-LEVEL masks/ folder
                if gt_path.exists():
                    base_name = img_path.stem
                    possible_mask_names = [
                        f"{base_name}_gt.png",
                        f"{base_name}_gt.jpg",
                        f"{base_name}.png",
                        f"{base_name}.jpg"
                    ]
                    
                    for mask_name in possible_mask_names:
                        mask_path = gt_path / mask_name
                        if mask_path.exists():
                            # SPLIT-LEVEL: processed/train/masks/
                            mask_dest = self.processed_dir / split / "masks" / f"casia_{img_path.stem}.png"
                            shutil.copy2(mask_path, mask_dest)
                            dataset_info[split]['masks'] += 1
                            break
        
        # Print summary
        print("\n" + "=" * 60)
        print("CASIA2 Organization Summary")
        print("=" * 60)
        for split in ['train', 'val', 'test']:
            print(f"{split.upper()}:")
            print(f"  Authentic: {dataset_info[split]['authentic']} (no masks)")
            print(f"  Tampered: {dataset_info[split]['tampered']}")
            print(f"  Masks: {dataset_info[split]['masks']} (for tampered only)")
        
        return dataset_info
    
    def organize_comofod(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """
        Organize CoMoFoD_small_v2 dataset with split-level masks
        """
        print("\n" + "=" * 60)
        print("Organizing CoMoFoD_small_v2 Dataset")
        print(f"Include transformations: {self.include_transformations}")
        print("=" * 60)
        
        # Get all files
        all_files = list(self.comofod_dir.glob("*.png"))
        
        # Parse filenames
        parsed_data = defaultdict(lambda: {'O': [], 'F': [], 'B': [], 'M': []})
        
        for file_path in all_files:
            parsed = self.parse_comofod_filename(file_path.name)
            if parsed:
                set_num = parsed['set_num']
                file_type = parsed['type']
                parsed_data[set_num][file_type].append((file_path, parsed))
        
        # Separate original and forged images
        original_images = []
        forged_images = []
        
        for set_num, files in parsed_data.items():
            # Original images (type O)
            for file_path, parsed in files['O']:
                if self.include_transformations or parsed['is_base']:
                    original_images.append((file_path, parsed))
            
            # Forged images (type F)
            for file_path, parsed in files['F']:
                if self.include_transformations or parsed['is_base']:
                    # Find corresponding binary mask (type B)
                    mask_path = None
                    for mask_file, mask_parsed in files['B']:
                        # Match set number and transformation
                        if mask_parsed['transformation'] == parsed['transformation']:
                            mask_path = mask_file
                            break
                    
                    forged_images.append((file_path, parsed, mask_path))
        
        print(f"Found {len(original_images)} original images")
        print(f"Found {len(forged_images)} forged images")
        
        # Create splits
        random.seed(42)
        random.shuffle(original_images)
        random.shuffle(forged_images)
        
        def split_files(files, train_r, val_r, test_r):
            n = len(files)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)
            return {
                'train': files[:train_end],
                'val': files[train_end:val_end],
                'test': files[val_end:]
            }
        
        orig_splits = split_files(original_images, train_ratio, val_ratio, test_ratio)
        forged_splits = split_files(forged_images, train_ratio, val_ratio, test_ratio)
        
        dataset_info = defaultdict(lambda: {'authentic': 0, 'tampered': 0, 'masks': 0})
        
        for split in ['train', 'val', 'test']:
            # Create directories - SPLIT-LEVEL MASKS
            (self.processed_dir / split / "authentic").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / "tampered").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)
            
            # Copy original images (NO MASKS)
            print(f"\nProcessing {split} split - original images...")
            for img_path, parsed in tqdm(orig_splits[split], desc=f"CoMoFoD {split} original"):
                dest_name = f"comofod_{parsed['filename']}"
                dest = self.processed_dir / split / "authentic" / dest_name
                shutil.copy2(img_path, dest)
                dataset_info[split]['authentic'] += 1
            
            # Copy forged images WITH masks
            print(f"Processing {split} split - forged images...")
            for img_path, parsed, mask_path in tqdm(forged_splits[split], desc=f"CoMoFoD {split} forged"):
                # Copy forged image
                dest_name = f"comofod_{parsed['filename']}"
                dest = self.processed_dir / split / "tampered" / dest_name
                shutil.copy2(img_path, dest)
                dataset_info[split]['tampered'] += 1
                
                # Copy mask to SPLIT-LEVEL masks/ folder
                if mask_path and mask_path.exists():
                    mask_dest_name = f"comofod_{parsed['set_num']}_F"
                    if parsed['transformation']:
                        mask_dest_name += f"_{parsed['transformation']}"
                    mask_dest_name += ".png"
                    
                    # SPLIT-LEVEL: processed/train/masks/
                    mask_dest = self.processed_dir / split / "masks" / mask_dest_name
                    shutil.copy2(mask_path, mask_dest)
                    dataset_info[split]['masks'] += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("CoMoFoD Organization Summary")
        print("=" * 60)
        for split in ['train', 'val', 'test']:
            print(f"{split.upper()}:")
            print(f"  Authentic: {dataset_info[split]['authentic']} (no masks)")
            print(f"  Tampered: {dataset_info[split]['tampered']}")
            print(f"  Masks: {dataset_info[split]['masks']} (for tampered only)")
        
        return dataset_info
    
    def create_dataset_summary(self):
        """Create comprehensive dataset summary"""
        print("\n" + "=" * 60)
        print("FINAL DATASET SUMMARY")
        print("=" * 60)
        
        total_stats = defaultdict(lambda: {'authentic': 0, 'tampered': 0, 'masks': 0})
        
        for split in ['train', 'val', 'test']:
            auth_dir = self.processed_dir / split / "authentic"
            tamp_dir = self.processed_dir / split / "tampered"
            mask_dir = self.processed_dir / split / "masks"
            
            if auth_dir.exists():
                total_stats[split]['authentic'] = len(list(auth_dir.glob("*")))
            if tamp_dir.exists():
                total_stats[split]['tampered'] = len(list(tamp_dir.glob("*")))
            if mask_dir.exists():
                total_stats[split]['masks'] = len(list(mask_dir.glob("*")))
        
        # Print table
        print(f"\n{'Split':<10} {'Authentic':<15} {'Tampered':<15} {'Masks':<15} {'Total Images':<15}")
        print("-" * 75)
        
        grand_total = {'authentic': 0, 'tampered': 0, 'masks': 0}
        
        for split in ['train', 'val', 'test']:
            stats = total_stats[split]
            total = stats['authentic'] + stats['tampered']
            print(f"{split.upper():<10} {stats['authentic']:<15} {stats['tampered']:<15} "
                  f"{stats['masks']:<15} {total:<15}")
            
            grand_total['authentic'] += stats['authentic']
            grand_total['tampered'] += stats['tampered']
            grand_total['masks'] += stats['masks']
        
        print("-" * 75)
        total_images = grand_total['authentic'] + grand_total['tampered']
        print(f"{'TOTAL':<10} {grand_total['authentic']:<15} {grand_total['tampered']:<15} "
              f"{grand_total['masks']:<15} {total_images:<15}")
        
        print("\n📊 Key Statistics:")
        print(f"  • Authentic images: {grand_total['authentic']} (no masks - unaltered)")
        print(f"  • Tampered images: {grand_total['tampered']}")
        print(f"  • Ground truth masks: {grand_total['masks']} (for tampered only)")
        print(f"  • Mask coverage: {grand_total['masks']/grand_total['tampered']*100:.1f}% of tampered images")
        
        print("\n" + "=" * 60)
        print("✓ Dataset preparation complete!")
        print("=" * 60)
        print("\n📁 Final Directory Structure:")
        print("  data/processed/")
        print("  ├── train/")
        print("  │   ├── authentic/   (clean images, no masks)")
        print("  │   ├── tampered/    (forged images)")
        print("  │   └── masks/       (masks for tampered only)")
        print("  ├── val/")
        print("  │   ├── authentic/")
        print("  │   ├── tampered/")
        print("  │   └── masks/")
        print("  └── test/")
        print("      ├── authentic/")
        print("      ├── tampered/")
        print("      └── masks/")
        
        print("\n💡 Usage Note:")
        print("  - Authentic images have NO corresponding masks (nothing to highlight)")
        print("  - Tampered images have masks in same split's masks/ folder")
        print("  - Mask filename matches tampered image filename (1:1 mapping)")
        
        return total_stats


def main():
    parser = argparse.ArgumentParser(description="Prepare VeriPix datasets v3.0")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--skip_casia", action="store_true", help="Skip CASIA2 organization")
    parser.add_argument("--skip_comofod", action="store_true", help="Skip CoMoFoD organization")
    parser.add_argument("--no_transformations", action="store_true", 
                       help="Exclude CoMoFoD transformations (only base forgeries)")
    args = parser.parse_args()
    
    # Validate ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 0.01, \
        "Train/val/test ratios must sum to 1.0"
    
    # Initialize organizer
    organizer = DatasetOrganizer(
        args.data_dir, 
        include_transformations=not args.no_transformations
    )
    
    # Verify datasets
    if not organizer.verify_raw_datasets():
        print("\n⚠ Please check dataset paths!")
        return
    
    # Organize datasets
    if not args.skip_casia:
        organizer.organize_casia2(args.train_ratio, args.val_ratio, args.test_ratio)
    
    if not args.skip_comofod:
        organizer.organize_comofod(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Create summary
    organizer.create_dataset_summary()
    
    print("\n✅ Your datasets are now ready for training!")
    print(f"   Processed data location: {organizer.processed_dir}")


if __name__ == "__main__":
    main()
