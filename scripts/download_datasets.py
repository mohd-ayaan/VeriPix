"""
Dataset Download Script for VeriPix
Downloads CASIA v2.0 and CoMoFoD datasets
"""

import os
import gdown
import zipfile
import requests
from tqdm import tqdm
import argparse


def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def download_casia_v2(output_dir="data/raw"):
    """Download CASIA v2.0 dataset from Kaggle"""
    print("=" * 50)
    print("Downloading CASIA v2.0 Dataset")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nManual download required:")
    print("1. Visit: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset")
    print("2. Download the dataset (requires Kaggle account)")
    print("3. Extract to: data/raw/casia_v2/")
    print("\nAlternative: Use Kaggle API")
    print("kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset")
    
    # Check if already downloaded
    casia_path = os.path.join(output_dir, "casia_v2")
    if os.path.exists(casia_path):
        print(f"\n✓ CASIA v2.0 already exists at {casia_path}")
        return True
    
    return False


def download_comofod(output_dir="data/raw"):
    """Download CoMoFoD dataset"""
    print("\n" + "=" * 50)
    print("Downloading CoMoFoD Dataset")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nManual download required:")
    print("1. Visit: https://www.vcl.fer.hr/comofod/")
    print("2. Download all image sets (Small 256, Large 512)")
    print("3. Extract to: data/raw/comofod/")
    
    # Alternative: Kaggle source
    print("\nAlternative Kaggle source:")
    print("kaggle datasets download -d tusharchauhan1898/comofod")
    
    comofod_path = os.path.join(output_dir, "comofod")
    if os.path.exists(comofod_path):
        print(f"\n✓ CoMoFoD already exists at {comofod_path}")
        return True
    
    return False


def verify_datasets(data_dir="data/raw"):
    """Verify dataset integrity"""
    print("\n" + "=" * 50)
    print("Verifying Datasets")
    print("=" * 50)
    
    casia_path = os.path.join(data_dir, "casia_v2")
    comofod_path = os.path.join(data_dir, "comofod")
    
    issues = []
    
    # Check CASIA v2
    if not os.path.exists(casia_path):
        issues.append("CASIA v2.0 not found")
    else:
        # Count files
        authentic_path = os.path.join(casia_path, "Au")
        tampered_path = os.path.join(casia_path, "Tp")
        
        if os.path.exists(authentic_path):
            auth_count = len([f for f in os.listdir(authentic_path) if f.endswith(('.jpg', '.png', '.bmp'))])
            print(f"✓ CASIA Authentic images: {auth_count}")
        else:
            issues.append("CASIA authentic folder not found")
            
        if os.path.exists(tampered_path):
            tamp_count = len([f for f in os.listdir(tampered_path) if f.endswith(('.jpg', '.png', '.bmp'))])
            print(f"✓ CASIA Tampered images: {tamp_count}")
        else:
            issues.append("CASIA tampered folder not found")
    
    # Check CoMoFoD
    if not os.path.exists(comofod_path):
        issues.append("CoMoFoD not found")
    else:
        comofod_files = len([f for f in os.listdir(comofod_path) if f.endswith(('.png', '.jpg'))])
        print(f"✓ CoMoFoD images: {comofod_files}")
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All datasets verified successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download VeriPix datasets")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing datasets")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_datasets(args.output_dir)
    else:
        download_casia_v2(args.output_dir)
        download_comofod(args.output_dir)
        verify_datasets(args.output_dir)


if __name__ == "__main__":
    main()
