"""
Error Level Analysis (ELA) Implementation
Detects JPEG compression artifacts to identify image manipulation
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
from pathlib import Path


class ELAProcessor:
    """
    Error Level Analysis processor for image forgery detection
    
    ELA works by:
    1. Saving image at known JPEG quality
    2. Computing difference between original and recompressed
    3. Amplifying differences to highlight manipulated regions
    """
    
    def __init__(self, quality=90, scale=10):
        """
        Args:
            quality (int): JPEG compression quality (1-100). Default 90.
            scale (int): Amplification factor for differences. Default 10.
        """
        self.quality = quality
        self.scale = scale
    
    def apply_ela(self, image_path, output_size=(224, 224)):
        """
        Apply ELA to an image
        
        Args:
            image_path (str or Path): Path to input image
            output_size (tuple): Output dimensions (height, width)
            
        Returns:
            np.ndarray: ELA image as numpy array (H, W, C), values 0-255
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        img = img.resize((output_size[1], output_size[0]), Image.LANCZOS)
        
        # Save to buffer with specified quality
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        
        # Load compressed version
        compressed = Image.open(buffer)
        
        # Compute difference
        ela_img = ImageChops.difference(img, compressed)
        
        # Get extrema for normalization
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Avoid division by zero
        if max_diff == 0:
            max_diff = 1
        
        # Scale and enhance
        scale_factor = 255.0 / max_diff * self.scale
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor)
        
        # Convert to numpy array
        ela_array = np.array(ela_img)
        
        # Clip values to valid range
        ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
        
        return ela_array
    
    def apply_ela_opencv(self, image_path, output_size=(224, 224)):
        """
        Alternative ELA implementation using OpenCV
        Faster but slightly different results
        
        Args:
            image_path (str or Path): Path to input image
            output_size (tuple): Output dimensions (height, width)
            
        Returns:
            np.ndarray: ELA image as numpy array (H, W, C)
        """
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (output_size[1], output_size[0]))
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        
        # Decode
        compressed = cv2.imdecode(encimg, 1)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        # Compute difference
        diff = cv2.absdiff(img, compressed)
        
        # Normalize and scale
        diff = diff.astype(np.float32)
        diff = diff * self.scale
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        return diff
    
    def visualize_ela(self, image_path, save_path=None):
        """
        Create visualization comparing original and ELA
        
        Args:
            image_path (str or Path): Path to input image
            save_path (str or Path, optional): Path to save visualization
            
        Returns:
            np.ndarray: Concatenated visualization
        """
        # Load original
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (224, 224))
        
        # Apply ELA
        ela = self.apply_ela(image_path)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, ela])
        
        if save_path:
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), comparison_bgr)
        
        return comparison


def batch_ela_processing(input_dir, output_dir, quality=90, scale=10):
    """
    Apply ELA to all images in a directory
    
    Args:
        input_dir (str or Path): Input directory
        output_dir (str or Path): Output directory
        quality (int): JPEG quality
        scale (int): Amplification factor
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = ELAProcessor(quality, scale)
    
    # Process all images
    for img_path in input_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            try:
                ela = processor.apply_ela(img_path)
                
                # Save
                output_path = output_dir / f"{img_path.stem}_ela.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(ela, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")


if __name__ == "__main__":
    # Test ELA processor
    print("ELA Processor Test")
    print("=" * 50)
    
    processor = ELAProcessor(quality=90, scale=10)
    
    # Find any test image from processed data
    from pathlib import Path
    
    test_dirs = [
        Path("data/processed/train/tampered"),
        Path("data/processed/train/authentic"),
        Path("data/processed/val/tampered"),
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if test_dir.exists():
            # Look for any image with any extension
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp']:
                images = list(test_dir.glob(ext))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if test_image and test_image.exists():
        print(f"✓ Test image found: {test_image.name}")
        print(f"  Extension: {test_image.suffix}")
        print(f"  Directory: {test_image.parent.name}")
        
        try:
            # Test PIL-based ELA
            ela_result = processor.apply_ela(test_image)
            print(f"\n✓ PIL-based ELA computed successfully")
            print(f"  Shape: {ela_result.shape}")
            print(f"  Dtype: {ela_result.dtype}")
            print(f"  Value range: [{ela_result.min()}, {ela_result.max()}]")
            
            # Test OpenCV-based ELA
            ela_cv = processor.apply_ela_opencv(test_image)
            print(f"\n✓ OpenCV-based ELA computed successfully")
            print(f"  Shape: {ela_cv.shape}")
            print(f"  Value range: [{ela_cv.min()}, {ela_cv.max()}]")
            
            # Save visualization
            vis_path = Path("results") / "ela_test_comparison.png"
            vis_path.parent.mkdir(exist_ok=True)
            processor.visualize_ela(test_image, vis_path)
            print(f"\n✓ Visualization saved to {vis_path}")
            
            # Test with different formats
            print("\n" + "=" * 50)
            print("Testing with multiple image formats...")
            format_tests = []
            
            for test_dir in test_dirs:
                if test_dir.exists():
                    for ext in ['*.jpg', '*.tif', '*.png']:
                        imgs = list(test_dir.glob(ext))
                        if imgs:
                            format_tests.append((imgs[0], ext.replace('*', '')))
                            if len(format_tests) >= 3:
                                break
                if len(format_tests) >= 3:
                    break
            
            for img_path, ext in format_tests:
                try:
                    ela = processor.apply_ela(img_path, output_size=(224, 224))
                    print(f"✓ {ext}: {img_path.name[:40]}... OK")
                except Exception as e:
                    print(f"✗ {ext}: {img_path.name[:40]}... ERROR: {e}")
            
            print("\n" + "=" * 50)
            print("✓ All ELA tests passed!")
            
        except Exception as e:
            print(f"\n✗ Error during ELA processing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ No test images found in processed directories")
        print("Please run 'python scripts/prepare_data.py' first")
        print("\nExpected directories:")
        for d in test_dirs:
            print(f"  - {d}")

