"""
Inference script for VeriPix
Test classifier and localizer on single images
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent))

from models.efficientnet_classifier import create_efficientnet_classifier
from models.unet_localizer import create_unet_localizer
from preprocessing.ela import ELAProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VeriPixInference:
    """Inference pipeline for image forgery detection"""
    
    def __init__(self, config_path, classifier_path, localizer_path):
        """
        Args:
            config_path (str): Path to config file
            classifier_path (str): Path to classifier checkpoint
            localizer_path (str): Path to localizer checkpoint
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = self.config['image']['size'][0]
        
        # Initialize ELA processor
        self.ela_processor = ELAProcessor(
            quality=self.config['ela']['quality'],
            scale=self.config['ela']['scale']
        )
        
        # Load classifier
        print("Loading classifier...")
        self.classifier = create_efficientnet_classifier(
            pretrained=False,
            dropout=self.config['classifier']['dropout'],
            device=self.device
        )
        checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        print(f"  ✓ Classifier loaded")
        
        # Load localizer
        print("Loading localizer...")
        self.localizer = create_unet_localizer(
            n_channels=3,
            n_classes=1,
            device=self.device
        )
        checkpoint = torch.load(localizer_path, map_location=self.device, weights_only=False)
        self.localizer.load_state_dict(checkpoint['model_state_dict'])
        self.localizer.eval()
        print(f"  ✓ Localizer loaded")
        
        # Preprocessing transform
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            tuple: (original_image, preprocessed_tensor)
        """
        # Load original
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (self.image_size, self.image_size))
        
        # Apply ELA
        ela_image = self.ela_processor.apply_ela(image_path, output_size=(self.image_size, self.image_size))
        
        # Transform
        transformed = self.transform(image=ela_image)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return original, tensor
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict if image is tampered and localize manipulation
        
        Args:
            image_path (str): Path to input image
            threshold (float): Classification threshold
            
        Returns:
            dict: Prediction results
        """
        # Preprocess
        original, tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            # Classification
            cls_logits = self.classifier(tensor)
            cls_prob = torch.sigmoid(cls_logits).item()
            is_tampered = cls_prob >= threshold
            
            # Localization (only if tampered)
            if is_tampered:
                loc_logits = self.localizer(tensor)
                loc_prob = torch.sigmoid(loc_logits)
                mask = loc_prob.squeeze().cpu().numpy()
                mask = (mask >= threshold).astype(np.uint8) * 255
            else:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        return {
            'is_tampered': is_tampered,
            'confidence': cls_prob,
            'label': 'Tampered' if is_tampered else 'Authentic',
            'original_image': original,
            'mask': mask
        }
    
    def visualize_result(self, result, save_path=None):
        """
        Visualize prediction result
        
        Args:
            result (dict): Prediction result
            save_path (str, optional): Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(result['mask'], cmap='hot')
        axes[1].set_title('Tampering Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = result['original_image'].copy()
        if result['is_tampered']:
            # Create colored mask overlay
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = result['mask']  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"{result['label']} (Conf: {result['confidence']:.3f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="VeriPix Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--classifier', type=str, default='checkpoints/classifier/best.pth',
                       help='Classifier checkpoint')
    parser.add_argument('--localizer', type=str, default='checkpoints/localizer/best.pth',
                       help='Localizer checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Create inference pipeline
    inference = VeriPixInference(args.config, args.classifier, args.localizer)
    
    # Predict
    print(f"\nAnalyzing image: {args.image}")
    result = inference.predict(args.image, threshold=args.threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"  Status: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    if result['is_tampered']:
        tampered_pixels = (result['mask'] > 0).sum()
        total_pixels = result['mask'].size
        tampered_percent = (tampered_pixels / total_pixels) * 100
        print(f"  Tampered region: {tampered_percent:.2f}% of image")
    
    # Visualize
    output_path = args.output or f"results/inference_{Path(args.image).stem}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    inference.visualize_result(result, save_path=output_path)


if __name__ == "__main__":
    main()
