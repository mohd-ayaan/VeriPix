"""
Comprehensive evaluation script for VeriPix
Evaluates both classifier and localizer on test set
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_classifier import create_efficientnet_classifier
from models.unet_localizer import create_unet_localizer
from utils.dataset import create_dataloaders
from training.metrics import ClassificationMetrics, LocalizationMetrics


class ModelEvaluator:
    """Evaluate trained models on test set"""
    
    def __init__(self, config, classifier_path=None, localizer_path=None):
        """
        Args:
            config (dict): Configuration dictionary
            classifier_path (str): Path to classifier checkpoint
            localizer_path (str): Path to localizer checkpoint
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load classifier
        if classifier_path:
            print("Loading classifier...")
            self.classifier = create_efficientnet_classifier(
                pretrained=False,
                dropout=config['classifier']['dropout'],
                device=self.device
            )
            checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.eval()
            print(f"  ✓ Loaded from {classifier_path}")
        else:
            self.classifier = None
        
        # Load localizer
        if localizer_path:
            print("Loading localizer...")
            self.localizer = create_unet_localizer(
                n_channels=3,
                n_classes=1,
                device=self.device
            )
            checkpoint = torch.load(localizer_path, map_location=self.device, weights_only=False)
            self.localizer.load_state_dict(checkpoint['model_state_dict'])
            self.localizer.eval()
            print(f"  ✓ Loaded from {localizer_path}")
        else:
            self.localizer = None
        
        # Load test dataloader
        print("Loading test data...")
        dataloaders = create_dataloaders(
            data_dir=config['dataset']['processed_path'],
            batch_size=config['training']['classifier']['batch_size'],
            num_workers=config['hardware']['num_workers'],
            use_ela=config.get('use_ela', True),
            image_size=config['image']['size'][0]
        )
        self.test_loader = dataloaders['test']
        print(f"  ✓ Test samples: {len(self.test_loader.dataset)}")

    
    def evaluate_classifier(self):
        """Evaluate classifier on test set"""
        if self.classifier is None:
            print("No classifier loaded, skipping...")
            return None
        
        print("\n" + "=" * 60)
        print("Evaluating Classifier")
        print("=" * 60)
        
        metrics = ClassificationMetrics(threshold=0.5)
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing Classifier"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.classifier(images)
                probs = torch.sigmoid(outputs)
                
                # Update metrics
                metrics.update(outputs, labels)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        results = metrics.compute()
        
        # Print results
        print(f"\nClassification Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  AUC-ROC: {results['auc_roc']:.4f}")
        print(f"  Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"    TN: {cm['tn']}, FP: {cm['fp']}")
        print(f"    FN: {cm['fn']}, TP: {cm['tp']}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(results['confusion_matrix'])
        
        # Plot ROC curve
        self._plot_roc_curve(all_labels, all_probs, results['auc_roc'])
        
        # Save results
        with open(self.results_dir / 'classifier_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_localizer(self):
        """Evaluate localizer on test set"""
        if self.localizer is None:
            print("No localizer loaded, skipping...")
            return None
        
        print("\n" + "=" * 60)
        print("Evaluating Localizer")
        print("=" * 60)
        
        metrics = LocalizationMetrics(threshold=0.5)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing Localizer"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                has_mask = batch['has_mask']
                
                # Skip if no ground truth masks
                if not has_mask.any():
                    continue
                
                # Filter to samples with masks
                mask_indices = has_mask.nonzero(as_tuple=True)[0]
                images = images[mask_indices]
                masks = masks[mask_indices]
                
                # Forward pass
                outputs = self.localizer(images)
                
                # Update metrics
                metrics.update(outputs, masks)
        
        # Compute metrics
        results = metrics.compute()
        
        # Print results
        print(f"\nLocalization Results:")
        print(f"  IoU: {results['iou']:.4f}")
        print(f"  Dice: {results['dice']:.4f}")
        print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"  Samples evaluated: {results['num_samples']}")
        
        # Save results
        with open(self.results_dir / 'localizer_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Authentic', 'Tampered'],
                   yticklabels=['Authentic', 'Tampered'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved confusion matrix to {self.results_dir / 'confusion_matrix.png'}")
    
    def _plot_roc_curve(self, labels, probs, auc_score):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curve.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved ROC curve to {self.results_dir / 'roc_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VeriPix Models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--classifier', type=str, default='checkpoints/classifier/best.pth', 
                       help='Classifier checkpoint')
    parser.add_argument('--localizer', type=str, default='checkpoints/localizer/best.pth',
                       help='Localizer checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        config,
        classifier_path=args.classifier if Path(args.classifier).exists() else None,
        localizer_path=args.localizer if Path(args.localizer).exists() else None
    )
    
    # Evaluate
    cls_results = evaluator.evaluate_classifier()
    loc_results = evaluator.evaluate_localizer()
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to {evaluator.results_dir}")


if __name__ == "__main__":
    main()
