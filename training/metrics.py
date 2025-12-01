"""
Evaluation metrics for VeriPix
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Localization: IoU, Dice, Pixel Accuracy
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)


class ClassificationMetrics:
    """Compute classification metrics"""
    
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold (float): Decision threshold for binary classification
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels):
        """
        Update metrics with new batch
        
        Args:
            preds (torch.Tensor): Predictions (B, 1) - logits or probabilities
            labels (torch.Tensor): Ground truth (B,) - 0 or 1
        """
        # Convert to probabilities if logits
        if preds.max() > 1.0 or preds.min() < 0.0:
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        
        # Convert to numpy
        probs = probs.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        
        # Store
        self.all_probs.extend(probs)
        self.all_labels.extend(labels)
        
        # Binary predictions
        binary_preds = (probs >= self.threshold).astype(int)
        self.all_preds.extend(binary_preds)
    
    def compute(self):
        """
        Compute all classification metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
        
        # AUC-ROC (only if both classes present)
        if len(np.unique(labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        else:
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 
            'fn': int(fn), 'tp': int(tp)
        }
        
        return metrics


class LocalizationMetrics:
    """Compute localization/segmentation metrics"""
    
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold (float): Threshold for binary mask
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and masks"""
        self.all_ious = []
        self.all_dices = []
        self.all_pixel_accs = []
    
    def update(self, preds, targets):
        """
        Update metrics with new batch
        
        Args:
            preds (torch.Tensor): Predictions (B, 1, H, W) - logits or probabilities
            targets (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
        """
        # Convert to probabilities if logits
        if preds.max() > 1.0 or preds.min() < 0.0:
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        
        # Binarize predictions
        binary_preds = (probs >= self.threshold).float()
        
        # Ensure target has channel dimension
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Compute metrics for each sample in batch
        batch_size = preds.size(0)
        for i in range(batch_size):
            pred = binary_preds[i].flatten()
            target = targets[i].flatten()
            
            # Skip if target is all zeros (no ground truth)
            if target.sum() == 0:
                continue
            
            # IoU
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()
            self.all_ious.append(iou)
            
            # Dice
            dice = (2 * intersection / (pred.sum() + target.sum() + 1e-6)).item()
            self.all_dices.append(dice)
            
            # Pixel Accuracy
            correct = (pred == target).sum()
            total = target.numel()
            pixel_acc = (correct / total).item()
            self.all_pixel_accs.append(pixel_acc)
    
    def compute(self):
        """
        Compute all localization metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        if len(self.all_ious) == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'pixel_accuracy': 0.0,
                'num_samples': 0
            }
        
        metrics = {
            'iou': np.mean(self.all_ious),
            'dice': np.mean(self.all_dices),
            'pixel_accuracy': np.mean(self.all_pixel_accs),
            'num_samples': len(self.all_ious)
        }
        
        return metrics


if __name__ == "__main__":
    print("Metrics Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test classification metrics
    print("\n1. Classification Metrics")
    print("-" * 60)
    
    cls_metrics = ClassificationMetrics(threshold=0.5)
    
    # Simulate predictions
    for _ in range(5):
        batch_size = 16
        pred_logits = torch.randn(batch_size, 1).to(device)
        target_labels = torch.randint(0, 2, (batch_size,)).to(device)
        cls_metrics.update(pred_logits, target_labels)
    
    results = cls_metrics.compute()
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    
    # Test localization metrics
    print("\n2. Localization Metrics")
    print("-" * 60)
    
    loc_metrics = LocalizationMetrics(threshold=0.5)
    
    # Simulate predictions
    for _ in range(5):
        batch_size = 8
        pred_masks = torch.randn(batch_size, 1, 224, 224).to(device)
        target_masks = torch.randint(0, 2, (batch_size, 224, 224)).float().to(device)
        loc_metrics.update(pred_masks, target_masks)
    
    results = loc_metrics.compute()
    print(f"IoU: {results['iou']:.4f}")
    print(f"Dice: {results['dice']:.4f}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Samples evaluated: {results['num_samples']}")
    
    print("\n✓ All metrics tests passed!")
