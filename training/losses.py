"""
Advanced Loss Functions for VeriPix
- Binary Cross-Entropy for classification
- Dice Loss + BCE for segmentation
- Focal Loss for class imbalance
- Tversky Loss for precision/recall control
- Combo Loss for optimal localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SEGMENTATION LOSSES
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Handles class imbalance better than BCE alone
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth (float): Smoothing constant to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits or probabilities
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply sigmoid if logits
        pred = torch.sigmoid(pred)
        
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Convert target to float
        target = target.float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Return Dice loss
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss for segmentation
    Balances pixel-wise accuracy (BCE) with region overlap (Dice)
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        """
        Args:
            dice_weight (float): Weight for Dice loss
            bce_weight (float): Weight for BCE loss
            smooth (float): Smoothing constant for Dice
        """
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Convert target to float
        target = target.float()
        
        # Compute individual losses
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    Allows control over false positives vs false negatives
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        Args:
            alpha (float): Weight for false positives (0-1)
            beta (float): Weight for false negatives (0-1)
            smooth (float): Smoothing constant
            
        Note:
            - alpha=beta=0.5 → Dice Loss
            - alpha>beta → Emphasize recall (reduce false negatives)
            - alpha<beta → Emphasize precision (reduce false positives)
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Tversky loss value
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Focuses on hard examples
    Combines Tversky and Focal approaches
    """
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.5, smooth=1.0):
        """
        Args:
            alpha (float): Tversky alpha parameter
            beta (float): Tversky beta parameter
            gamma (float): Focal parameter (higher = more focus on hard examples)
            smooth (float): Smoothing constant
        """
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Focal Tversky loss value
        """
        tversky = self.tversky_loss(pred, target)
        focal_tversky = torch.pow(tversky, self.gamma)
        
        return focal_tversky


class ComboLoss(nn.Module):
    """
    Combo Loss - Weighted combination of Dice and modified CE
    Excellent for localization tasks
    """
    
    def __init__(self, alpha=0.5, ce_weight=0.5, smooth=1.0):
        """
        Args:
            alpha (float): Controls weight between Dice and CE
            ce_weight (float): Weight for cross-entropy component
            smooth (float): Smoothing constant
        """
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_weight = ce_weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Combo loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Sigmoid for probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        # Dice component
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Modified Cross-Entropy component
        ce = F.binary_cross_entropy(pred_prob, target, reduction='mean')
        
        # Combo loss
        combo = (self.alpha * (1 - dice)) + (self.ce_weight * ce)
        
        return combo


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss - Best for severe class imbalance
    Combines Focal Loss with Dice Loss
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.6, focal_weight=0.4, smooth=1.0):
        """
        Args:
            alpha (float): Weight for positive class
            gamma (float): Focusing parameter
            dice_weight (float): Weight for Dice loss
            focal_weight (float): Weight for Focal loss
            smooth (float): Smoothing constant
        """
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1, H, W) - logits
            target (torch.Tensor): Ground truth (B, 1, H, W) or (B, H, W)
            
        Returns:
            torch.Tensor: Combined Focal + Dice loss
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Focal loss
        pred_prob = torch.sigmoid(pred)
        
        # Focal term
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # BCE with focal weighting
        bce = F.binary_cross_entropy(pred_prob, target, reduction='none')
        focal = alpha_t * focal_term * bce
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.focal_weight * focal.mean()
        
        return total_loss


# ============================================================================
# CLASSIFICATION LOSSES
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification
    Focuses training on hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for class balance
            gamma (float): Focusing parameter (higher = focus more on hard examples)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1) - logits
            target (torch.Tensor): Ground truth (B,) - 0 or 1
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float().unsqueeze(1), reduction='none')
        
        # Compute probabilities
        probs = torch.sigmoid(pred)
        
        # Compute focal term
        pt = torch.where(target.unsqueeze(1) == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        
        # Compute alpha term
        alpha_term = torch.where(target.unsqueeze(1) == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        loss = alpha_term * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for class imbalance
    """
    
    def __init__(self, pos_weight=None):
        """
        Args:
            pos_weight (float): Weight for positive class (tampered images)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1) - logits
            target (torch.Tensor): Ground truth (B,) - 0 or 1
            
        Returns:
            torch.Tensor: Weighted BCE loss
        """
        
        # Accepts both [B] and [B,1]
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=pred.device)
            return F.binary_cross_entropy_with_logits(
                pred, 
                target.float().unsqueeze(1), 
                pos_weight=pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(pred, target.float().unsqueeze(1))


class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing for BCE Loss
    Prevents overconfident predictions
    """
    
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing (float): Label smoothing factor (0.0 to 0.5)
        """
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 1) - logits
            target (torch.Tensor): Ground truth (B,) - 0 or 1
            
        Returns:
            torch.Tensor: Label-smoothed BCE loss
        """
        target = target.float().unsqueeze(1)
        
        # Apply label smoothing
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return F.binary_cross_entropy_with_logits(pred, target)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Advanced Loss Functions Test")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Test classification losses
    print("1. CLASSIFICATION LOSSES")
    print("-" * 70)
    
    batch_size = 8
    pred_logits = torch.randn(batch_size, 1).to(device)
    target_labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # BCE Loss
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(pred_logits, target_labels.float().unsqueeze(1))
    print(f"BCE Loss:                  {bce_loss.item():.4f}")
    
    # Focal Loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal(pred_logits, target_labels)
    print(f"Focal Loss:                {focal_loss.item():.4f}")
    
    # Weighted BCE
    weighted_bce = WeightedBCELoss(pos_weight=2.0)
    weighted_loss = weighted_bce(pred_logits, target_labels)
    print(f"Weighted BCE Loss:         {weighted_loss.item():.4f}")
    
    # Label Smoothing BCE
    ls_bce = LabelSmoothingBCELoss(smoothing=0.1)
    ls_loss = ls_bce(pred_logits, target_labels)
    print(f"Label Smoothing BCE Loss:  {ls_loss.item():.4f}")
    
    # Test segmentation losses
    print("\n2. SEGMENTATION LOSSES")
    print("-" * 70)
    
    pred_mask = torch.randn(batch_size, 1, 224, 224).to(device)
    target_mask = torch.randint(0, 2, (batch_size, 224, 224)).float().to(device)
    
    # Dice Loss
    dice = DiceLoss(smooth=1.0)
    dice_loss = dice(pred_mask, target_mask)
    print(f"Dice Loss:                 {dice_loss.item():.4f}")
    
    # Dice + BCE Loss
    dice_bce = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    combined_loss = dice_bce(pred_mask, target_mask)
    print(f"Dice + BCE Loss:           {combined_loss.item():.4f}")
    
    # Tversky Loss
    tversky = TverskyLoss(alpha=0.7, beta=0.3)
    tversky_loss = tversky(pred_mask, target_mask)
    print(f"Tversky Loss:              {tversky_loss.item():.4f}")
    
    # Focal Tversky Loss
    focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)
    ft_loss = focal_tversky(pred_mask, target_mask)
    print(f"Focal Tversky Loss:        {ft_loss.item():.4f}")
    
    # Combo Loss
    combo = ComboLoss(alpha=0.5, ce_weight=0.5)
    combo_loss = combo(pred_mask, target_mask)
    print(f"Combo Loss:                {combo_loss.item():.4f}")
    
    # Focal Dice Loss
    focal_dice = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.6, focal_weight=0.4)
    fd_loss = focal_dice(pred_mask, target_mask)
    print(f"Focal Dice Loss:           {fd_loss.item():.4f}")
    
    # Test gradients
    print("\n3. GRADIENT FLOW TEST")
    print("-" * 70)
    
    pred_mask.requires_grad = True
    loss = focal_dice(pred_mask, target_mask)
    loss.backward()
    
    print(f"Loss computed:             {loss.item():.4f}")
    print(f"Gradients computed:        {pred_mask.grad is not None}")
    print(f"Gradient range:            [{pred_mask.grad.min():.6f}, {pred_mask.grad.max():.6f}]")
    print(f"Gradient mean:             {pred_mask.grad.mean():.6f}")
    print(f"Gradient std:              {pred_mask.grad.std():.6f}")
    
    print("\n" + "=" * 70)
    print("✓ All loss function tests passed!")
    print("=" * 70)
    
    # Print recommendations
    print("\n📊 RECOMMENDED LOSS FUNCTIONS:")
    print("-" * 70)
    print("For Classifier:")
    print("  • Focal Loss (alpha=0.25, gamma=2.0) - Best for imbalanced data")
    print("  • Label Smoothing BCE - Prevents overconfident predictions")
    print("\nFor Localizer:")
    print("  • Focal Dice Loss - Best overall for severe imbalance")
    print("  • Focal Tversky Loss - Control precision/recall trade-off")
    print("  • Combo Loss - Good balance of pixel and region accuracy")
