"""
Training script for U-Net localizer
Trains segmentation model to localize tampered regions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet_localizer import create_unet_localizer
from utils.dataset import create_dataloaders
from training.losses import DiceBCELoss, DiceLoss
from training.metrics import LocalizationMetrics


class LocalizerTrainer:
    """Trainer class for U-Net localizer"""
    
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoints']['save_dir']) / 'localizer'
        self.log_dir = Path(config['logging']['tensorboard_dir']) / 'localizer'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print("Initializing U-Net model...")
        self.model = create_unet_localizer(
            n_channels=3,
            n_classes=1,
            bilinear=True,
            device=self.device
        )
        
        # Initialize dataloaders
        print("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            data_dir=config['dataset']['processed_path'],
            batch_size=config['training']['localizer']['batch_size'],
            num_workers=config['hardware']['num_workers'],
            use_ela=config.get('use_ela', True),
            image_size=config['image']['size'][0]
        )
        
        # Initialize loss function
        # Initialize loss function
        loss_type = config['loss']['localizer']
        if loss_type == 'dice_bce':
            self.criterion = DiceBCELoss(
                dice_weight=config['loss'].get('dice_weight', 0.5),
                bce_weight=config['loss'].get('bce_weight', 0.5)
            )
        elif loss_type == 'focal_dice':
            from training.losses import FocalDiceLoss
            focal_dice_config = config['loss'].get('focal_dice', {})
            self.criterion = FocalDiceLoss(
                alpha=focal_dice_config.get('alpha', 0.75),
                gamma=focal_dice_config.get('gamma', 2.0),
                dice_weight=focal_dice_config.get('dice_weight', 0.6),
                focal_weight=focal_dice_config.get('focal_weight', 0.4)
            )
            print(f"  Using Focal Dice Loss (alpha={focal_dice_config.get('alpha', 0.75)}, "
                  f"gamma={focal_dice_config.get('gamma', 2.0)})")
        elif loss_type == 'focal_tversky':
            from training.losses import FocalTverskyLoss
            ft_config = config['loss'].get('focal_tversky', {})
            self.criterion = FocalTverskyLoss(
                alpha=ft_config.get('alpha', 0.7),
                beta=ft_config.get('beta', 0.3),
                gamma=ft_config.get('gamma', 1.5)
            )
        elif loss_type == 'combo':
            from training.losses import ComboLoss
            self.criterion = ComboLoss(alpha=0.5, ce_weight=0.5)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print(f"  Using loss: {loss_type}")
        
        
        # Initialize optimizer
        optimizer_type = config['training']['localizer']['optimizer']
        lr = config['training']['localizer']['learning_rate']
        weight_decay = config['training']['localizer']['weight_decay']
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Initialize scheduler
        scheduler_type = config['training']['localizer']['scheduler']
        patience = config['training']['localizer']['patience']
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=patience,
                factor=0.5
            )
            print(f"  Using ReduceLROnPlateau scheduler (patience={patience})")
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['localizer']['epochs']
            )
            print(f"  Using CosineAnnealing scheduler")
        else:
            self.scheduler = None
            print(f"  No scheduler")
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_iou = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = patience * 2
        
        print(f"✓ Trainer initialized on {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"  Validation samples: {len(self.dataloaders['val'].dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        metrics = LocalizationMetrics(threshold=0.5)
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            has_mask = batch['has_mask']
            
            # Skip batch if no masks (only train on tampered images with ground truth)
            if not has_mask.any():
                continue
            
            # Filter to only samples with masks
            mask_indices = has_mask.nonzero(as_tuple=True)[0]
            images = images[mask_indices]
            masks = masks[mask_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            # Gradient clipping (NEW!)
            if self.config['training']['localizer'].get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['localizer']['gradient_clip']
                )
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            metrics.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.dataloaders['train']) + batch_idx
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.dataloaders['train'])
        epoch_metrics = metrics.compute()
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        metrics = LocalizationMetrics(threshold=0.5)
        
        pbar = tqdm(self.dataloaders['val'], desc=f"Epoch {self.current_epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                has_mask = batch['has_mask']
                
                # Skip batch if no masks
                if not has_mask.any():
                    continue
                
                # Filter to only samples with masks
                mask_indices = has_mask.nonzero(as_tuple=True)[0]
                images = images[mask_indices]
                masks = masks[mask_indices]
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                running_loss += loss.item()
                metrics.update(outputs, masks)
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.dataloaders['val'])
        epoch_metrics = metrics.compute()
        
        return avg_loss, epoch_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (IoU: {self.best_iou:.4f})")
    
    def train(self, num_epochs):
        """
        Main training loop
        
        Args:
            num_epochs (int): Number of epochs to train
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Metrics/Train_IoU', train_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/Val_IoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/Train_Dice', train_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/Val_Dice', val_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/Val_PixelAcc', val_metrics['pixel_accuracy'], epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
            print(f"  Samples with masks - Train: {train_metrics['num_samples']}, Val: {val_metrics['num_samples']}")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['iou']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config['training']['localizer']['early_stopping']:
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                    print(f"  Best IoU: {self.best_iou:.4f}")
                    break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best Validation IoU: {self.best_iou:.4f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Localizer")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['localizer']['epochs'] = args.epochs
    
    # Create trainer
    trainer = LocalizerTrainer(config)
    
    # Train
    num_epochs = config['training']['localizer']['epochs']
    trainer.train(num_epochs)


if __name__ == "__main__":
    main()
