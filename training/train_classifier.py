"""
Training script for EfficientNet-B0 classifier
Trains binary classifier to detect authentic vs tampered images
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

from models.efficientnet_classifier import create_efficientnet_classifier
from utils.dataset import create_dataloaders
from training.losses import FocalLoss, WeightedBCELoss
from training.metrics import ClassificationMetrics


class ClassifierTrainer:
    """Trainer class for EfficientNet-B0 classifier"""
    
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoints']['save_dir']) / 'classifier'
        self.log_dir = Path(config['logging']['tensorboard_dir']) / 'classifier'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print("Initializing model...")
        self.model = create_efficientnet_classifier(
            pretrained=config['classifier']['pretrained'],
            dropout=config['classifier']['dropout'],
            device=self.device
        )
        
        # Initialize dataloaders
        print("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            data_dir=config['dataset']['processed_path'],
            batch_size=config['training']['classifier']['batch_size'],
            num_workers=config['hardware']['num_workers'],
            use_ela=config.get('use_ela', True),
            image_size=config['image']['size'][0]
        )
        
        # Initialize loss function
        loss_type = config['loss']['classifier']
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            from training.losses import FocalLoss
            self.criterion = FocalLoss(
                alpha=config['loss'].get('focal_alpha', 0.25),
                gamma=config['loss'].get('focal_gamma', 2.0)
            )
        elif loss_type == 'label_smoothing':
            from training.losses import LabelSmoothingBCELoss
            self.criterion = LabelSmoothingBCELoss(
                smoothing=config['loss'].get('label_smoothing', 0.1)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"  Using loss: {loss_type}")

        
        # Initialize optimizer
        optimizer_type = config['training']['classifier']['optimizer']
        lr = config['training']['classifier']['learning_rate']
        weight_decay = config['training']['classifier']['weight_decay']
        
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
        
        # Initialize scheduler (FIXED - removed verbose parameter)
        scheduler_type = config['training']['classifier']['scheduler']
        patience = config['training']['classifier']['patience']
        
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
                T_max=config['training']['classifier']['epochs']
            )
            print(f"  Using CosineAnnealing scheduler")
        else:
            self.scheduler = None
            print(f"  No scheduler")
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = patience * 2  # 2x scheduler patience
        
        print(f"✓ Trainer initialized on {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"  Validation samples: {len(self.dataloaders['val'].dataset)}")

    
    def _calculate_pos_weight(self):
        """Calculate positive class weight for imbalanced dataset"""
        dataset = self.dataloaders['train'].dataset
        num_positive = sum(dataset.labels)
        num_negative = len(dataset.labels) - num_positive
        pos_weight = num_negative / (num_positive + 1e-6)
        print(f"  Calculated pos_weight: {pos_weight:.2f}")
        return pos_weight
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        metrics = ClassificationMetrics(threshold=0.5)
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, labels.float())

            
            # Backward pass
            loss.backward()
            # Gradient clipping (NEW!)
            if self.config['training']['classifier'].get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['classifier']['gradient_clip']
                )
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            metrics.update(outputs, labels)
            
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
        metrics = ClassificationMetrics(threshold=0.5)
        
        pbar = tqdm(self.dataloaders['val'], desc=f"Epoch {self.current_epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, labels.float())

                
                # Update metrics
                running_loss += loss.item()
                metrics.update(outputs, labels)
                
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
            'best_f1': self.best_f1,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (F1: {self.best_f1:.4f})")
    
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
            self.writer.add_scalar('Metrics/Train_Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Metrics/Val_Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Metrics/Train_F1', train_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/Val_F1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/Val_AUC', val_metrics['auc_roc'], epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['f1']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config['training']['classifier']['early_stopping']:
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                    print(f"  Best F1: {self.best_f1:.4f}")
                    break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best Validation F1: {self.best_f1:.4f}")
        
        self.writer.close()
        
    def _get_transform(self, split='train'):
        """Get appropriate transform based on split"""
        if split == 'train' and self.config['training']['classifier'].get('use_advanced_augmentation', False):
            from utils.advanced_augmentations import get_advanced_train_transforms
            return get_advanced_train_transforms(self.config['image']['size'][0])
        else:
            from utils.advanced_augmentations import get_advanced_val_transforms
            return get_advanced_val_transforms(self.config['image']['size'][0])



def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Classifier")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['classifier']['epochs'] = args.epochs
    
    # Create trainer
    trainer = ClassifierTrainer(config)
    
    # Train
    num_epochs = config['training']['classifier']['epochs']
    trainer.train(num_epochs)


if __name__ == "__main__":
    main()
