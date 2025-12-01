"""
EfficientNet-B0 based classifier for image forgery detection
Binary classification: Authentic (0) vs Tampered (1)
"""

import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 classifier with custom head
    
    Architecture:
    - EfficientNet-B0 backbone (pre-trained on ImageNet)
    - Global Average Pooling
    - Dropout
    - Fully Connected layers
    - Sigmoid output for binary classification
    """
    
    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=False
    ):
        """
        Args:
            num_classes (int): Number of output classes (1 for binary)
            pretrained (bool): Use ImageNet pre-trained weights
            dropout (float): Dropout probability
            freeze_backbone (bool): Freeze backbone during training
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 1280 for EfficientNet-B0
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification head weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            
        Returns:
            torch.Tensor: Logits (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (B, 1280)
        
        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits
    
    def get_features(self, x):
        """
        Extract feature embeddings (useful for visualization)
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            
        Returns:
            torch.Tensor: Feature embeddings (B, 1280)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False


def create_efficientnet_classifier(
    pretrained=True,
    dropout=0.3,
    device='cuda'
):
    """
    Factory function to create classifier
    
    Args:
        pretrained (bool): Use pre-trained weights
        dropout (float): Dropout probability
        device (str): Device to load model on
        
    Returns:
        EfficientNetClassifier: Model instance
    """
    model = EfficientNetClassifier(
        num_classes=1,
        pretrained=pretrained,
        dropout=dropout
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    print("EfficientNet-B0 Classifier Test")
    print("=" * 60)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_efficientnet_classifier(
        pretrained=True,
        dropout=0.3,
        device=device
    )
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Feature dimension: {model.feature_dim}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output probabilities shape: {probs.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"\nFeature extraction test:")
    print(f"  Features shape: {features.shape}")
    
    # Memory usage
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\nGPU Memory usage: {memory_allocated:.2f} MB")
    
    print("\n✓ All classifier tests passed!")
