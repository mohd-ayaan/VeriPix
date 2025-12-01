"""
U-Net architecture for pixel-level forgery localization
Generates binary segmentation masks highlighting tampered regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle different spatial sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    
    Architecture:
    - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
    - Bottleneck: 512 -> 1024
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1 channel (binary mask)
    """
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        """
        Args:
            n_channels (int): Number of input channels (3 for RGB)
            n_classes (int): Number of output channels (1 for binary mask)
            bilinear (bool): Use bilinear upsampling (True) or transposed conv (False)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            
        Returns:
            torch.Tensor: Logits for segmentation mask (B, 1, H, W)
        """
        # Encoder with skip connections
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # Output
        logits = self.outc(x) # (B, 1, H, W)
        
        return logits


def create_unet_localizer(
    n_channels=3,
    n_classes=1,
    bilinear=True,
    device='cuda'
):
    """
    Factory function to create U-Net localizer
    
    Args:
        n_channels (int): Input channels
        n_classes (int): Output channels
        bilinear (bool): Use bilinear upsampling
        device (str): Device to load model on
        
    Returns:
        UNet: Model instance
    """
    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    print("U-Net Localizer Test")
    print("=" * 60)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_unet_localizer(
        n_channels=3,
        n_classes=1,
        bilinear=True,
        device=device
    )
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    image_size = 224
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
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
    
    # Memory usage
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\nGPU Memory usage: {memory_allocated:.2f} MB")
    
    print("\n✓ All localizer tests passed!")
