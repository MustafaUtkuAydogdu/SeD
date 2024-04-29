import torch
import torch.nn as nn
from semantic_aware_fusion_block import SemanticAwareFusionBlock

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_downscales=3):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(PatchDiscriminatorBlock(input_channels, num_filters))
        
        # Intermediate layers
        for i in range(num_downscales):
            in_channels = num_filters * 2 ** i
            out_channels = num_filters * 2 ** (i + 1)
            stride = 1 if i == (num_downscales - 1) else 2
            layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
        
        # Last layer
        in_channels = num_filters * 2 ** num_downscales
        out_channels = 1
        layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class PatchDiscriminatorWithSeD(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_downscales=3):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(PatchDiscriminatorBlock(input_channels, num_filters))
        
        # Intermediate layers
        for i in range(num_downscales):
            in_channels = num_filters * 2 ** i
            out_channels = num_filters * 2 ** (i + 1)
            stride = 1 if i == (num_downscales - 1) else 2
            layers.append(PatchDiscriminatorBlock(in_channels, out_channels, use_semfb=True))
        
        # Last layer
        in_channels = num_filters * 2 ** num_downscales
        out_channels = 1
        layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, fs):
        return self.model(x)
    
    
class PatchDiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, num_filters, use_semfb=False):
        super().__init__()
        
        self.use_semfb = use_semfb
        if self.use_semfb:
            self.semantic_aware_fusion_block = SemanticAwareFusionBlock()
            
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, inplace=True)
        ) if not self.use_semfb else nn.Sequential(
            self.semantic_aware_fusion_block,
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x, fs=None):
        print("Input shape before block:", x.shape)
        x = self.layer(x) if not self.use_semfb else self.layer(x, fs) #TODO: check correct ordering of x and fs
        print("Input shape after block:", x.shape)
        return x


if __name__ == "__main__":
    a = torch.randn(1, 3, 480, 480)
    b = torch.randn(1,1024,14,14)
    model = PatchDiscriminatorWithSeD(3)
    with torch.no_grad():
        output = model(tensor)
        print(output.shape)