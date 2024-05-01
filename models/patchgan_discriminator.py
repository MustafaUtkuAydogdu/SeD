import torch
import torch.nn as nn
from semantic_aware_fusion_block import SemanticAwareFusionBlock

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_downscales=3):
        super().__init__()
        
        self.layers = []
        
        # First layer
        self.layers.append(PatchDiscriminatorBlock(input_channels, num_filters))
        
        # Intermediate self.layers
        for i in range(num_downscales):
            in_channels = num_filters * 2 ** i
            out_channels = num_filters * 2 ** (i + 1)
            stride = 1 if i == (num_downscales - 1) else 2
            self.layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
        
        # Last layer
        in_channels = num_filters * 2 ** num_downscales
        out_channels = 1
        self.layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
        
    
    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        return output
    
class PatchDiscriminatorWithSeD(nn.Module):
    def __init__(self, input_channels, num_filters=64, num_downscales=3):
        super().__init__()
        
        self.layers = []
        
        # First layer
        self.layers.append(PatchDiscriminatorBlock(input_channels, num_filters))
        
        # Intermediate self.layers
        for i in range(num_downscales):
            in_channels = num_filters * 2 ** i
            out_channels = num_filters * 2 ** (i + 1)
            stride = 1 if i == (num_downscales - 1) else 2
            self.layers.append(PatchDiscriminatorBlock(in_channels, out_channels, use_semfb=True,channel_size_changer_input_nc=in_channels))
        
        # Last layer
        in_channels = num_filters * 2 ** num_downscales
        out_channels = 1
        self.layers.append(PatchDiscriminatorBlock(in_channels, out_channels))
    
    def forward(self, semantic_feature_map, fs):

        for layer in self.layers:
            fs = layer(semantic_feature_map, fs)

        return fs
    
    
class PatchDiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, num_filters, use_semfb=False,channel_size_changer_input_nc=None):
        super().__init__()
        
        self.use_semfb = use_semfb
        if self.use_semfb:
            self.semantic_aware_fusion_block = SemanticAwareFusionBlock(channel_size_changer_input_nc=channel_size_changer_input_nc)
            
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if self.use_semfb:
            self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_filters)
            
    def forward(self, x, fs=None):
        
        print("Input shape before block:", x.shape)
        if self.use_semfb:
            print("Entered sembf")
            x = self.semantic_aware_fusion_block(x, fs)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
        else:
            print("Entered else, x.shape : ", fs.shape)
            x = self.conv1(fs)
            x = self.bn1(x)
            x = self.relu(x)
        print("Input shape after block:", x.shape)
        return x



if __name__ == "__main__":
    b = torch.randn(1, 128, 32, 32)
    a = torch.randn(1,1024,16,16)
    model = PatchDiscriminatorWithSeD(3)
    with torch.no_grad():
        output = model(a,b)
        print(output.shape)