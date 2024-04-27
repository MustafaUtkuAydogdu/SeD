import torch
import torch.nn as nn
from attention import CrossAttention
from attention import SelfAttention

class SemanticAwareFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, -1) #TODO: check for the number of groups
        self.layer_norm_1 = nn.LayerNorm(-1)
        self.layer_norm_2 = nn.LayerNorm(-1)
        self.layer_norm_3 = nn.LayerNorm(-1)

        self.self_attention = SelfAttention(-1, num_heads=-1, dimensionality=-1)
        self.cross_attention = CrossAttention(-1, num_heads=-1, dimensionality=-1)

        self.GeLU = nn.GELU()

        #define 1x1 convolutions
        self.conv_down = nn.Conv2d(-1, -1, 1)

        #define upsampling 1x1 convolutions
        self.conv_up = nn.Conv2d(-1, -1, 1)

    def forward(self, feature_maps, sh):
        #first handle S_h
        sh = self.group_norm(sh)
        #permute the dimensions

        # Permute dimensions to rearrange the tensor
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, x.size(1))

        #apply layer normalization
        sh = self.layer_norm_1(sh)
        #apply self attention
        sh = self.self_attention(sh)
        #apply layer normalization
        query = self.layer_norm_2(sh)

        #now handle fs or  fh
        context = self.conv_down(feature_maps)

        context_residual = context #.clone()

        #permute the dimensions
        context = context.permute(0, 2, 3, 1).contiguous().view(context.size(0), -1, context.size(1))

        #apply cross attention
        context = self.cross_attention(query, context)

        #apply layer normalization
        context = self.layer_norm_3(context)

        #apply GeLU
        context = self.GeLU(context)

        #permute the dimensions
        context = context.view(context.size(0), context.size(1), 1, 1).permute(0, 3, 1, 2)

        #concat with the original feature maps
        output = torch.cat([context, context_residual], dim=1)

        #apply 1x1 convolution
        output = self.conv_up(output)

        return output



        


        

        
