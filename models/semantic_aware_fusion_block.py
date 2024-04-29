import torch
import torch.nn as nn
from attention import CrossAttention
from attention import SelfAttention

class SemanticAwareFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, 1024) #TODO: check for the number of groups

        self.reduce_channels1 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1)
        self.reduce_channels2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1)

        self.layer_norm_1 = nn.LayerNorm(128)
        self.layer_norm_2 = nn.LayerNorm(128)
        self.layer_norm_3 = nn.LayerNorm(128)

        self.self_attention = SelfAttention(128, num_heads=1, dimensionality=128)
        self.cross_attention = CrossAttention(128, heads=1, dim_head=128)

        self.GeLU = nn.GELU()

        #define 1x1 convolutions
        self.increase_channels1 = nn.Conv2d(128, 1024, 1)

    def forward(self, semantic_feature_maps, fs):

        # sh have shape batch,1024,x,x
        #feature maps (fs or fh) have shape batch x 128 
        final_permute_height = semantic_feature_maps.shape[2]
        final_permute_width = semantic_feature_maps.shape[3]

        print(semantic_feature_maps.shape)
        print(fs.shape)

        #first handle S_h
        fs = self.group_norm(fs)
        print("sh shape after group norm", fs.shape)

        #reduce the channel dimensions for the feature maps
        semantic_feature_maps = self.reduce_channels2(semantic_feature_maps)
        print("feature maps shape after reduce channels", semantic_feature_maps.shape)

        #permute the dimensions


        # Permute dimensions to rearrange the tensor
        semantic_feature_maps = semantic_feature_maps.permute(0, 2, 3, 1).contiguous().view(semantic_feature_maps.size(0), -1, semantic_feature_maps.size(1))

        print("semantic_feature_maps shape after permute", semantic_feature_maps.shape)

        #apply layer normalization
        semantic_feature_maps = self.layer_norm_1(semantic_feature_maps)

        print("semantic_feature_maps shape after layer norm", semantic_feature_maps.shape)
        

        #apply self attention
        semantic_feature_maps = self.self_attention(semantic_feature_maps) #returned has shape 1,196,128 for now
        #apply layer normalization
        query = self.layer_norm_2(semantic_feature_maps)

        #now handle fs or  fh
        #reduce the channel dimensions for the sh
        fs = self.reduce_channels1(fs)
        print("sh shape after reduce channels", fs.shape)


        fs_residual = fs.clone()

        #permute the dimensions
        fs = fs.permute(0, 2, 3, 1).contiguous().view(fs.size(0), -1, fs.size(1))
        print("sh shape after permute", fs.shape)

        #apply cross attention
        out = self.cross_attention(query, fs)
        print("out shape after cross attention", out.shape)

        #apply layer normalization
        out = self.layer_norm_3(out)
        print("out shape after layer norm", out.shape)

        #apply GeLU
        out = self.GeLU(out)
        print("out shape after GeLU", out.shape)

        #permute the dimensions
        print("out shape before permute", out.shape)

        #out = out.view(out.shape[0], out.shape[2], int(out.shape[1] ** 0.5), -1 ) #.permute(0, 3, 1, 2)
        #TODO:check if below or above is correct
        out = out.permute(0,2,1).contiguous().view(out.size(0), -1, final_permute_height, final_permute_width)

        print("out shape after permute", out.shape)

        #add the residual
        output = out + fs_residual
        print("output shape after adding the residual", output.shape)

        #increase the channels
        output = self.increase_channels1(output)
        print("output shape after increasing the channels", output.shape)
    
        return output



model = SemanticAwareFusionBlock()
x = torch.randn(1, 1024, 14, 14)
sh = torch.randn(1, 1024, 14, 14)

with torch.no_grad():
    out = model(x, sh)
    print("out shape:", out.shape)
 

        
[[1,1,1], [2,2,2]] 
        
