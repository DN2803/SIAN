import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SIANNorm(nn.Module):
    def __init__(self, in_channels, out_channels, semantic_nc, style_dim, directional_nc, distance_nc):
        super(SIANNorm, self).__init__()
        self.conv_c = 128  # Cố định số kênh output trung gian

        # Semantization
        self.conv1 = nn.Conv2d(semantic_nc, self.conv_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(semantic_nc, self.conv_c, kernel_size=3, padding=1)

        # Stylization
        self.style_proj = nn.Linear(style_dim, self.conv_c)
        self.conv3 = nn.Conv2d(self.conv_c, self.conv_c, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.conv_c, self.conv_c, kernel_size=3, padding=1)

        # Instantiation
        self.layout_proj1 = nn.Conv2d(directional_nc, self.conv_c, kernel_size=1)
        self.layout_proj2 = nn.Conv2d(distance_nc, self.conv_c, kernel_size=1)
        self.conv5 = nn.Conv2d(self.conv_c, self.conv_c, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.conv_c, self.conv_c, kernel_size=3, padding=1)

        # Modulation
        self.conv7 = nn.Conv2d(self.conv_c, out_channels, kernel_size=3, padding=1)  # gamma_i
        self.conv8 = nn.Conv2d(self.conv_c, out_channels, kernel_size=3, padding=1)  # gamma_j
        self.conv9 = nn.Conv2d(self.conv_c, out_channels, kernel_size=3, padding=1)  # beta_i
        self.conv10 = nn.Conv2d(self.conv_c, out_channels, kernel_size=3, padding=1) # beta_j

        # Batch norm with fixed out_channels = 128
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=False)

        # Project input if needed
        # self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, input, semantic_map, style_vector, directional_map, distance_map):
        # Semantization
        p_feature = self.conv1(semantic_map)
        q_feature = self.conv2(semantic_map)

        # Stylization
        B = style_vector.size(0) 
        style_matrix = self.style_proj(style_vector).view(B, self.conv_c, 1, 1)
        p_feature = self.conv3(style_matrix * p_feature)
        q_feature = self.conv4(style_matrix * q_feature)

        # Instantiation
        p_dir = self.layout_proj1(directional_map)
        p_feature = self.conv5(p_feature * p_dir)

        q_dis = self.layout_proj2(distance_map)
        q_feature = self.conv6(q_feature * q_dis)

        # Modulation
        gamma_i = self.conv7(p_feature)
        gamma_j = self.conv8(q_feature)
        beta_i = self.conv9(p_feature)
        beta_j = self.conv10(q_feature)

        gamma = gamma_i + gamma_j
        beta = beta_i + beta_j
        
        # Normalize and modulate
        # x = self.input_proj(input)  # Project input to ith layer channels if needed
        x_norm = self.instance_norm(input)
        print("x_norm shape:", x_norm.shape, x_norm.mean(), x_norm.min(), x_norm.max())
        print("gamma shape:", gamma.shape, gamma.mean(), gamma.min(), gamma.max())
        print("beta shape:", beta.shape, beta.mean(), beta.min(), beta.max())
        out = gamma * x_norm + beta
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class SIANResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 semantic_nc, style_dim, 
                 directional_nc, distance_nc):
        super().__init__()
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 2 SIAN blocks
        self.sian1 = SIANNorm(in_channels, in_channels,  semantic_nc, style_dim, directional_nc, distance_nc)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.sian2 = SIANNorm(out_channels, out_channels, semantic_nc, style_dim, directional_nc, distance_nc)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.sian_skip = SIANNorm(in_channels, out_channels, semantic_nc, style_dim, directional_nc, distance_nc)
        self.conv_skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)

        

    
    def forward(self, x, semantic_map, style_vector, directional_map, distance_map):
       
        # Residual path 
        # print(x.shape, semantic_map.shape, style_vector.shape, directional_map.shape, distance_map.shape)
        out = self.sian1(x, semantic_map, style_vector, directional_map, distance_map)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.sian2(out, semantic_map, style_vector, directional_map, distance_map)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Skip connection path
        skip = self.sian_skip(x, semantic_map, style_vector, directional_map, distance_map)
        skip = self.relu(skip)
        skip = self.conv_skip(skip)
        out =  out + skip 

        # print(f"SIANResBlk: in_channels={self.in_channels}, out_channels={self.out_channels}, out_shape={out.shape}")
        return out
    

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out