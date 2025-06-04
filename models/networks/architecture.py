


import torch.nn as nn 
class SIANNorm (nn.Module):
    def __init__(self, in_c):
        super(SIANNorm, self).__init__()
        
        #layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3)

        #layer 2
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)


        # layer 3
        self.layout_proj1 = nn.Conv2d(3, 128, kernel_size=1)
        self.layout_proj2 = nn.Conv2d(1, 128, kernel_size=1)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # layer 4
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.batch_norm = nn.InstanceNorm2d(128, affine=False)

    def forward(self, input, semantic_map, style_vector, directional_map, distance_map):
        # semanitzation
        p_feature = self.conv1(semantic_map)
        q_feature = self.conv2(semantic_map)
        
        #Stylzation

        style_matrix = style_vector.view(P_feature.size)
        p_feature= self.conv3(style_matrix * p_feature)

        q_feature = self.conv4(style_matrix * p_feature)

        #Instantiation

        p = self.layout_proj1(directional_map)
        P_feature = self.conv5(P_feature * p)
        
        q = self.layout_proj2(distance_map)
        q_feature = self.conv6(q_feature * q)

        # Modulation

        gamma_i = self.conv7(p_feature)
        beta_i = self.conv9(p_feature)

        gamma_j = self.conv8(q_feature)
        beta_j = self.conv10(q_feature)


        gamma = gamma_i + gamma_j
        beta = beta_i + beta_j

        x_norm = self.batch_norm(input)

        out = gamma * x_norm + beta
        return out



class SIANResBik(nn.Module):
    pass 