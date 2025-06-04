import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SIANResBlk

class SIANGenerator(BaseNetwork):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    #     parser.add_argument('--num_upsampling_layers',
    #                         choices=('normal', 'more', 'most'), default='normal',
    #                         help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
    #     return parser

    def __init__(self, semantic_nc, style_dim, directional_nc, distance_nc, base_channels=64, num_blocks=7):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        # Thiết lập dải channel (có thể điều chỉnh)
        channels = [base_channels * (2 ** i) for i in range(num_blocks)][::-1]
        
        # Khởi tạo convolution đầu vào, giả sử đầu vào có semantic_nc channel
        self.initial_conv = nn.Conv2d(semantic_nc, channels[0], kernel_size=3, padding=1)
        
        # Tạo dãy SIANResBlk
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_c = channels[i] if i == 0 else channels[i-1]
            out_c = channels[i]
            self.blocks.append(
                SIANResBlk(
                    in_channels=in_c,
                    out_channels=out_c,
                    semantic_nc=semantic_nc,
                    style_dim=style_dim,
                    directional_nc=directional_nc,
                    distance_nc=distance_nc,
                    upsample=True
                )
            )
        
        # Conv cuối để ra ảnh RGB 3 channel
        self.final_conv = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
    
    def forward(self, x, semantic_map, style_vector, directional_map, distance_map):
        out = self.initial_conv(x)
        for block in self.blocks:
            out = block(out, semantic_map, style_vector, directional_map, distance_map)
        out = self.final_conv(out)
        out = torch.tanh(out)
        return out
