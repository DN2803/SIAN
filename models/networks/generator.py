import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SIANResBlk
from models.networks.mask_generator import MaskProcessorModel
from models.networks.encoder import ConvEncoder

class SIANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        parser.add_argument('--num_blocks', type=int, default=7, help="Num of SIANResBlk blocks") 
        parser.add_argument('--base_channels', type=int, default=128, help="The starting number of feature channels")    
        parser.add_argument('--style_dim', type=int, default=128, help="Dim of vector style")    
        return parser

    def __init__(self, opt):
        super().__init__()
        
        self.num_blocks = opt.num_blocks
        
        # Thiết lập dải channel (có thể điều chỉnh)
        channels = [opt.base_channels * (2 ** i) for i in range(opt.num_blocks)][::-1]
        
        # Khởi tạo convolution đầu vào, giả sử đầu vào có semantic_nc channel
        self.initial_conv = nn.Conv2d(opt.semantic_nc, channels[0], kernel_size=3, padding=1)

        self.encoder = ConvEncoder(opt)

        self.mask_generator = MaskProcessorModel()
        
        # Tạo dãy SIANResBlk
        self.blocks = nn.ModuleList()
        for i in range(opt.num_blocks):
            in_c = channels[i] if i == 0 else channels[i-1]
            out_c = channels[i]
            self.blocks.append(
                SIANResBlk(
                    in_channels=in_c,
                    out_channels=out_c,
                    semantic_nc=opt.semantic_nc,
                    style_dim=opt.style_dim,
                    directional_nc=opt.directional_nc,
                    distance_nc=opt.distance_nc,
                    upsample=True
                )
            )
    

        # Conv cuối để ra ảnh RGB 3 channel
        self.final_conv = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
    
    def forward(self, input, semantic_map, directional_map, distance_map, real_image=None, z=None):
        # Nếu z (style latent) không được truyền vào, tự sinh từ real_image
        if z is None and real_image is not None:
            z = self.encoder(real_image)

        out = self.initial_conv(input)
        for block in self.blocks:
            out = block(out, semantic_map, z, directional_map, distance_map)

        out = self.final_conv(out)
        return torch.tanh(out)
    def extract_style(self, real_image):
        return self.encoder(real_image)
