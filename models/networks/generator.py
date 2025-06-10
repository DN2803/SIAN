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
        parser.add_argument('--style_dim', type=int, default=256, help="Dim of vector style")  
        parser.add_argument('--directional_nc', type=int, default=1, help="Num channel of directional")
        parser.add_argument('--distance_nc', type=int, default=1, help="Num channel of distance")  
        parser.add_argument('--semantic_nc', type=int, default=1, help="Semantic label channels (1 label: 'cell')")
        parser.add_argument('--input_nc', type=int, default=6, help="Number of input channels (e.g., 3 for RGB images)")
        return parser

    def __init__(self, opt):
        super().__init__()
        
        self.num_blocks = opt.num_blocks
        
        # Thiết lập dải channel (có thể điều chỉnh)
        # Cấu hình kênh tương ứng với từng SIANResBlk (giảm dần)
        base_channel = opt.base_channel if hasattr(opt, 'base_channel') else 64
        channels = [base_channel * 8, base_channel * 8, base_channel * 4, base_channel * 4, base_channel * 2, base_channel * 2, base_channel]
        channel_pairs = list(zip(channels, channels[1:] + [channels[-1]]))  # đảm bảo có 7 cặp in-out

        # channels = [512, 512, 256, 256, 128, 128, 64]
        
        # Khởi tạo convolution đầu vào, giả sử đầu vào có semantic_nc channel
        self.initial_conv = nn.Conv2d(opt.input_nc, channels[0], kernel_size=3, padding=1)

        self.encoder = ConvEncoder(opt)

        self.mask_generator = MaskProcessorModel()
        
        # Tạo dãy SIANResBlk
        self.blocks = nn.ModuleList()
        for in_c, out_c in channel_pairs:
            print(f"Adding SIANResBlk with in_channels={in_c}, out_channels={out_c}")
            self.blocks.append(
                SIANResBlk(
                    in_channels=in_c,
                    out_channels=out_c,
                    semantic_nc=opt.semantic_nc,           # = 1
                    style_dim=opt.style_dim,               # = 256
                    directional_nc=opt.directional_nc,     # = 1
                    distance_nc=opt.distance_nc,           # = 1
                    upsample=True
                )
            )
            

        # Conv cuối để ra ảnh RGB 3 channel
        self.final_conv = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
    
    def forward(self, input, semantic_map, directional_map, distance_map, real_image=None, z=None):
        # Nếu z (style latent) không được truyền vào, tự sinh từ real_image
        if z is None and real_image is not None:
            mu, logvar = self.encoder(real_image)
            z = self.reparameterize(mu, logvar)

        out = self.initial_conv(input)
        for block in self.blocks:
            out = block(out, semantic_map, z, directional_map, distance_map)

        out = self.final_conv(out)
        return torch.tanh(out)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def extract_style(self, real_image, use_mu_only=True):
        mu, logvar = self.encoder(real_image)
        if use_mu_only:
            return mu
        return self.reparameterize(mu, logvar)
    
