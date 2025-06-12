import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SIANResBlk, UpsampleBlock
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
        # parser.add_argument('--style_dim', type=int, default=256, help="Dim of vector style")  
        parser.add_argument('--directional_nc', type=int, default=1, help="Num channel of directional")
        parser.add_argument('--distance_nc', type=int, default=1, help="Num channel of distance")  
        parser.add_argument('--semantic_nc', type=int, default=1, help="Semantic label channels (1 label: 'cell')")
        parser.add_argument('--input_nc', type=int, default=3, help="Number of input channels (e.g., 3 for RGB images)")
        return parser

    def __init__(self, opt):
        super().__init__()
        
        self.num_blocks = opt.num_blocks
        
        # Thiết lập dải channel (có thể điều chỉnh) 
        # Cấu hình kênh tương ứng với từng SIANResBlk (giảm dần)
        nf = opt.ngf
        channels = [nf * 16, nf * 16, nf * 8, nf * 4, nf * 2, nf, nf // 2]
        
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # channels = [512, 512, 256, 256, 128, 128, 64]
        
        # Khởi tạo convolution đầu vào, giả sử đầu vào có semantic_nc channel
        # if (opt.use_vae):
        #     self.fc = nn.Linear(opt.z_dim,  channels[0] * self.sw *self.sh)
        # else:
        #     self.fc = nn.Conv2d(opt.input_nc, channels[0], kernel_size=3, padding=1)
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(opt.input_nc, 16 * nf, 3, padding=1)

        self.encoder = ConvEncoder(opt)

        self.mask_generator = MaskProcessorModel()
        
        # Tạo dãy SIANResBlk
        self.sian_blocks = nn.ModuleList()
        self.upSamplingBlks = nn.ModuleList()
        for in_c in channels:
           
            # print(f"Adding SIANResBlk with in_channels={in_c}, out_channels={out_c}")
            self.sian_blocks.append(
                SIANResBlk(
                    in_channels=in_c,
                    out_channels=in_c,
                    semantic_nc=opt.semantic_nc,           # = 1
                    style_dim=opt.z_dim,               # = 256
                    directional_nc=opt.directional_nc,     # = 1
                    distance_nc=opt.distance_nc          # = 1
                )
            )
            self.upSamplingBlks.append(
                UpsampleBlock(in_c, 2)
            )
        
        

        
        

        # Conv cuối để ra ảnh RGB 3 channel
        self.final_conv = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
    
    def forward(self, input, semantic_map, directional_map, distance_map, real_image=None, z=None):
        seg = input 
        # Nếu z (style latent) không được truyền vào, tự sinh từ real_image
        if z is None and real_image is not None:
            mu, logvar = self.encoder(real_image)
            z = self.reparameterize(mu, logvar)
        # input
        sh, sw = self.sh, self.sw
        seg = F.interpolate(seg, size=(self.sh, self.sw))
        m = semantic_map
        p = directional_map
        q = distance_map
        out = self.fc(seg)
        # print(f"Initial conv output shape: {out.shape}")
        for block, up_block in zip(self.sian_blocks, self.upSamplingBlks):
            m = F.interpolate(semantic_map,  size=(sh, sw), mode='bilinear', align_corners=False)
            p = F.interpolate(directional_map, size=(sh, sw), mode='bilinear', align_corners=False)
            q = F.interpolate(distance_map, size=(sh, sw), mode='bilinear', align_corners=False)
            out = block(out, m, z, p, q)
            out = up_block(out)
            sh = sh * 2
            sw = sw * 2
            print(out.shape)
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
        return mu, logvar, self.reparameterize(mu, logvar)
    def compute_latent_vector_size(self, opt):
        num_up_layer = opt.num_blocks

        sw = opt.crop_size // (2**num_up_layer)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh
