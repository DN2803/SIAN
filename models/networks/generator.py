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
        self.opt = opt
        self.num_blocks = opt.num_blocks
        
        # Thiết lập dải channel (có thể điều chỉnh) 
        # Cấu hình kênh tương ứng với từng SIANResBlk (giảm dần)
        nf = opt.ngf
        channels = [nf * 16, nf * 8, nf * 4, nf * 2, nf, nf // 2, nf // 4]
        
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # channels = [512, 512, 256, 256, 128, 128, 64]
        self.fc0 = nn.Conv2d(opt.input_nc, channels[0], kernel_size=3, padding=1)
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.input_nc, 16 * nf, 3, padding=1)
        
        # Tạo dãy SIANResBlk
        self.fcs = nn.ModuleList()
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
        self.final_conv = nn.Conv2d(channels[-1] // 2, 3, kernel_size=1, padding=0)
    
    def forward(self, input, semantic_map, directional_map, distance_map, z=None):
        seg = input 

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            # x = self.fc(z)
            # x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        # else:
            # we downsample segmap and run convolution
            # x = F.interpolate(seg, size=(self.sh, self.sw))
            # x = self.fc(x)
        x = z

        # input
        sh, sw = self.sh, self.sw
        seg = F.interpolate(seg, size=(self.sh, self.sw))  # 2 x 2 x 3
        m = semantic_map
        p = directional_map
        q = distance_map
        out = self.fc0(seg)  # 2 x 2 x 1024
        # print(f"Initial conv output shape: {out.shape}")
        for block, up_block in zip(self.sian_blocks, self.upSamplingBlks):
            m = F.interpolate(semantic_map,  size=(sh, sw), mode='bilinear', align_corners=False)
            p = F.interpolate(directional_map, size=(sh, sw), mode='bilinear', align_corners=False)
            q = F.interpolate(distance_map, size=(sh, sw), mode='bilinear', align_corners=False)
            out = block(out, m, x, p, q)
            out = up_block(out)
            sh = sh * 2
            sw = sw * 2
            # x = x.view(-1, out.shape[1], sh, sw)
            # print(out.shape)
        out = self.final_conv(out)
        return torch.tanh(out)
    def compute_latent_vector_size(self, opt):
        num_up_layer = opt.num_blocks

        sw = opt.crop_size // (2**num_up_layer)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh
