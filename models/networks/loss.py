
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19

from utils.util import extract_instance_patches

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda().to(torch.bfloat16) 
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    

class InstancePerceptualLoss(nn.Module):
    def __init__(self):
        super(InstancePerceptualLoss, self).__init__()
        self.vgg = VGG19().cuda().to(torch.bfloat16) 
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, real_patches, fake_patches):
        if real_patches is None or fake_patches is None:
            return torch.tensor(0.0).cuda()
        loss = torch.tensor(0.0, device=fake_patches.device)
        x_vgg, y_vgg = self.vgg(fake_patches), self.vgg(real_patches)
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class SIANLoss(nn.Module):
    def __init__(self, opt):
        super(SIANLoss, self).__init__()
        # self.gan_loss = GANLoss(opt.gan_mode, tensor=torch.cuda.FloatTensor, opt=opt)
        self.vgg_loss = VGGLoss(opt.gpu_ids)
        self.kld_loss = KLDLoss()
        self.instance_perceptual_loss = InstancePerceptualLoss()
        self.l1_loss = nn.L1Loss()
        lambda_f = 10.0
        lambda_p = 5.0
        # lambda_kld = 0.05
        lambda_reg = 1.0
        # Hệ số lambda (có thể điều chỉnh từ opt hoặc đặt thủ công)
        self.lambda_f = lambda_f      # perceptual
        self.lambda_p = lambda_p      # instance/patch loss
        # self.lambda_kld = lambda_kld
        self.lambda_reg = lambda_reg  # style consistency

    def forward(self, style_fake, style_real, real_img, fake_img, mask):
        
        # GAN loss (Generator)
        # loss_gan = self.gan_loss(pred_fake, pred_real, for_discriminator=False)

        # Feature matching loss in discriminator
        loss_f = self.vgg_loss(fake_img, real_img)


        real_patches = extract_instance_patches(real_img, mask)
        fake_patches = extract_instance_patches(fake_img, mask)
        # Patch-level or instance-level loss (L1 loss)
        loss_p = self.instance_perceptual_loss(real_patches, fake_patches)

        # Style regularization loss
        loss_reg = self.l1_loss(style_fake, style_real)

        
        # Tổng hợp loss
        total_loss = self.lambda_f * loss_f + \
                     self.lambda_p * loss_p + self.lambda_reg * loss_reg 
                     

        return total_loss, {
            # 'GAN': loss_gan.item(),
            'VGG': self.lambda_f * loss_f.item(),
            'Patch': self.lambda_p * loss_p.item(),
            'StyleReg': self.lambda_reg * loss_reg.item(),
        }
