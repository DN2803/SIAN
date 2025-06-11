"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import gc
import models.networks as networks
import utils.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionSIAN = networks.SIANLoss(opt)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map)
            del input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map
            gc.collect()
            torch.cuda.empty_cache()
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            del input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map
            gc.collect()
            torch.cuda.empty_cache()
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            del input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map
            gc.collect()
            torch.cuda.empty_cache()
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            del input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map
            gc.collect()
            torch.cuda.empty_cache()
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # Chuyển label sang long để tạo one-hot
        data['label'] = data['label'].long()

        # Chuyển dữ liệu sang GPU nếu có
        if self.use_gpu():
            for key in ['label', 'instance', 'image', 'semantic_map', 'directional_map', 'distance_map']:
                if key in data:
                    data[key] = data[key].cuda()

        #TODO: Fix this to fit model  
        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # Instance edge
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)  # (B, nc+1, H, W)

        # Xử lý các bản đồ phụ trợ (resize hoặc unsqueeze nếu cần)
        def ensure_shape(x, nc):
            if x.dim() == 3:
                x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            return x[:, :nc, :, :]  # Đảm bảo đúng số kênh

        semantic_map = ensure_shape(data['semantic_map'].float(), self.opt.semantic_nc)
        directional_map = ensure_shape(data['directional_map'].float(), self.opt.directional_nc)
        distance_map = ensure_shape(data['distance_map'].float(), self.opt.distance_nc)

        # Ghép toàn bộ làm input đầu vào
        input_all = torch.cat([input_semantics, semantic_map, directional_map, distance_map], dim=1)

        return input_all, data['image'],  semantic_map, directional_map, distance_map, inst_map


    def  compute_generator_loss(self, input_semantics, real_image, semantic_map, directional_map, distance_map, inst_map):

        G_losses = {}
        # Tạo ảnh giả và mã hóa z nếu cần
        fake_image, mu, logvar = None, None, None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            fake_image = self.netG(input_semantics, semantic_map, directional_map, distance_map, real_image, z=z)
        else:
            fake_image = self.netG(input_semantics, semantic_map, directional_map, distance_map, real_image)

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        # fake_image, KLD_loss = self.generate_fake(
        #     input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        # if self.opt.use_vae:
        #     G_losses['KLD'] = KLD_loss

        # Style vector (giả sử được trích xuất từ G)
        style_real = self.netG.extract_style(real_image)  # hoặc một hàm style encoder riêng
        style_fake = self.netG.extract_style(fake_image)

        # Mask instance map (để tính patch loss)
        mask = self.get_instance_mask()  # bạn cần định nghĩa hàm này hoặc lấy từ data['instance']

        # compute SIAN loss
        total_loss, loss_dict = self.sian_loss(
            pred_fake, pred_real, real_image, fake_image,
            mu, logvar, style_real, style_fake, mask
        )

        G_losses.update(loss_dict)
        G_losses['Total'] = total_loss

        return G_losses, fake_image


    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            # fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z, mu, logvar = None, None, None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
        fake_image = self.netG(input_semantics, z=z)
        return fake_image, mu, logvar
    
    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
    