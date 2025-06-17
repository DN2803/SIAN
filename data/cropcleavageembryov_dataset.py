from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform
import torch.nn.functional as F
class CropcleavageEmbryovDataset(Pix2pixDataset):
    """
    Dataset dùng cho mô hình SIAN với các đầu vào:
    - instance mask (label)
    - semantic map
    - directional map
    - distance map
    - ground truth image path (không load ảnh thật ở đây)
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        return parser

    def initialize(self, opt):
        # Khởi tạo Pix2pixDataset đúng cách với đối số opt
        self.opt = opt

        # Lấy tất cả các đường dẫn đến các dữ liệu
        self.instance_dir, self.semantic_dir, self.direction_dir, self.distance_dir, self.image_dir = self.get_paths(opt)
        self.paths = sorted(make_dataset(self.instance_dir))
        self.image_paths = sorted(make_dataset(self.image_dir))

    def __getitem__(self, index):
        # Đường dẫn đến instance mask (đầu vào chính)
        inst_path = self.paths[index]
        inst_name = os.path.splitext(os.path.basename(inst_path))[0]

        # Load instance mask (ảnh nhị phân)
        instance_img = None
        if self.opt.load_size > 0:
            instance_img = self.resize_map(inst_path, False)
            # if instance_img.shape[0] > 1: 
            #     instance_img = sum(instance_img, dim=0, keepdim=True)  # gộp các kênh nếu có nhiều kênh
        # to_tensor = transforms.ToTensor()
        instance_tensor = instance_img#to_tensor(instance_img)

        # Load các bản đồ phụ trợ (semantic, direction, distance)
        semantic_path = os.path.join(self.semantic_dir, f'{inst_name}.npy')
        direction_path = os.path.join(self.direction_dir, f'{inst_name}.npy')
        distance_path = os.path.join(self.distance_dir, f'{inst_name}.npy')
        image_path = os.path.join(self.image_dir, f'{inst_name}.npy')

        semantic_map = self.resize_map(semantic_path, False)
        distance_map = self.resize_map(distance_path)
        direction_map = self.resize_map(direction_path)
        image = np.load(image_path)
        # ressize ảnh về kích thước chuẩn
        
        if self.opt.load_size > 0:
            image = cv2.resize(image, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))  # chuyển từ (H, W, C) sang (C, H, W)
        image = torch.from_numpy(image).float()  # chuyển sang tensor float
        



        return {
            'label': instance_tensor.long(),
            'instance': instance_tensor,  # bạn có thể sửa nếu cần dùng riêng inst
            'semantic_map': semantic_map,
            'distance_map': distance_map,
            'directional_map': direction_map,
            'image' : image,
            'path': image_path
        }

    def __len__(self):
        return len(self.paths)

    def get_paths(self, opt):
        # Tùy vào phase (train/val/test), xác định thư mục
        phase = 'val' if opt.phase == 'test' else opt.phase
        root = opt.dataroot
        instance_dir = os.path.join(root, f'{phase}_inst')
        semantic_dir = os.path.join(root, f'{phase}_semantic')
        direction_dir = os.path.join(root, f'{phase}_direction')
        distance_dir = os.path.join(root, f'{phase}_distance')
        image_dir = os.path.join(root, f'{phase}_image')
        return instance_dir, semantic_dir, direction_dir, distance_dir, image_dir
    # Load và chuyển về tensor, thêm batch dim (1, C, H, W), resize, rồi bỏ batch dim
    def resize_map(self, np_path, normalize=True):
        # Load numpy array và chuyển sang tensor float32
        map_tensor = torch.from_numpy(np.load(np_path)).float()  # shape: (H, W) hoặc (C, H, W)

        # Đảm bảo tensor có shape (1, C, H, W)
        if map_tensor.dim() == 2:
            map_tensor = map_tensor.unsqueeze(0)  # (1, H, W)
        if map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(0)  # (1, C, H, W)

        # Resize về kích thước chuẩn
        map_tensor = F.interpolate(
            map_tensor,
            size=(self.opt.load_size, self.opt.load_size),
            mode='bilinear',
            align_corners=False
        )

        if normalize:
            # Tính mean & std theo spatial (H, W) cho từng channel
            mean = map_tensor.mean(dim=(2, 3), keepdim=True)  # shape: (1, C, 1, 1)
            std = map_tensor.std(dim=(2, 3), keepdim=True)    # shape: (1, C, 1, 1)
            std[std == 0] = 1e-6  # tránh chia cho 0

            # Z-score normalization theo từng channel
            map_tensor = (map_tensor - mean) / std

        return map_tensor.squeeze(0)  # Trả về shape: (C, H, W)