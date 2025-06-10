from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform
import torch.nn.functional as F
class CleavageEmbryovDataset(Pix2pixDataset):
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
        instance_img = Image.open(inst_path).convert('L')
        if self.opt.load_size > 0:
            instance_img = instance_img.resize((self.opt.load_size, self.opt.load_size))
        to_tensor = transforms.ToTensor()
        instance_tensor = to_tensor(instance_img)

        # Load các bản đồ phụ trợ (semantic, direction, distance)
        semantic_path = os.path.join(self.semantic_dir, f'{inst_name}.npy')
        direction_path = os.path.join(self.direction_dir, f'{inst_name}.npy')
        distance_path = os.path.join(self.distance_dir, f'{inst_name}.npy')
        image_path = os.path.join(self.image_dir, f'{inst_name}.png')

        semantic_map = self.resize_map(semantic_path)
        distance_map = self.resize_map(distance_path)
        direction_map = self.resize_map(direction_path)
         # input image (real images)
        image_path = self.image_paths[index]
        params = get_params(self.opt, (self.opt.load_size,self.opt.load_size) )
        
        image = Image.open(image_path)
        image = image.convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        print("instance_tensor dtype:", instance_tensor.dtype)
        print("instance_tensor min/max:", instance_tensor.min(), instance_tensor.max())
        print("instance_tensor shape:", instance_tensor.shape)


        return {
            'label': instance_tensor,
            'instance': instance_tensor,  # bạn có thể sửa nếu cần dùng riêng inst
            'semantic_map': semantic_map,
            'distance_map': distance_map,
            'directional_map': direction_map,
            'image' : image_tensor,
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
    def resize_map(self, np_path):
        map_tensor = torch.from_numpy(np.load(np_path)).float()
        if len(map_tensor.shape) == 2:  # (H, W)
            map_tensor = map_tensor.unsqueeze(0)  # (1, H, W)
        map_tensor = map_tensor.unsqueeze(0)  # (1, C, H, W)
        map_tensor = F.interpolate(map_tensor, size=(self.opt.load_size, self.opt.load_size), mode='bilinear', align_corners=False)
        return map_tensor.squeeze(0)  # (C, H, W)
