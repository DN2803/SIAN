from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os
from PIL import Image
import numpy as np
import torch

#TODO: dataloader 
class CleavageEmbryovDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --data_dir
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase
        self.instance_dir = os.path.join(self.root, f'{phase}_instance')
        self.semantic_dir = os.path.join(self.root, f'{phase}_semantic')
        self.direction_dir = os.path.join(self.root, f'{phase}_direction')
        self.distance_dir = os.path.join(self.root, f'{phase}_distance')
    def __getitem__(self, index):
        # Load instance mask (input image)
        inst_path = self.paths[index]
        inst_name = os.path.splitext(os.path.basename(inst_path))[0]
        instance_img = Image.open(inst_path).convert('L')
        if self.opt.load_size > 0:
            instance_img = instance_img.resize((self.opt.load_size, self.opt.load_size))

        instance_tensor = self.transform(instance_img)

        # Load semantic, direction, distance maps
        semantic_path = os.path.join(self.semantic_dir, f'{inst_name}.npy')
        direction_path = os.path.join(self.direction_dir, f'{inst_name}.npy')
        distance_path = os.path.join(self.distance_dir, f'{inst_name}.npy')

        semantic_map = torch.from_numpy(np.load(semantic_path)).float()
        distance_map = torch.from_numpy(np.load(distance_path)).float()
        direction_map = torch.from_numpy(np.load(direction_path)).float()
        return {
            'label': instance_tensor,
            'inst': instance_tensor,
            'semantic': semantic_map,
            'distance': distance_map,
            'direction': direction_map,
            'path': inst_path
        }
    def __len__(self):
        return len(self.paths)
