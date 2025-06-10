import os
import argparse
import numpy as np
import skimage.io as io
from skimage.draw import polygon
import torch
from pycocotools.coco import COCO

from models.networks.mask_generator import MaskProcessorModel

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_file', type=str,
                    default="C:/Users/Fujinet/Documents/ws/CleavageEmbryov1.1/CleavageEmbryov1.1/annotations/instances_train.json")
parser.add_argument('--output_dir', type=str, default="./datasets/cleavageembryov")
parser.add_argument('--state', type=str, default='train')
opt = parser.parse_args()

print("annotation file at:", opt.annotation_file)
print("output dir at:", opt.output_dir)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)
output_instance_dir = os.path.join(opt.output_dir, f"{opt.state}_inst")
output_direction_dir = os.path.join(opt.output_dir, f"{opt.state}_direction")
output_distance_dir = os.path.join(opt.output_dir, f'{opt.state}_distance')
output_semantic_dir = os.path.join(opt.output_dir, f"{opt.state}_semantic")
os.makedirs(output_instance_dir)
os.makedirs(output_direction_dir)
os.makedirs(output_distance_dir)
os.makedirs(output_semantic_dir)
coco = COCO(opt.annotation_file)
# Tạo model
processor = MaskProcessorModel()
# Lấy ID của 'fragment'

fragment_cat_id = 2

imgIds = coco.getImgIds()
for ix, id in enumerate(imgIds):
    if ix % 50 == 0:
        print(f"{ix} / {len(imgIds)}")

    img_info = coco.loadImgs(id)[0]
    filename = img_info["file_name"].replace("jpg", "png")
    h, w = img_info["height"], img_info["width"]
    inst_path = os.path.join(output_instance_dir, filename)
    
    inst_img = np.zeros((h, w), dtype=np.int32)
    label_img = np.zeros((h, w), dtype=np.uint8)

    annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
    anns = [ann for ann in coco.loadAnns(annIds)
            if fragment_cat_id is None or ann["category_id"] != fragment_cat_id]

    semantic_masks = torch.zeros((h, w), dtype=torch.float32)
    direction_maps = torch.zeros((h, w), dtype=torch.float32)
    distance_maps = torch.zeros((h, w), dtype=torch.float32)
    count = 1
    for ann in anns:
        if isinstance(ann["segmentation"], list):
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                inst_img[rr, cc] = 255
                mask_tensor = torch.tensor(inst_img, dtype=torch.int32)

                semantic_mask, direction_map, distance_map = processor(mask_tensor)

                semantic_masks += semantic_mask.squeeze(0)
                direction_maps += direction_map.squeeze(0)
                distance_maps += distance_map.squeeze(0)
            count += 1


    io.imsave(inst_path, inst_img.astype(np.uint8))
    # Đường dẫn để lưu các file .npy
    base_name = os.path.splitext(os.path.basename(inst_path))[0]
    semantic_path = os.path.join(output_semantic_dir, f"{base_name}.npy")
    direction_path = os.path.join(output_direction_dir, f"{base_name}.npy")
    distance_path = os.path.join(output_distance_dir, f"{base_name}.npy")

    # Chuyển sang NumPy và lưu
    np.save(semantic_path, semantic_masks.cpu().numpy())
    np.save(direction_path, direction_maps.cpu().numpy())
    np.save(distance_path, distance_maps.cpu().numpy())


