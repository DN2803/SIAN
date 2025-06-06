import os
import argparse
import numpy as np
import skimage.io as io
from skimage.draw import polygon, polygon_perimeter
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_file', type=str,
                    default="C:/Users/Fujinet/Documents/ws/CleavageEmbryov1.1/CleavageEmbryov1.1/annotations/instances_train.json")
parser.add_argument('--output_dir', type=str, default="./cleavageembryov/train_inst/")
parser.add_argument('--state', type=str, default='train')
opt = parser.parse_args()

print("annotation file at:", opt.annotation_file)
print("output dir at:", opt.output_dir)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)
output_instance_dir = os.path.join(opt.output_dir, f"{opt.state}_inst")
output_contour_dir = os.path.join(opt.output_dir, f"{opt.state}_contour")
os.makedirs(output_instance_dir)
coco = COCO(opt.annotation_file)

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
    
    inst_img = np.zeros((h, w), dtype=np.uint8)
    label_img = np.zeros((h, w), dtype=np.uint8)

    annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
    anns = [ann for ann in coco.loadAnns(annIds)
            if fragment_cat_id is None or ann["category_id"] != fragment_cat_id]


    count = 1
    for ann in anns:
        if isinstance(ann["segmentation"], list):
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                inst_img[rr, cc] = 255
                # Thêm contour
                rr_contour, cc_contour = polygon_perimeter(poly[:, 1] - 1, poly[:, 0] - 1, shape=inst_img.shape)
                inst_img[rr_contour, cc_contour] = 128
            count += 1


    io.imsave(inst_path, inst_img.astype(np.uint8))