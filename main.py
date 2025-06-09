import json
import numpy as np
import cv2
from skimage.draw import polygon_perimeter
from models.networks.mask_generator import MaskProcessorModel
import torch
import matplotlib.pyplot as plt

with open('../example/20230304-17097-3-F0-176.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

shapes = data['shapes']
h = data['imageHeight']
w = data['imageWidth']
image = np.zeros((h, w), dtype=np.uint8)
# Tạo model
processor = MaskProcessorModel()
semantic_masks = torch.zeros((h, w), dtype=torch.float32)
direction_maps = torch.zeros((h, w), dtype=torch.float32)
distance_maps = torch.zeros((h, w), dtype=torch.float32)

for shape in shapes:
    points = shape['points']
    poly = np.array(points)
    if len(points) < 3:
        continue  # bỏ qua các hình không đủ điểm

    instance_mask = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    
    if shape['label'] == 'cell':
        # Tô vùng bên trong
        cv2.drawContours(image, [instance_mask], contourIdx=-1, color=127, thickness=-1)
        
        # Tạo binary mask để truyền vào model
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [instance_mask], contourIdx=-1, color=1, thickness=-1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        instance_mask_tensor = mask_tensor.detach().clone().to(torch.int32)
        # Gọi model
        semantic_mask, direction_map, distance_map = processor(instance_mask_tensor)

        # Cộng dồn kết quả
        semantic_masks += semantic_mask.squeeze(0)
        direction_maps += direction_map.squeeze(0)
        distance_maps += distance_map.squeeze(0)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(image, cmap='nipy_spectral')
axs[0].set_title("Instance Mask")
axs[0].axis('off')

axs[1].imshow(semantic_masks.cpu().numpy(), cmap='gray')
axs[1].set_title("Semantic Mask")
axs[1].axis('off')


axs[2].imshow(direction_maps.cpu().numpy(), cmap='gray' )
axs[2].set_title("Direction Map")
axs[2].axis('off')

axs[3].imshow(distance_maps.cpu().numpy(), cmap='gray')
axs[3].set_title("Distance Map")
axs[3].axis('off')
plt.tight_layout()
plt.show()
