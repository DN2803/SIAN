import json
import numpy as np
import cv2
from models.networks.mask_generator import InstanceMaskDataset
with open ('../example/20230304-17097-3-F0-176.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
shapes = data['shapes']
h = data['imageHeight']
w = data['imageWidth']
image = np.zeros((h, w), dtype = np.uint8)
for shape in shapes:
    
    points = shape['points']
    print(len(points))
    centroid = np.mean(points, axis=0)
    print(centroid)
    
    instance_mask = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    if(shape['label'] == 'cell'):
        cv2.drawContours(image, [instance_mask], contourIdx=-1, color=127, thickness=-1)
    # else:
    #     cv2.drawContours(image, [instance_mask], contourIdx=-1, color=255, thickness=-1)

cv2.imshow('mask', image)
cv2.waitKey(0)