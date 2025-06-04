import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from skimage.morphology import medial_axis
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt

class InstanceMaskDataset(Dataset):
    def __init__(self, instance_masks):
        self.instance_masks = instance_masks

    def __len__(self):
        return len(self.instance_masks)

    def __getitem__(self, idx):
        # Trả về instance_mask tensor
        instance_mask = self.instance_masks[idx]
        return torch.tensor(instance_mask, dtype=torch.int32)

def generate_semantic_direction_distance(instance_mask_np):
    semantic_mask = (instance_mask_np > 0).astype(np.float32)

    direction_map = np.zeros((2, *instance_mask_np.shape), dtype=np.float32)
    props = regionprops(instance_mask_np)
    for region in props:
        coords = region.coords
        centroid = np.array(region.centroid)
        vectors = coords - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
        unit_vectors = vectors / norms
        for i, (y, x) in enumerate(coords):
            direction_map[0, y, x] = unit_vectors[i, 0]
            direction_map[1, y, x] = unit_vectors[i, 1]

    distance_map = np.zeros(instance_mask_np.shape, dtype=np.float32)
    for region in props:
        nucleus_mask = (instance_mask_np == region.label).astype(np.uint8)
        skeleton = medial_axis(nucleus_mask)
        dist_to_skeleton = distance_transform_edt(~skeleton & nucleus_mask)
        distance_map[nucleus_mask > 0] = dist_to_skeleton[nucleus_mask > 0]

    return semantic_mask, direction_map, distance_map

class MaskProcessorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Không cần layers vì chỉ demo phần tiền xử lý
        # Nếu muốn bạn có thể thêm layers ở đây

    def forward(self, instance_mask):
        # instance_mask: tensor shape (H,W), int type
        instance_mask_np = instance_mask.cpu().numpy()
        semantic_mask, direction_map, distance_map = generate_semantic_direction_distance(instance_mask_np)
        # Chuyển numpy sang tensor
        semantic_mask = torch.tensor(semantic_mask, dtype=torch.float32, device=instance_mask.device)
        direction_map = torch.tensor(direction_map, dtype=torch.float32, device=instance_mask.device)
        distance_map = torch.tensor(distance_map, dtype=torch.float32, device=instance_mask.device)
        return semantic_mask, direction_map, distance_map

    def training_step(self, batch, batch_idx):
        instance_mask = batch  # batch chứa instance_mask
        semantic_mask, direction_map, distance_map = self.forward(instance_mask)

        # Ví dụ: in shape output
        self.log("semantic_mask_mean", semantic_mask.mean())
        self.log("direction_map_mean", direction_map.mean())
        self.log("distance_map_mean", distance_map.mean())

        # Chỉ demo, không có loss thực sự
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

# Demo sử dụng
if __name__ == "__main__":
    # Giả sử có 2 ảnh instance_mask 100x100, 2 object label
    instance_masks = [
        np.zeros((100, 100), dtype=np.int32),
        np.zeros((100, 100), dtype=np.int32)
    ]
    instance_masks[0][30:50, 30:50] = 1
    instance_masks[0][60:80, 60:80] = 2

    instance_masks[1][20:40, 20:40] = 1
    instance_masks[1][50:70, 50:70] = 2

    dataset = InstanceMaskDataset(instance_masks)
    dataloader = DataLoader(dataset, batch_size=1)

    model = MaskProcessorModel()

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(model, dataloader)
