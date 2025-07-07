# src/dataset_ssl.py
from pathlib import Path
from typing import Dict, Sequence, Tuple, List
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .augmentations import SSLPairTransform


class CocoImagePairDataset(Dataset):
    """
    Для PixPro-/SimCLR-подобной предтренировки.
        transform_pair(img) -> (view1, view2)
    Возвращает пару t1, t2 – **Tensor[3,H,W]** после .strong().
    """
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

    def __init__(self,
                 images_dir: str | Path,
                 ann_file: str | Path,
                 transform_pair: SSLPairTransform):
        self.root = Path(images_dir)
        self.coco = COCO(str(ann_file))
        self.ids  = sorted(self.coco.getImgIds())
        self.transform_pair = transform_pair

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(self.root / file_name).convert("RGB")
        return self.transform_pair(img)           # strong(), strong()


class CocoDetDataset(CocoDetection):
    """
    COCO-детектор с «базовой» аугментацией:
      • Resize → (img_size, img_size)   (как в SSLPairTransform.base)
      • Normalize(mean,std)
    Боксы масштабируются под новое разрешение.
    """
    def __init__(self,
                 images_dir: str | Path,
                 ann_file: str | Path,
                 img_size: int = 224,
                 remap_labels: bool = True):
        super().__init__(root=str(images_dir), annFile=str(ann_file))
        self.img_size = img_size
        self.base_tf  = SSLPairTransform(img_size).base
        if remap_labels:
            cats = self.coco.loadCats(self.coco.getCatIds())
            self._map = {c["id"]: i + 1 for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))}
        else:
            self._map = None

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        pil_img, anns = super().__getitem__(index)
        w0, h0 = pil_img.size
        img = self.base_tf(pil_img)              # Tensor[3,H,W] с H=W=img_size

        boxes, labels = [], []
        for a in anns:
            if a.get("iscrowd", 0) or "bbox" not in a:   # пропускаем crowd-объекты
                continue
            b = self._xywh_to_xyxy(a["bbox"])
            # масштабируем под новое разрешение
            b[0] = b[0] / w0 * self.img_size
            b[2] = b[2] / w0 * self.img_size
            b[1] = b[1] / h0 * self.img_size
            b[3] = b[3] / h0 * self.img_size
            if b[2] <= b[0] or b[3] <= b[1]:            # защита от нулевой площади
                continue
            boxes.append(b)
            label = self._map[a["category_id"]] if self._map else a["category_id"]
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes,  dtype=torch.float32),   # [N,4]
            "labels": torch.tensor(labels, dtype=torch.int64)     # [N]
        }
        return img, target
    
    def _xywh_to_xyxy(b: Sequence[float]) -> List[float]:
        x, y, w, h = b
        return [x, y, x + w, y + h]
    
    def detection_collate(batch):
        imgs, targets = zip(*batch)
        return torch.stack(imgs, 0), list(targets)