"""Clean Dataset: 支持 YOLO 格式 (images/ + labels/) 和 ImageFolder 格式"""
import torch
import os, glob
from pathlib import Path
from torchvision import datasets
from .transforms import get_base_transform

# KIIT-MiTA 7 类名称
KIIT_MITA_CLASSES = ["Artilary", "Missile", "Radar", "M.RocketLauncher", "Soldier", "Tank", "Vehicle"]


class CleanDataset(torch.utils.data.Dataset):
    """自动检测数据集格式:
       - YOLO: train/images/  + train/labels/  (KIIT-MiTA)
       - ImageFolder: train/{class0}/, train/{class1}/, ...
    """

    def __init__(self, root, transform=None, img_size=(32, 32)):
        base_tf = get_base_transform(img_size)
        if transform:
            from torchvision import transforms as T
            self.transform = T.Compose([base_tf, transform])
        else:
            self.transform = base_tf

        self.root = root
        train_dir = os.path.join(root, 'train')
        images_dir = os.path.join(train_dir, 'images')
        labels_dir = os.path.join(train_dir, 'labels')

        # 检测 YOLO 格式
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            self._init_yolo(images_dir, labels_dir)
        else:
            self._init_imagefolder(train_dir)

    def _init_yolo(self, images_dir, labels_dir):
        """YOLO 格式: train/images/*.jpeg + train/labels/*.txt"""
        self._mode = 'yolo'
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # 收集所有图片路径
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        self.img_paths = []
        for ext in exts:
            self.img_paths.extend(sorted(glob.glob(os.path.join(images_dir, ext))))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {images_dir} (jpg/jpeg/png/bmp)")

        # 生成 class_to_idx (基于 KIIT-MiTA 7 类)
        self.class_to_idx = {name: i for i, name in enumerate(KIIT_MITA_CLASSES)}
        self.classes = list(KIIT_MITA_CLASSES)

        print(f"[CleanDataset] YOLO mode: {len(self.img_paths)} images, {len(self.classes)} classes")

    def _init_imagefolder(self, train_dir):
        """ImageFolder 格式: train/{class0}, train/{class1}, ..."""
        self._mode = 'imagefolder'
        self._ds = datasets.ImageFolder(root=train_dir, transform=None)
        self.class_to_idx = self._ds.class_to_idx
        self.classes = self._ds.classes
        self.samples = self._ds.samples
        print(f"[CleanDataset] ImageFolder mode: {len(self.samples)} images, {len(self.classes)} classes")

    def __getitem__(self, idx):
        if self._mode == 'yolo':
            return self._getitem_yolo(idx)
        else:
            return self._getitem_imagefolder(idx)

    def _getitem_yolo(self, idx):
        img_path = self.img_paths[idx]

        # 解析对应 label 文件: image_xxx.jpeg → image_xxx.txt
        img_name = Path(img_path).stem  # e.g. "image_s3r2_kiit_1"
        label_path = os.path.join(self.labels_dir, img_name + '.txt')

        # 读取 YOLO label: class_id cx cy w h
        class_id = 0
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        class_id = int(float(parts[0]))  # 字符串 → float → int, 容错
            except Exception:
                class_id = 0

        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), class_id

    def _getitem_imagefolder(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label

    def __len__(self):
        if self._mode == 'yolo':
            return len(self.img_paths)
        return len(self.samples)
