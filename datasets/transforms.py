"""数据增强流水线"""
from torchvision import transforms

def get_train_transform(augmentation='light'):
    if augmentation == 'none':
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    elif augmentation == 'light':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])

def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_base_transform(img_size=(32, 32)):
    # 确保是 (H, W) tuple, 支持 int 输入 (等比缩放短边会破坏 batch 对齐)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
