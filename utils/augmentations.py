import torch
from torchvision import transforms


def base_transforms(img_size):
    
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.41099029779434204, 0.3926398754119873, 0.3452613949775696],
                            std=[0.16373096406459808, 0.15447314083576202, 0.14317607879638672])
    ])
    
    return base_transform

def advanced_augmentations(image_tensor, img_size):

    pil_img = transforms.ToPILImage()(image_tensor)
    
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])
    
    return augmentation(pil_img)

def batch_augmentations(batch_images, img_size):
    """
    Принимает батч изображений в виде тензора [B, C, H, W] и возвращает батч аугментированных изображений.
    """
    augmented_images = []
    for img in batch_images:
        augmented_images.append(advanced_augmentations(img, img_size))
    return torch.stack(augmented_images)