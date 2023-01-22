import torch
from torchvision import transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# specify image transforms for augmentation during training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                            shear=None, resample=False, fillcolor=(255, 255, 255)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# during validation we use only tensor and normalization transforms
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
