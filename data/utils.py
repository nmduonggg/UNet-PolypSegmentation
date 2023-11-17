from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, GaussianBlur, Compose, InterpolationMode, PILToTensor, RandomHorizontalFlip, \
    RandomAdjustSharpness, RandomAutocontrast, RandomVerticalFlip
import torch.nn as nn 

transform = Compose([
                    Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
                    RandomHorizontalFlip(p=0.6),
                    RandomVerticalFlip(p=0.2),
                    PILToTensor()
                        ])
