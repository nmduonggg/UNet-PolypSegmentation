import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super(UNetDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)
        
        images_list = [images_path + image_name for image_name in images_list]
        self.masks_paths = [masks_path]
        
        self.images_list = images_list
        self.transform = transform
        
    def add(self, dts: Dataset):
        self.images_list += dts.images_list
        self.masks_paths += dts.masks_paths
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        img_name = os.path.basename(img_path)
        for mp in self.masks_paths:
            if not os.path.exists(mp + img_name): 
                continue
            mask_path = mp + img_name
        
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
            
        if (self.transform):
            data = self.transform(data) / 255
            label = self.transform(label) / 255

        label = torch.where(label>0.65, 1.0, 0.0)
        
        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        
        return data, label
    
    def __len__(self):
        return len(self.images_list)