import torch
import os
import torch.nn.functional as F 
from torchvision.transforms import Compose,\
                                    Resize,\
                                    InterpolationMode, \
                                    PILToTensor, \
                                    ToPILImage
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from backbones_unet.model.unet import Unet
from configs import *

device='cuda:0' if torch.cuda.is_available() else 'cpu'
trained_path = '/mnt/disk1/nmduong/hust/bkai-igh-polyp/UNet-PolypDetection/output/unet_model.pth'
transform = Compose([
    Resize((512, 512)),
    PILToTensor()])

model = Unet(in_channels=3, 
             num_classes=num_classes)
checkpoint = torch.load(pretrained_path, map_location=device)['model']

new_checkpoint = dict()
for k in checkpoint:
    new_checkpoint[k[7:]] = checkpoint[k]
    
model.load_state_dict(new_checkpoint)
model.to(device)

class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform):
        super(UNetTestDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        images_list = [i for i in images_list]
        self.images_path = images_path
        
        self.images_list = images_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_name = self.images_list[index]
        img_path = self.images_path + img_name
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255        
        return data, img_name, h, w
    
    def __len__(self):
        return len(self.images_list)
    
path = '/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/test/test/'
# path = '/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/valid/valid/'
unet_test_dataset = UNetTestDataClass(path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=1, shuffle=False)

model.eval()
if not os.path.isdir("./predicted_masks"):
    os.mkdir("./predicted_masks")
for _, (img, name, H, W) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    a = name
    b = img
    h = H
    w = W
    
    with torch.no_grad():
        predicted_mask = model(b.to(device))
    for i in range(len(a)):
        image_id = a[i].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask.detach().cpu()[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join("./predicted_masks/", filename))
        
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

MASK_DIR_PATH = './predicted_masks' # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)