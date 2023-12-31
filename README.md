# Medical Segmentation - Polyp Detection using UNet

UNet is a SOTA segmentation model and used popularly in medical images segmentation. Its structure is simply the combination of Convolution, BatchNorm, and an fully connected network as the feature extractor. In this repo, I use ResNet50 as an extractor thanks to this nice [repository](https://github.com/mberkay0/pretrained-backbones-unet). This repository contains solution for the last year Kaggle [competition](https://www.kaggle.com/competitions/bkai-igh-neopolyp/overview). The trained weight can be found via this bash [file](./download.sh)

![ResUNet](https://th.bing.com/th?id=OIP.lvXoKMHoPJMKpKK7keZMEAHaE7&w=306&h=204&c=8&rs=1&qlt=90&o=6&dpr=2&pid=3.1&rm=2)

### For inference, run the following codes sequentially

- Installing necessary libraries for pretrained
```bash
!pip install git+https://github.com/mberkay0/pretrained-backbones-unet -q
```

- Setting on Kaggle, if you run on local machine, you can skip this step:
```bash
!git clone https://github.com/nmduonggg/UNet-PolypSegmentation.git
%cd /kaggle/working/UNet-PolypSegmentation
```

- Download trained weights
```bash
!bash download.sh
```

- Run infer.py file. Note that the infer.py has been configured for Kaggle competition running. In case that you want to run locally, fix the path in [infer.py](./infer.py)
```bash
!python infer.py
```

