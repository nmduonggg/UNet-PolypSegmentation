# Medical Image Segmentation - Polyp Detection using UNet

UNet is a SOTA segmentation model and used popularly in medical images segmentation. Its structure is simply the combination of Convolution, BatchNorm, and an fully connected network as the feature extractor. In this repo, I use ResNet50 as an extractor

![ResUNet](https://th.bing.com/th?id=OIP.lvXoKMHoPJMKpKK7keZMEAHaE7&w=306&h=204&c=8&rs=1&qlt=90&o=6&dpr=2&pid=3.1&rm=2)

### For inference, run the following codes sequentially

- Setting on Kaggle, if you run on local machine, you can skip this step:
```bash
! git clone https://github.com/nmduonggg/UNet-PolypSegmentation.git
! cd /kaggle/working/UNet-PolypSegmentation
```

- Download trained weights
```bash
! bash download.sh
```

- Run infer.py file
```bash
! python infer.py
```

