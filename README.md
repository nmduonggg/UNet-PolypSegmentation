# Medical Image Segmentation - Polyp Detection using UNet

### For inference, run the following codes sequentially

- Setting on Kaggle, if you run on local machine, you can skip this step:
```bash
! git clone https://github.com/nmduonggg/UNet-PolypDetection.git
! cd /kaggle/working/UNet-PolypDetection
```

- Download trained weights
```bash
! bash download.sh
```

- Run infer.py file
```bash
! python infer.py
```