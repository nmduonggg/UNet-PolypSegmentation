# Number of class in the data set (3: neoplastic, non neoplastic, background)
num_classes = 3

# Number of epoch
epochs = 5

# Hyperparameters for training 
learning_rate = 3e-6
batch_size = 8
display_step = 50

# Data path
train_path = "/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/train/train/"
valid_path = "/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/valid/valid/"
masks_path =  "/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/train_gt/train_gt/"

# Augment data path
augment_path = "/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/augment_v2/images/"
augment_masks =  "/mnt/disk1/nmduong/hust/bkai-igh-polyp/data/augment_v2/masks/"

# Model path
pretrained_path = './pretrained/unet_model.pth'
checkpoint_path = './output/unet_model.pth'