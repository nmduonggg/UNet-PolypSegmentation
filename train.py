import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from data import UNetDataClass, transform
from configs import *
from loss import CEDiceLoss
# from model import UNet
from utils import *
from backbones_unet.model.unet import Unet
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

# unet_dataset_not_aug = UNetDataClass(train_path, masks_path, transform=transform)
# unet_dataset_aug_1 = UNetDataClass(train_path, masks_path, transform=augmentation_1)
# unet_dataset_aug_2 = UNetDataClass(train_path, masks_path, transform=augmentation_2)
# unet_dataset_aug_3 = UNetDataClass(train_path, masks_path, transform=augmentation_3)
# unet_dataset_1 = ConcatDataset([unet_dataset_not_aug, unet_dataset_aug_1])
# unet_dataset_2 = ConcatDataset([unet_dataset_1, unet_dataset_aug_2])
# train_dataset = ConcatDataset([unet_dataset_2, unet_dataset_aug_3])

unet_dataset = UNetDataClass(train_path, masks_path, transform=transform)   
# augment_dataset = UNetDataClass(augment_path, augment_masks, transform=None)
# unet_dataset.add(augment_dataset)   # use transform of unet_dataset

train_size=0.8
valid_size=0.2
train_dataset, valid_dataset = random_split(unet_dataset, 
                                            [int(train_size*len(unet_dataset)), int(valid_size*len(unet_dataset))], 
                                            torch.Generator().manual_seed(42))

print("Total train size:", len(train_dataset))
print("Total valid size:", len(valid_dataset))

# train_dataset.add(augment_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# loss
weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
loss_function = CEDiceLoss(weights)

model = Unet(in_channels=3, 
             num_classes=num_classes)
model.apply(weights_init)
checkpoint = torch.load(pretrained_path, map_location=device)['model']

new_checkpoint = dict()
for k in checkpoint:
    new_checkpoint[k[7:]] = checkpoint[k]
    
model.load_state_dict(new_checkpoint)
model.to(device)

# optimizer
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Learning rate scheduler
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

# Train function for each epoch
def train(train_dataloader, valid_dataloader, learning_rate_scheduler, epoch, display_step):
    
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learning_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    test_coef_epoch = 0
    model.train()
    
    total_step = len(train_dataloader)
    for i, (data,targets) in tqdm(enumerate(train_dataloader), total=total_step):
        
        # Load data into GPU
        model.train()
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        # Backpropagation, compute gradients
        loss, _ = loss_function(outputs, targets.long())
        loss.backward()

        # Apply gradients
        optimizer.step()
        
        # Save loss
        train_loss_epoch += loss.item()
        if (i+1) % display_step == 0:
#             accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i+1) * len(data), len(train_dataloader.dataset), 100 * (i+1) * len(data) / len(train_dataloader.dataset), 
                loss.item()))
                  
    print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch/= (i + 1)
    
    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_dataloader, total=len(valid_dataloader)):
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss, test_coef = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()
            test_coef_epoch += test_coef.item()
    
    test_coef_epoch/= (i+1)
    test_loss_epoch/= (i+1)
    
    return train_loss_epoch , test_loss_epoch, test_coef_epoch

# Test function
def test(dataloader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            test_loss += targets.size(0)
            correct += torch.sum(pred == targets).item()
    return 100.0 * correct / test_loss

# train
wandb.init(
    # set the wandb project where this run will be logged
    project= "PolypSegment",
    entity='aiotlab'
)

# Training loop
save_model(model, optimizer, checkpoint_path)
# Initialize lists to keep track of loss and accuracy
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

train_loss_array = []
test_loss_array = []
last_loss = 9999999999999
for epoch in range(epochs):
    train_loss_epoch = 9999999999999
    test_loss_epoch = 99999999999999
    (train_loss_epoch, test_loss_epoch, test_coef_epoch) = train(train_loader, 
                                              valid_loader, 
                                              learing_rate_scheduler, epoch, display_step)
    
    if test_loss_epoch < last_loss:
        save_model(model, optimizer, checkpoint_path)
        last_loss = test_loss_epoch
        print("Save model with loss:", last_loss)
        
    learing_rate_scheduler.step()
    train_loss_array.append(train_loss_epoch)
    test_loss_array.append(test_loss_epoch)
    wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch, 'Valid dice coef': test_coef_epoch,
               "lr": learing_rate_scheduler.get_last_lr()[0]})
    