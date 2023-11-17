import torch
import time
import torch.nn as nn 
from tqdm import tqdm

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)
        
def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# Train function for each epoch
def train(train_dataloader, valid_dataloader, learning_rate_scheduler, epoch, display_step):
    
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learning_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    test_coef_epoch = 0
    last_loss = 999999999
    model.train()
    
    total_step = len(train_dataloader)
    for i, (data,targets) in tqdm(enumerate(train_dataloader), total=total_step):
        
        # Load data into GPU
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
    global model
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