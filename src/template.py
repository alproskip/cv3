# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 16
max_num_epoch = 100
kernel_size = 5
padding = kernel_size//2
hps = {'lr':0.001}
lr_list = [0.1, 0.001, 0.0001]
number_of_kernel_list = [2, 4, 8]
kernel_size_list = [3, 5]

early_stop_n = 5
out_channel_number = 8

# ---- options ----
DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False
EARLY_STOP = True

# --- imports ---
import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)

# ---- funcs ----

def loss_checking(arr):
    if len(arr) >= 3 :
        if arr[-2] - arr[-1] <= 0.05 :
            res = 0
        else:
            res = 1
    else:
        res = 1
    return res

def get_test_set():
    data_root = 'ceng483-s19-hw3-dataset'
    test_set = hw3utils.HW3ImageFolder(root=data_root,device=device)
    test_set_paths = []
    for img_name, i in test_set.imgs:
        if 'test_inputs' in str(img_name):
            path = img_name
            test_set_paths.append(path)
    selected_paths = random.sample(test_set_paths, 100)
    with open('test_images.txt', 'a') as myfile:
        myfile.truncate(0)
        for path in selected_paths:
            myfile.write(path)
            myfile.write("\n")


# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=padding)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = self.conv1(grayscale_image)
        return x

class Net2(nn.Module):
    def __init__(self) -> None:
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channel_number, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channel_number, 3, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channel_number)


    def forward(self, grayscale_image):
        x = self.conv1(grayscale_image)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x

class Net4(nn.Module):
    def __init__(self) -> None:
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channel_number, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channel_number, out_channel_number, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channel_number, out_channel_number, kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(out_channel_number, 3, kernel_size, padding=padding)

    def forward(self, grayscale_image):
        x = self.conv1(grayscale_image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x

# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
net = Net2().to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
train_loader, val_loader = get_loaders(batch_size,device)

# get_test_set()

if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    net.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

print('training begins')
validation_losses = []

for epoch in range(max_num_epoch):  
    running_loss = 0.0 # training loss of the network
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.
        
        optimizer.zero_grad() # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        print_n = 100 # feel free to change this constant
        if iteri % print_n == (print_n-1):    # print every print_n mini-batches
            print('[%d, %5d] network-loss: %.6f' %
                  (epoch + 1, iteri + 1, running_loss / 100))
            running_loss = 0.0
            # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

        if (iteri==0) and VISUALIZE:
            hw3utils.visualize_batch(inputs,preds,targets)

    if EARLY_STOP:
        val_loss = 0
        count = 0
        for iteri, data in enumerate(val_loader, 0):
            inputs, targets = data
            preds = net(inputs)
            loss = criterion(preds, targets)
            val_loss += loss.item()
            count += 1
        validation_losses.append(val_loss)
        print('[%d, %5d] validation-loss: %.6f' %
                  (epoch + 1, iteri + 1, val_loss / count))

        if loss_checking(validation_losses) == 0:
            break

    print('Saving the model, end of epoch %d' % (epoch+1))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint.pt'))
    hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,'example.png'))

print('Finished Training')
