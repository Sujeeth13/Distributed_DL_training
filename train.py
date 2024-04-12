"""### Import the dependencies"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import argparse
import time
import os
from matplotlib import pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

"""### RestNet18 Model Setup"""

class BasicBlock(nn.Module):
    def __init__(self,kernel_size=(3,3),in_channels=64,out_channels=64,stride=1,padding=1,is_batch_norm=True):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.batchN = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        self.skip = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=stride,bias=False)
        self.is_batch_norm = is_batch_norm

    def forward(self,x):
        out = self.conv1(x)
        if self.is_batch_norm:
            out = self.batchN(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.is_batch_norm:
            out = self.batchN(out)
        out = self.relu(out)
        out = out + self.skip(x) # residual connections
        out = self.relu(out)
        return out

class SubGroup(nn.Module):
    def __init__(self,blocks=2,in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1,is_batch_norm=True):
        super(SubGroup,self).__init__()
        self.blocks = nn.ModuleList([BasicBlock(kernel_size,in_channels,out_channels,stride,padding,is_batch_norm) if i==0
                                     else BasicBlock(kernel_size,out_channels,out_channels,stride=1,padding=1,is_batch_norm=is_batch_norm)
                                     for i in range(blocks)])

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNet18(nn.Module):
    def __init__(self,sub_groups,is_batch_norm=True):
        super(ResNet18,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 64,kernel_size = (3,3),stride=(1,1),padding=(1,1),bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=64)
        self.sb_1 = SubGroup(2,64,sub_groups[0],kernel_size=(3,3),stride=1,padding=1,is_batch_norm=is_batch_norm)
        self.sb_2 = SubGroup(2,sub_groups[0],sub_groups[1],kernel_size=(3,3),stride=2,padding=1,is_batch_norm=is_batch_norm)
        self.sb_3 = SubGroup(2,sub_groups[1],sub_groups[2],kernel_size=(3,3),stride=2,padding=1,is_batch_norm=is_batch_norm)
        self.sb_4 = SubGroup(2,sub_groups[2],sub_groups[3],kernel_size=(3,3),stride=2,padding=1,is_batch_norm=is_batch_norm)
        self.avg_pool = nn.AvgPool2d(4)
        # self.flatten = nn.Flatten()
        self.linear = nn.Linear(512,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        out = self.conv_1(x)
        if is_batch_norm:
            out = self.batch_norm_1(out)
        out = self.sb_1(out)
        out = self.sb_2(out)
        out = self.sb_3(out)
        out = self.sb_4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # out = self.flatten(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out

"""### Downloading the data"""

def dataload(batch_size,num_workers):
    # Define the sequence of transformations
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random cropping with size 32x32 and padding 4
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flipping with probability 0.5
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalize each image's RGB channels
    ])

    # Import the CIFAR10 dataset
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False,
                                sampler=DistributedSampler(cifar10_dataset,shuffle=True), num_workers=num_workers)

    # Example: Iterate over the DataLoader (Here, we just print the size of first batch for demonstration)
    for images, labels in cifar10_loader:
        if int(os.environ['RANK']) == 0:
            print(f'Batch size: {images.size(0)}')
        # Break after the first batch for demonstration purposes
        break

    return cifar10_loader

"""### Model & Hyperparameter initiation"""

def setup(optim_name,device,is_batch_norm):
    model = ResNet18([64,128,256,512],is_batch_norm).to(device)
    criterion = nn.CrossEntropyLoss() # loss function
    # picking the optimizer
    optim = None
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=wd)
    elif optim_name == 'sgd_nesterov':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=wd,nesterov=True)
    elif optim_name == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr,weight_decay=wd)
    elif optim_name == 'adadelta':
        optim = torch.optim.Adadelta(model.parameters(), lr=lr,weight_decay=wd)
    elif optim_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    else:
        print("Error: Optimizer not found")
        exit()
    return (model,criterion,optim)

"""### Training setup"""

def train(cifar10_loader,model,epochs,device,criterion,optim,local_rank,rank):
    print(f"Proc {rank} using device {device}")
    model = DDP(model,device_ids=[local_rank])
    total_step = len(cifar10_loader)
    for epoch in range(epochs):
        avg_acc = 0
        avg_loss = 0
        total_batch = 0
        avg_data_time = 0
        avg_training_time = 0
        avg_total_time = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cifar10_loader.sampler.set_epoch(epoch)

        print(f"Epoch {epoch + 1}:")
        cifar10_iterator = iter(cifar10_loader)
        try:
            for i in tqdm(range(len(cifar10_loader))):
                model.train()
                data_start_time = time.time()
                images,labels = next(cifar10_iterator)
                data_end_time = time.time()
                avg_data_time += data_end_time - data_start_time
                # Move tensors to configured device
                t_start_time = time.time()
                images = images.to(device)
                labels = labels.to(device)

                optim.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optim.step()

                model.eval()
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                total = labels.size(0)
                accuracy = 100 * correct / total

                avg_acc += (correct/total)
                avg_loss += loss.item()

                total_batch += 1
                t_end_time = time.time()

                avg_training_time += t_end_time - t_start_time
                avg_total_time += t_end_time - data_start_time

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print(f'Proc: {rank} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss/total_batch:.4f}, Accuracy: {avg_acc/total_batch:.2f}, Data loading time: {avg_data_time}, Training Time: {avg_training_time}, Total running time: {avg_total_time}')
        except RuntimeError as e:
            print("ERROR: CUDA OUT OF MEMORY")
            return False
    return True

if __name__ == "__main__":

    n_w = 2
    optim_name = 'sgd'
    lr = 0.1
    momentum = 0.9
    wd = 5e-4
    is_batch_norm = True

    parser = argparse.ArgumentParser(description="DDL for Resnet on cifar")
    parser.add_argument('--batch_size', default=16, type=int, help='per GPU')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=0, type=int)

    args = parser.parse_args()

    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args.batch_size
    epochs = args.epochs

#   os.environ['MASTER_ADDR'] = 'localhost'  # Use the appropriate IP address
#   os.environ['MASTER_PORT'] = '12355'      # Use an appropriate free port

    if not torch.cuda.is_available():
        print("Error: Distrbuted training is not supported without GPU")

    # init the process group for DDL
    init_process_group(backend='nccl',rank=args.rank,world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    while batch_size <= 512: # start the training
        device = torch.device('cuda')
        cifar10_loader = dataload(batch_size,n_w) # load the data
        (model,criterion,optim) = setup(optim_name,device,is_batch_norm) # setup the model and the hyperparameters
        if not train(cifar10_loader,model,epochs,device,criterion,optim,args.local_rank,args.rank):
            print(f"Batch size: {batch_size}")
            break
        batch_size *= 4

    destroy_process_group()
