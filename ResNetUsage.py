from ResNetModel import ResNet, ResBlock
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from torchvision import datasets,transforms

#就用resnet做了，弄了再说,tmd
EPOCHS = 10
pre_epoch = 0
BATCH_SIZE = 2048
LR = 0.01
data_train_dir = '/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTrain'
# data_train_dir_notflaw = '/Users/jiangxuanke/Desktop/ISSM/issm2020-ai-challenge/semTrain'
# data_test_dir = '/Users/jiangxuanke/Desktop/ISSM/issm2020-ai-challenge/semTest'

device = torch.device("cpu")

transform_train = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

train_datasets = datasets.ImageFolder(data_train_dir, transform = transform_train)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=2)
# test_datasets = datasets.ImageFolder(data_test_dir, transform = transform_test)
# test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True, num_workers=2)
#

net = ResNet(ResBlock, [2, 2, 2]).to(device)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train net
pre_epoch_total_step = len(train_dataloader)
current_lr = LR

for epoch in range(EPOCHS):
    print("EPOCH： ", 11-EPOCHS)
    for i, (x, y) in enumerate(train_dataloader):

        x = x.to(device)
        y = y.to(device)

        # forward
        print('Processing')
        prediction = net(x)
        loss = criterion(prediction, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}"
            print(template.format(epoch+1, EPOCHS, i+1, pre_epoch_total_step, loss.item()))

    # decay learning rate
    if (epoch+1) % 20 == 0:
        current_lr = current_lr/2
        update_lr(optimizer, current_lr)
