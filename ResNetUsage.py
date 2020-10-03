from ResNetModel import GlobalAvgPool2d, Residual, resnet_block,FlattenLayer
import time
import torch.nn as nn
import torch
import torchvision
from torchvision import datasets,transforms
from PIL import Image
#修改称能够显示accuraccy的样子，然后加入test，看test的分类
#就用resnet做了，弄了再说
EPOCHS = 10
pre_epoch = 0
BATCH_SIZE = 16
LR = 0.01
data_train_dir = '/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTrain'
# data_train_dir_notflaw = '/Users/jiangxuanke/Desktop/ISSM/issm2020-ai-challenge/semTrain'
data_test_dir = '/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTest'

device = torch.device("cpu")

normalize = transforms.Normalize(mean=[0.286], std=[0.352])#对像素值归一化
transform_train = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize
])

train_datasets = datasets.ImageFolder(data_train_dir, transform = transform_train)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# test_datasets = datasets.ImageFolder(data_test_dir, transform = transform_test)
# test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# train_datasets = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_train)
# train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# test_datasets = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform_test)
# test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=100, shuffle=False, num_workers=2)

net = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # TODO: 缩小感受野, 缩channel
    nn.BatchNorm2d(32),
    nn.ReLU())
# nn.ReLU(),
# nn.MaxPool2d(kernel_size=2, stride=2))   # TODO：去掉maxpool缩小感受野

# 然后是连续4个block
net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))  # TODO: channel统一减半
net.add_module("resnet_block2", resnet_block(32, 64, 2))
net.add_module("resnet_block3", resnet_block(64, 128, 2))
net.add_module("resnet_block4", resnet_block(128, 256, 2))
# global average pooling
net.add_module("global_avg_pool", GlobalAvgPool2d())
# fc layer
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))

def evaluate_accuracy(data_iter, net, device=None):
	#评估模型在测试集的准确率
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()  # 改回训练模式
    return acc_sum / n


def train_model(net, train_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # test_acc = evaluate_accuracy(test_iter, net)
        # 输出evaluate的结果
        print('epoch %d, loss %.4f, train acc %.3f,  time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,  time.time() - start))
        # if test_acc > best_test_acc:
        #     print('find best! save at model/best.pth')
        #     best_test_acc = test_acc
        #     torch.save(net.state_dict(), 'model/best.pth')


lr, num_epochs = 0.01, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
train_model(net, train_dataloader, batch_size, optimizer, device, num_epochs)




