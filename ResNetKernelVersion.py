# %% [code]
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable as V
import csv
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)

        self.fc = nn.Sequential(nn.Linear(64 * 64 * 32, num_classes))  # fully connected

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out_size = out.size(0)*out.size(1)*out.size(2)*out.size(3)
        # out_size = out_size//128
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.Linear(64, 10)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss_MultiLabel(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss_MultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        criterion = FocalLoss(self.alpha, self.gamma, self.size_average)
        loss = torch.zeros(1, target.shape[1]).cuda()

        # 对每个 Label 计算一次 Focal Loss
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:, label], target[:, label])
            loss[0, label] = batch_loss.mean()

        # Loss Function的常规操作
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


dir_train = "../input/settrain/semTrain"
dir_test = "../input/settest/semTest/semTest/"
dir_model_save = "../input/model-save/semResModel"
# dir_test_result = "../output/testResult"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 100  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 16  # 批处理尺寸(batch_size) 16
LR = 0.05  # 学习率 0。05

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(90),
    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_validation = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #???????
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #???????
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.ImageFolder(dir_train, transform=transform_train)  # 训练数据集
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

validationset = torchvision.datasets.ImageFolder(dir_train, transform=transform_train)
validation_dataloader = torch.utils.data.DataLoader(validationset, batch_size=BATCH_SIZE, shuffle=True,
                                                    num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.05,
                      weight_decay=0)

# 训练
if __name__ == "__main__":
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("log.txt", "w") as f2:
        for epoch in range(pre_epoch, EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            train_accuracy = 0
            for i, data in enumerate(train_dataloader):
                # 准备数据
                length = len(train_dataloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                #                 if 100. * correct / total > train_accuracy:
                #                     torch.save(net.state_dict(), dir_model_save)
                #                     net.save("semResModel")
                #                   '%s/net_%03d.pth' % (args.outf, epoch + 1)
                f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                f2.write('\n')
                f2.flush()

    #     net = ResNet18().to(device)  # 先定义net的结构
    #     net.load_state_dict(torch.load(dir_model_save))
    print("Starting Test!")

    net = net.eval()
    resultFrame = pd.DataFrame(columns=["Id", "LABEL"])

    results = []
    image_number = 0
    img_test = os.listdir(dir_test)
    img_test.sort()

    for i in range(len(img_test)):
        img = Image.open(dir_test + img_test[i])
        input = transform_test(img).cuda()

        input = input.unsqueeze(0)
        input = V(input)
        score = net(input)

        probability = torch.nn.functional.softmax(score, dim=1)

        max_value, index = torch.max(probability, 1)
        image_number += 1
        probability = np.round(probability.cpu().detach().numpy())
        index = np.round(index.cpu().detach().numpy())
        print(image_number)
        resultFrame = resultFrame.append([image_number, index[0] + 1])

    resultFrame.to_csv('TestResult.csv', index=False)



