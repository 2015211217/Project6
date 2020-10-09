import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from ResNetModel import ResNet18
from torch.autograd import Variable as V
import csv
import os
from PIL import Image
import numpy as np
import nni#auto manage the parameter
RCV_CONFIG = nni.get_next_parameter()

dir_train = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTrain"
dir_test = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semTest/"
dir_model_save = "/Users/jiangxuanke/Desktop/issm2020-ai-challenge/semResModel"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 20  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 32  # 批处理尺寸(batch_size)
LR = 0.05  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(128, padding=4),
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差,啊这。。。
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_validation = transforms.Compose([
    transforms.RandomCrop(128),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.RandomCrop(128),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.ImageFolder(dir_train, transform=transform_train)  # 训练数据集
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

validationset = torchvision.datasets.ImageFolder(dir_train, transform=transform_train)
validation_dataloader = torch.utils.data.DataLoader(validationset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)


net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
# 训练
if __name__ == "__main__":
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("log.txt", "w")as f2:
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
                if 100. * correct / total > train_accuracy:
                    torch.save(net.state_dict(), dir_model_save)
                  # '%s/net_%03d.pth' % (args.outf, epoch + 1)
                f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                f2.write('\n')
                f2.flush()

    net = ResNet18().to(device)  # 先定义net的结构
    net.load_state_dict(torch.load(dir_model_save))
    print("Starting Test!")
    net = net.eval()

    csvFile = open("/Users/jiangxuanke/Desktop/issm2020-ai-challenge/TestResult.csv", "w")  # 创建csv文件
    writer = csv.writer(csvFile)
    writer.writerow(["Id", "LABEL"])
    results = []
    image_number = 0
    img_test = os.listdir(dir_test)
    img_test.sort()
    print(img_test)
    for i in range(len(img_test)):
        img = Image.open(dir_test + img_test[i])
        # img = img.convert('RGB')
        input = transform_test(img)

        input = input.unsqueeze(0)

        input = V(input)
        score = net(input)

        probability = torch.nn.functional.softmax(score, dim=1)

        max_value, index = torch.max(probability, 1)
        image_number += 1
        probability = np.round(probability.cpu().detach().numpy())
        index = np.round(index.cpu().detach().numpy())
        writer.writerow([image_number, index[0]+1])
    csvFile.close()

