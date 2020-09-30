from ResNetModel import ResNet, ResBlock
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
from torchvision import datasets,transforms

#就用resnet做了，弄了再说,tmd
EPOCHS = 10
pre_epoch = 0
BATCH_SIZE = 32
LR = 0.01
data_train_dir = '/Users/jiangxuanke/Desktop/ISSM/issm2020-ai-challenge/semTrain'
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
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True, num_workers=2)
# test_datasets = datasets.ImageFolder(data_test_dir, transform = transform_test)
# test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True, num_workers=2)
#

#torch download pictures

model = ResNet(ResBlock, [2, 2, 2]).to(device)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train model
pre_epoch_total_step = len(train_dataloader)
current_lr = LR

# for epoch in range(pre_epoch, EPOCH):
#     print('\nEpoch: %d' % (epoch + 1))
#     net.train()
#     sum_loss = 0.0
#     correct = 0.0
#     total = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # prepare dataset
#         length = len(train_dataloader)
#         inputs, labels = data
#         print(inputs.size())
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#
#         # forward & backward
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         sum_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += predicted.eq(labels.data).cpu().sum()
#         print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
#               % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
#     #get the ac with testdataset in each epoch
#     # print('Waiting Test...')
#     # with torch.no_grad():
#     #     correct = 0
#     #     total = 0
#     #     for data in test_dataloader:
#     #         net.eval()
#     #         images, labels = data
#     #         images, labels = images.to(device), labels.to(device)
#     #         outputs = net(images)
#     #         _, predicted = torch.max(outputs.data, 1)
#     #         total += labels.size(0)
#     #         correct += (predicted == labels).sum()
#     #     print('Test\'s ac is: %.3f%%' % (100 * correct / total))
# print('Train has finished, total epoch is %d' % EPOCH)
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        prediction = model(x)
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
