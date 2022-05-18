from resnet18 import *
from dataset import cifar100_dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np

#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set hyperparameter
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01
CIFAR_PATH = "data"
model_name = "cutmix"

#define ResNet18
net = ResNet18().to(device)

#prepare dataset
trainloader, testloader = cifar100_dataset(CIFAR_PATH, train_batch_size = BATCH_SIZE,test_batch_size = BATCH_SIZE,num_workers=0)
length = len(trainloader)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

History = {"train_loss":[], "test_acc":[]}

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

#train
for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        #prepare dataset
        inputs, labels = data

        mixed_inputs, y_a, y_b, lam = cutmix_data(inputs,labels)
        mixed_inputs = mixed_inputs.to(device)
        y_a = y_a.to(device)
        y_b = y_b.to(device)
        optimizer.zero_grad()
        
        #forward & backward
        outputs = net(mixed_inputs)
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        loss.backward()
        optimizer.step()
        
        #print ac & loss in each batch
        sum_loss += loss.cpu().item()
        print('[epoch:%d, iter:%d] Loss: %.03f' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1)))
    

    History["train_loss"].append(sum_loss / (i + 1))

    #get the ac with testdataset in each epoch
    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum()
        test_acc = float(100 * correct / total)
        print('Test\'s ac is: %.3f%%' % test_acc)
        History["test_acc"].append(test_acc)

    #save model
    net.save_to_file("model\%s_epoch_%d.pth" % (model_name,(epoch+1)))

    #save history file
    pd.DataFrame(History).to_csv("model\%s_history.csv" % model_name)


print('Train has finished, total epoch is %d' % EPOCH)