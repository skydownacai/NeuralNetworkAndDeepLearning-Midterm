from resnet18 import *
from dataset import cifar100_dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import pandas as pd

#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set hyperparameter
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01
CIFAR_PATH = "data"
model_name = "baseline"

#define ResNet18
net = ResNet18().to(device)

#prepare dataset
trainloader, testloader = cifar100_dataset(CIFAR_PATH, train_batch_size = BATCH_SIZE,test_batch_size = BATCH_SIZE,num_workers=0)
length = len(trainloader)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

History = {"train_loss":[],"train_acc":[],"test_acc":[]}

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
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print ac & loss in each batch
        sum_loss += loss.cpu().item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
    

    History["train_acc"].append(float(100. * correct / total))
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