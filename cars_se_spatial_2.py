'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import logging
import argparse
import torchvision
#from models import *
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
#from my_pooling import my_MaxPool2d
import torchvision.transforms as transforms
#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, type=bool, help='resume from checkpoint')
args = parser.parse_args()
logging.info(args)
#from rate import CyclicScheduler




store_name = "4"
# setup output
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name 





nb_epoch = 100



try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')


results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc,test_loss\n')
results_test_file.flush()



use_cuda = torch.cuda.is_available()

#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



#trainset    = torchvision.datasets.ImageFolder(root='./train', transform=transform_train)
trainset    = torchvision.datasets.ImageFolder(root='/data/changdongliang/StandCars/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)


#testset = torchvision.datasets.ImageFolder(root='./test', transform=transform_test)
testset = torchvision.datasets.ImageFolder(root='/data/changdongliang/StandCars/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        x = self.sigmoid(x)

        channel_feature = module_input * x


        return channel_feature


class Spatial_SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(Spatial_SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)


        x = self.softmax(x)



        size = [module_input.size()[0],module_input.size()[1],module_input.size()[2],module_input.size()[3]]


        spatial_mask = module_input * x
        size = [spatial_mask.size()[0],1,spatial_mask.size()[2],spatial_mask.size()[3]]
        spatial_mask = torch.sum(spatial_mask,1).resize(size[0],size[2]*size[3])
        spatial_mask = self.sigmoid(spatial_mask)
        spatial_mask = spatial_mask.resize(size[0],1,size[2],size[3])
        spatial_feature = module_input * spatial_mask
        return spatial_feature

# Model

print('==> Building model..')


import torchvision.models as models

net = models.vgg16_bn(pretrained=True)
# 1024 * 7 * 7 
# for param in net.parameters():
#     param.requires_grad = False

class model_bn(nn.Module):
    def __init__(self, model, feature_size,classes_num):

        super(model_bn, self).__init__() 
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.se_module = SEModule(512, reduction=16)
        self.s_se_module = Spatial_SEModule(512, reduction=16)
        self.conv = nn.Conv2d(1024, 512, kernel_size=1,padding=0)
        self.num_ftrs = 512*7*7
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
    def forward(self, x):
        #pdb.set_trace()
        x = self.features(x)
        x_1=self.se_module(x)
        x_2=self.s_se_module(x)
        x = torch.cat([x_1,x_2],1)
        x = self.conv(x)

        #x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net =model_bn(net, 512, 196)

# net = VGG('VGG11')
x = torch.randn(2,3,224,224)
print(net(Variable(x)).size())

if use_cuda:
    net.cuda()
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
# scheduler = CyclicScheduler(base_lr=0.00001, max_lr=0.01, step=2050., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #optimizer.param_groups[0]['lr'] = scheduler.get_rate()
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)


        loss = criterion(outputs, targets)


        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc,train_loss))
    results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc,train_loss))
    results_train_file.flush()
    return train_acc, train_loss

def test(epoch):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        #outputs = net(inputs[0].unsqueeze(0))


        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('Iteration %d, test_acc = %.4f,test_loss = %.4f' % (epoch, test_acc,test_loss))
    results_test_file.write('%d, %.4f,%.4f\n' % (epoch, test_acc,test_loss))
    results_test_file.flush()

    return test_acc
 



def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)



# optimizer = optim.SGD(net.classifier.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)



optimizer = optim.SGD([
                        {'params': net.classifier.parameters(), 'lr': 0.1},
                        {'params': net.se_module.parameters(),  'lr': 0.1},
                        {'params': net.s_se_module.parameters(),'lr': 0.1},
                        {'params': net.features.parameters(),   'lr': 0.01}
                        
                     ], 
                      momentum=0.9, weight_decay=1e-4)



max_val_acc = 0
for epoch in range(0, nb_epoch):
    optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch)
    optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch)
    optimizer.param_groups[2]['lr'] = cosine_anneal_schedule(epoch)
    optimizer.param_groups[3]['lr'] = cosine_anneal_schedule(epoch) / 10
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train(epoch)
    test_acc = test(epoch)
    if test_acc>max_val_acc:
        max_val_acc = test_acc
        torch.save(net.state_dict(), store_name+'.pth')
print(max_val_acc)

