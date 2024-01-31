from __future__ import print_function
from __future__ import division
import os
import sys
import time
import struct
import shutil
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

best_acc1 = 0.0

class MNIST_Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        x = np.reshape(x, (28, 28))
        x = np.resize(x, (224, 224))
        x = np.stack((x,)*3, axis=0)
        x = torch.from_numpy(x)
        x = x.float()
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_mnist(path: str, kind: str='train') -> np.ndarray:
    image_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))
    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))
    
    with open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        else:
            pass

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        # 在 train 函數內部
        train_losses.append(losses.avg)  # 將每個 epoch 的訓練 loss 加入列表
        train_accuracies.append(top1.avg)  # 將每個 epoch 的訓練 accuracy 加入列表


      

        # 繪製訓練和測試 loss 圖表
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(test_losses, label='Testing Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')  # 將圖表保存為圖片
        plt.show()

        # 繪製訓練和測試 accuracy 圖表
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(test_accuracies, label='Testing Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('accuracy_plot.png')  # 將圖表保存為圖片
        plt.show()



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
        else:
            pass

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint(state, is_best, filename='model.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    lr = 1e-3 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

if __name__ == '__main__':
    model = models.vgg19_bn(pretrained=False)  # Use VGG19 with batch normalization
    model.classifier._modules['6'] = nn.Linear(4096, 10)  # Modify the output layer


# if __name__ == '__main__':
#     model = models.vgg16(pretrained = False)
#     model.classifier._modules['6'] = nn.Linear(4096, 10)
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Using CPU.')
    
    training_set, training_label = load_mnist('./Q4', 'train')
    testing_set, testing_label = load_mnist('./Q4', 't10k')

    train_loader = DataLoader(
        MNIST_Dataset(training_set, training_label),
        batch_size=64, shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        MNIST_Dataset(testing_set, testing_label),
        batch_size=64, shuffle=True,
        pin_memory=True
    )

    summary(model, (3, 224, 224), device='cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0)
    
    for epoch in range(0, 40):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        acc1 = validate(test_loader, model, criterion)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss' : criterion
        }, is_best)