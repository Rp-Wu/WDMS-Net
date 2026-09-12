import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision.transforms import transforms
from Model.WDMS_Net import WDMS_Net
import os

#指定日志及参数保存路径
train_path="$pth$"
writer= SummaryWriter(train_path)
#Tensorboard --logdir=$pth$

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#官方文档提供的三目运算判断是否使用gpu或者cpu方法

#准备数据集
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1087, 0.2388, 0.7831],std=[0.2581, 0.3698, 0.2378]),#This is the mean&std of the original environment train set of 10 person
    # transforms.Normalize(mean=[0.099352, 0.22218624, 0.78959554],std=[0.24717692, 0.35978845, 0.22849967]),#This is the mean&std of the original environment train set of 5 person
    transforms.Resize([299,299])
])


transform_valid = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.099352, 0.22218624, 0.78959554],std=[0.24717692, 0.35978845, 0.22849967]),#这是14679五人数据集的均值和方差
    transforms.Normalize(mean=[0.1087, 0.2388, 0.7831],std=[0.2581, 0.3698, 0.2378]),#这是18949原始十人数据集的均值和方差
    transforms.Resize([299,299])
])

train_data = torchvision.datasets.ImageFolder(root="E:\Environment_Finetuning_DEV21\\train",  transform=transform_train)
train_data_size=len(train_data)
valid_data = torchvision.datasets.ImageFolder(root="E:\Environment_Finetuning_DEV21\\val",transform=transform_valid)
valid_data_size=len(valid_data)
train_loader =DataLoader(train_data,batch_size=16, shuffle=True,num_workers=0)
valid_loader =DataLoader(valid_data,batch_size=16, shuffle=False,num_workers=0)

#保存模型参数
def save_checkpoint(epoch, model, optimizer,best_accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy':best_accuracy
    }, path)

# 加载模型和优化器状态
def load_checkpoint(model, optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    print(f"Checkpoint loaded, starting at epoch {start_epoch}.")
    return start_epoch,best_accuracy

#网络
model = WDMS_Net(num_classes=10)
model.cuda()

#损失函数(交叉熵)
loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()

#优化器
learning_rate = 1e-1#原始学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=1e-2)
checkpoint = torch.load("E:\Gait_Data\state (2).pth")
model.load_state_dict(checkpoint['model_state_dict'])
#冻结卷积层，解冻全连接分类头
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

#设置训练参数
total_train_step=0
count_epoch=0
best_accuracy=0.7

epoch=100
start_time = time.time()
for i in range(epoch):
    model.train()  # bn层和dropout层会受影响
    print("--------第{}轮训练--------".format(i+1))
    train_correct_number=0
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model.forward(imgs)

        correct_temp=(outputs.argmax(1)==labels).sum()
        train_correct_number += correct_temp
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播算梯度
        optimizer.step()#更新梯度
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()#在每个epoch中每训练100次完成一次计时
            print("训练次数{},损失函数：{}".format(total_train_step, loss.item()))#每训练200次后当前的损失函数
            print(end_time-start_time)
    print("训练集的正确率: {}".format(train_correct_number/train_data_size))
    count_epoch+=1
    #学习率衰减
    if count_epoch % 10 == 0:
        for params in optimizer.param_groups:
            params['lr'] *= 0.6

    writer.add_scalar("train_loss_1",loss.item(),count_epoch)#每一轮记录一次当前损失函数
    writer.add_scalar("accuracy_train",train_correct_number/train_data_size,count_epoch)

    #训练完一轮验证？测试一次

    model.eval()
    valid_correct_number = 0
    with torch.no_grad():
        for data in valid_loader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            outputs = model.forward(imgs)
            correct_temp=(outputs.argmax(1) == labels).sum()#正确数
            valid_correct_number= valid_correct_number + correct_temp
            loss = loss_fn(outputs, labels)
    writer.add_scalar("valid_loss_1", loss.item(), count_epoch)
    valid_accuracy = valid_correct_number/valid_data_size
    writer.add_scalar("accuracy_valid",valid_correct_number/valid_data_size,count_epoch)
    print("验证集的正确率: {}".format(valid_correct_number / valid_data_size))
    if valid_accuracy> best_accuracy:
        best_accuracy = valid_accuracy
        #torch.save(state,"Tensorboard_log/vgg16Channel_Normalize_ViT_batch=16_lr=0.001_Adam&weight_decay=1e-4-Hybrid_VGG16_ViT.pth")
        save_checkpoint(count_epoch,model,optimizer,best_accuracy,os.path.join(train_path,'state.pth'))
        print('最优验证正确率为{},当前模型参数已保存'.format(best_accuracy))
    # scheduler.step()
writer.close()
