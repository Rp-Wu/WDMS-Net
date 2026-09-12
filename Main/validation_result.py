import time
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from Model.WDMS_Net import WDMS_Net
from checkout_saveload import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1087, 0.2388, 0.7831], std=[0.2581, 0.3698, 0.2378]),  # 这是18949原始十人数据集的均值和方差
    transforms.Resize([299,299]),
])
valid_data = torchvision.datasets.ImageFolder(root="3_pig_cropped_18949/val",transform=transform_valid)
valid_data_size=len(valid_data)
valid_loader =DataLoader(valid_data,batch_size=16, shuffle=False,num_workers=0)

model = WDMS_Net(num_classes=10)
optimizer=torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9,weight_decay=1e-2)
load_checkpoint(model,optimizer=optimizer,path="E:\AccResult\FinalModel_ForPTH_21.699MB_lr1e3_wd1e2\state.pth")
model.cuda()

correct_pred = {classname: 0 for classname in valid_data.classes}
total_pred = {classname: 0 for classname in valid_data.classes}
correct = 0
total = 0
iter=0
model.eval()
with torch.no_grad():
    time_list = []
    for data in valid_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        ts = time.perf_counter()#计时开始
        outputs = model(images)
        td = time.perf_counter()#计时结束
        time_list.append(td - ts)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[valid_data.classes[label]] += 1
            total_pred[valid_data.classes[label]] += 1
    print(f"avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}")
    print(f"full time: {sum(time_list[5:]):.5f}")

acc=100*float(correct)/total
print(f'Accuracy of the network on the valid images: {acc:.3f} %')
# print accuracy for each class

print(f'每个类别预测分类正确个数（‘类别：个数’键值对）：{correct_pred}')
print(f'每个类别总数数（‘类别：个数’键值对）：{total_pred}')
# for i in cls:
#     correct_count=correct_pred[i]
#     accuracy = 100 * float(correct_count) / total_pred[i]
#     print(f'Accuracy of the network on the person{i}: {accuracy:.3f} %')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.3f} %')