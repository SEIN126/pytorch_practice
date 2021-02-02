import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os

try:
    os.mkdir("images")
    os.mkdir("images/dogs")
    os.mkdir("images/cats")
except:
    pass


batch_size = 2
learning_rate = 0.001
num_epoch = 10
num_category = 2

img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))

# print(img_data.classes)
# print(img_data.class_to_idx)
# print(img_data.imgs)

train_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True,
                          num_workers=2, drop_last=True)
'''
for img, label in train_loader:
    print(img.size())
    print(label)
'''

resnet = models.resnet50(pretrained=True)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:-1]) # conv1 ~ avgpool (fc만 제외) => feature extractor
        self.layer1 = nn.Sequential(
            nn.Linear(2048,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,num_category),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer0(x)
        out = out.view(batch_size,-1)
        out = self.layer1(out)
        return out


if __name__ == '__main__':

    for img, label in train_loader:
        print(img.size())
        print(label)

    for name, module in resnet.named_children():
        print(name)
        '''
        conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool
        fc
        '''
    #     print(module)
    #     print('----------')
    #
    # print('==================================')
    # for i in resnet.named_children(): # ex) ('fc', Linear(in_features=2048, out_features=1000, bias=True))
    #     print(i)
    #     print("+++++++++++++++++++++++++++++++++++++")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Resnet().to(device)

    for params in model.layer0.parameters():
        params.requires_grad = False

    for params in model.layer1.parameters():
        params.requires_grad = True

    # for m in model.children():
    #     print(m)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    #Train
    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

            # if i%10==0:
            print(f'i = {i}')
            print(loss)


    #Test
    model.eval() # batch normalization 끄기
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in train_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()

        print("Accuracy of Train Data: {}".format(100 * correct / total))
        # 왜 50이 나올까.. 100%가 나와야하는데...
        # 75나 25도 몇번씩 나온다. 사실 data set도 부족하고 ( 각 class마다 두개), 그냥 이런식으로 학습이 진행된다는 것만 보여주려는듯 하다.
        # 실제로 보면 gpu는 두개가 작동하는 것 같다. 쓰레드로 작동하는건가? 잘 모르겠다. 왜 두번씩이나 되는거지?
