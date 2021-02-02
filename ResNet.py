import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

## ResNet - 50

batch_size= 1
learning_rate = 0.0002
num_epoch = 100

img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
            ]))

train_loader = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

def conv_block_1(in_dim,out_dim,act_fn,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=stride),
        act_fn
    )
    return model

def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
        act_fn
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, act_fn, down=False):
        super(BottleNeck,self).__init__()
        self.act_fn=act_fn
        self.down = down # resolution down 여부, boolean

        if self.down: # feature map의 size가 감소하는 경우(stride =2)
            self.layer = nn.Sequential(
                conv_block_1(in_dim,mid_dim,act_fn,2),
                conv_block_3(mid_dim,mid_dim,act_fn),
                conv_block_1(mid_dim,out_dim,act_fn)
            )
            self.downsample = nn.Conv2d(in_dim,out_dim,1,2)

        else: # feature map의 size가 그대로인 경우
            self.layer = nn.Sequential(
                conv_block_1(in_dim,mid_dim,act_fn),
                conv_block_3(mid_dim,mid_dim,act_fn),
                conv_block_1(mid_dim,out_dim,act_fn)
            )
        # 더하기(+)를 위해 차원을 맞춰줌
        self.dim_equalizer = nn.Conv2d(in_dim,out_dim,kernel_size=1)

    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else :
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet(nn.Module):

    def __init__(self, base_dim, num_classes=2):
        super(ResNet, self).__init__()
        self.act_fn=nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn,down=True),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn, down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn, down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
        )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)

        return out

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ResNet(base_dim=64).to(device)

    for i in model.children():
        print(i)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for i in range(num_epoch):
        for j,[image, label] in enumerate(train_loader):
            x = image.to(device)
            y_= label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

        if i %10 ==0:
            print(loss)
