# Autoencoder - 데이터에 대한 효율적인 압축을 신경망을 통해 자동으로 학습하는 모델

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 256
learning_rate = 0.0002
num_epoch = 5

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=False, num_workers=2, drop_last=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Linear(28*28,20) # data를 길이 20짜리 벡터로 압축
        self.decoder = nn.Linear(20,28*28) # 원래 크기로 돌려놓음

    def forward(self,x):
        x = x.view(batch_size,-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size,1,28,28)
        return out

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Autoencoder().to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

## Train ##
    loss_arr=[]
    for i in range(num_epoch):
        for j,[image,label] in enumerate(train_loader):
            x = image.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,x)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print(loss)
                loss_arr.append(loss.cpu().data.numpy()) # array 형태로 loss_arr에 저장
                # loss_arr.append(loss) #tensor 형태로 저장 : [tensor(0.1377, device='cuda:0', grad_fn=<MseLossBackward>), tensor(...]
    print(loss_arr)
    print(output.size()) # [256, 1, 28, 28]
    print(output.cpu().size()) # [256, 1, 28, 28]
    out_img = torch.squeeze(output.cpu().data) # squeeze를 왜해주는지는 모르겠음 = > [256, 1, 28, 28]에서 1 제거
    '''
    # torch.squeeze : 특정 차원의 크기가 1 일때, 그 치원을 제거 
    # ex)[3,1] => [3,], [256, 1, 28, 28] => [256, 28, 28]
    '''
    print(out_img.size()) # [256, 28, 28]

## Check with Train Image ##
    for i in range(10):
        plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
        plt.show()
        plt.imshow(out_img[i].numpy(),cmap='gray')
        plt.show()

## Test and Check with Test Image ##
    with torch.no_grad():
        for i in range(1):
            for j,[image,label] in enumerate(test_loader):
                x = image.to(device)

                optimizer.zero_grad()
                output = model.forward(x)

                if j % 1000 == 0:
                    print(loss)

    out_img = torch.squeeze(output.cpu().data)
    print(out_img.size())

    for i in range(10):
        plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
        plt.show()
        plt.imshow(out_img[i].numpy(),cmap='gray')
        plt.show()