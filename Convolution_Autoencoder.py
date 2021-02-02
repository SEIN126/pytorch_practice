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

#data
mnist_train = dset.MNIST("./", train=True,transform=transforms.ToTensor(),target_transform=None, download=True)
mnist_test = dset.MNIST("./",train=False,transform=transforms.ToTensor(),target_transform=None,download=True)

#set dataloader
train_loader = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=2, drop_last=True)

#Model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), # batch x 16 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,padding=1), # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1), # batch x 64 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2) # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2), # batch x 128 x 7 x 7
            nn.Conv2d(128,256,3,padding=1), # batch x 256 x 7 x 7
            nn.ReLU()
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1) # [batch, 256 x 7 x 7]
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1), # batch x 128 x 7 x 7, size가 두배됨
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1), # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64,16,3,1,1), # batch x 16 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16,1,3,2,1,1), # batch x 1 x 28 x 28
            nn.ReLU()
        )
    def forward(self,x):
        out = x.view(batch_size,256,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    # 인코더와 디코더의 parameter들을 동시에 학습시키기 위해 list로 묶음(원래 parameter들은 generator로 return됨)
    parameters = list(encoder.parameters())+list(decoder.parameters())

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(parameters, lr=learning_rate)

    # 모델 불러오기
    try:
        encoder, decoder = torch.load('./model/conv_autoencoder.pkl')
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device)

            output = encoder(image)
            output = decoder(output)

            loss = loss_func(output,image)
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                #모델 저장하는 방법
                # torch.save([encoder,decoder],'./model/conv_autoencoder.pkl')
                print(f"i = {i}, {j}th loss = {loss}")

    out_img = torch.squeeze(output.cpu().data)
    print(out_img.size())

    #Cheke with training set
    for i in range(5):
        plt.figure()
        ax1 = plt.subplot(1,2,1)
        ax1.imshow(torch.squeeze(image[i]).cpu().numpy(), cmap='gray')
        ax2 = plt.subplot(1,2,2)
        ax2.imshow(out_img[i].numpy(),cmap='gray')
        plt.show()

    #Test with test set
    with torch.no_grad():
        for j, [image,label] in enumerate(test_loader):
            image = image.to(device)

            output = encoder(image)
            output = decoder(output)

            if j % 10 == 0:
                print(f"{j}th loss = {loss}")

    out_img = torch.squeeze(output.cpu().data)
    print(out_img.size())

    for i in range(5):
        plt.figure()
        ax1 = plt.subplot(1,2,1)
        ax1.imshow(torch.squeeze(image[i]).cpu().numpy(), cmap='gray')
        ax2 = plt.subplot(1,2,2)
        ax2.imshow(out_img[i].numpy(),cmap='gray')
        plt.show()

