import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 1
learning_rate = 0.0002
num_epoch = 100

# # ImageFolder라는 함수를 이용해 따로 이미지-라벨 쌍을 만들지 않고 폴더에 저장하는것만으로 쉽게 이미지-라벨 쌍을 만들 수 있다.
# # root dir
# img_dir = "./images"
#
# # 해당 루트 디렉토리를 ImageFolder 함수에 전달합니다.
# # 이때 이미지들에 대한 변형도 같이 전달해줍니다.
# img_data = dset.ImageFolder(img_dir, transforms.Compose([transforms.Resize(256),
#                                                          transforms.RandomResizedCrop(224), # 256x256 이미지의 랜덤한 위치에서 224x224 크기만큼 샘플링 합니다.
#                                                          transforms.RandomHorizontalFlip(), # 랜덤한 확률로 이미지를 좌우반전 합니다.
#                                                          transforms.ToTensor()]))

train_loader = DataLoader(img_data, batch_size = batch_size, shuffle=True, num_workers=2) # num_workers : data를 읽어올때 사용할 subprocess의 갯수
'''
하이퍼-파라미터를 튜닝하는 것처럼 결국 모델에 가장 적합한 num_workers 수치를 찾아내는 것도 파라미터 튜닝으로 볼 수 있습니다. num_workers 튜닝을 위해 고려해야 하는 것은 학습 환경의 GPU개수, CPU개수, I/O 속도, 메모리 등이 있습니다. I/O를 포함시킨 것은 데이터의 종류에 따라 디스크상에 존재하는 데이터를 로드하는것은 I/O에 상당히 많은 영향을 주고받을 수 있기 때문이고, 메모리는 loading된 데이터를 메모리상에 들고 있어야 하는 부담 때문에 포함되겠습니다.
'''

def conv_2_block(in_dim, out_dim): # convolution이 두번 연속하는경우
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim): # convolution이 세번 연속하는경우
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(VGG,self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim),
            conv_2_block(base_dim,2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim * 7 * 7,100),
            nn.ReLU(True),
            nn.Linear(100,20),
            nn.ReLU(True),
            nn.Linear(20,num_classes)
        )

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layer(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)

    model = VGG(base_dim=16).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for i in model.named_children():
        print(i)
    # print('_____________________________________')
    # for i in model.named_buffers():
    #     print(i)
    # print('_____________________________________')
    # for i in model.named_modules():
    #     print(i)
    # print('_____________________________________')
    # for i in model.named_parameters(): # model parameter들의 값 확인 가능
    #     print(i)
    # print('_____________________________________')

    for i in range(num_epoch):
        for j, [image,label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

        if i%10==0:
            print(loss)

    