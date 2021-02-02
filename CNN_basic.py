import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # DataLoader는 전처리가 끝난 데이터들을 지정한 배치 크기에 맞게 모아서 전달해주는 역할을 합니다.

batch_size = 256
learning_rate = 0.0002
num_epoch = 10

# data를 불러옴
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(),  # train=True : 학습데이터
                         target_transform=None, download=True)  # target_transform : 라벨에 대한 변형
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(),  # train=False : 테스트데이터
                        target_transform=None, download=True)

# data를 batch size대로 묶거나 정렬, 섞어주는 역할을 하는 모듈(DataLoader)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)


# drop_last : batch_size대로 묶고 남은 데이터는 버릴것인지

class CNN(nn.Module):  # nn.Module 상속
    def __init__(self):
        super(CNN, self).__init__()  # CNN클래스의 부모 클래스인 nn.Module을 초기화하는 역할
        self.layer = nn.Sequential(
            # nn.Conv2d(1, 16, 5),  # in_channels, out_channels, kernel_size => rgb가 아닌 흑백이기 때문에 in_channels는 1
            # output = (batch_size, out_channels, 가로, 세로)
            nn.Conv2d(1,16,3,padding=1), # 28 x 28
            nn.BatchNorm2d(16), # batch_normalization
            nn.ReLU(),
            # nn.Conv2d(16, 32, 5),
            nn.Conv2d(16,32,3,padding=1), # 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14 x 14
            # nn.Conv2d(32, 64, 5),
            nn.Conv2d(32,64,3,padding=1), # 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 7 x 7
        )
        self.fc_layer = nn.Sequential(
            # nn.Linear(64 * 3 * 3, 100),
            nn.Linear(64 * 7 * 7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                # Init with small numbers
                m.weight.data.normal_(0.0,0.02)
                m.bias.data.fill_(0)
                
                # Xavier Initialization
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0) 
                '''
                # Kaiming Initialization
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)  # reshape인듯
        out = self.fc_layer(out)
        return out




if __name__ == '__main__': #CNN class를 재귀적으로 생성하지 않기 위해서 필요
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device) # 모델을 지정한 장치로 옮김
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 매 epoch마다 학습률 조정
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    loss_arr = []
    for i in range(num_epoch):
        # 매 epoch마다 step호출
        if i != 0: # optimizer.step() 전에 호출되면 경고 뜸
            scheduler.step()
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
        print(i, scheduler.get_lr())

    param_list = list(model.parameters())
    print(param_list)
    # Test
    correct = 0
    total = 0

    # 배치정규화나 드롭아웃은 학습할때와 테스트 할때 다르게 동작하기 때문에 모델을 evaluation 모드로 바꿔서 테스트해야합니다.
    model.eval()
    # 인퍼런스 모드를 위해 no_grad 해줍니다.
    with torch.no_grad(): # 기울기를 개선하지 않고 진행
        # 테스트로더에서 이미지와 정답을 불러옵니다.
        for image, label in test_loader:
            # 두 데이터 모두 장치에 올립니다.
            x = image.to(device)
            y_ = label.to(device)

            # 모델에 데이터를 넣고 결과값을 얻습니다.
            output = model.forward(x)

            # https://pytorch.org/docs/stable/torch.html?highlight=max#torch.max
            # torch.max를 이용해 최대 값 및 최대값 인덱스를 뽑아냅니다.
            # 여기서는 최대값은 필요없기 때문에 인덱스만 사용합니다.
            _, output_index = torch.max(output, 1)

            # 전체 개수는 라벨의 개수로 더해줍니다.
            # 전체 개수를 알고 있음에도 이렇게 하는 이유는 batch_size, drop_last의 영향으로 몇몇 데이터가 잘릴수도 있기 때문입니다.
            total += label.size(0) # size(0)에는 batch size가 들어감

            # 모델의 결과의 최대값 인덱스와 라벨이 일치하는 개수를 correct에 더해줍니다.
            correct += (output_index == y_).sum().float()

        # 테스트 데이터 전체에 대해 위의 작업을 시행한 후 정확도를 구해줍니다.
        print("Accuracy of Test Data: {}%".format(100 * correct / total))