import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 500

x = init.uniform_(torch.Tensor(num_data,1),-10,10)
noise = init.normal_(torch.FloatTensor(num_data,1),std=1)
y = 2*x + 3
y_noise = 2*(x+noise)+3

model = nn.Linear(1,1) ## input_feature, output feature 수
loss_func = nn.L1Loss()

optimizer = optim.SGD(model.parameters(),lr = 0.01) ##model.parameter()로 변수 w,b를 전달

label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad() ## 지난번에 계산했던 기울기를 0으로 초기화
    output = model(x)

    loss = loss_func(output,label)
    loss.backward() ## w,b에 대한 기울기 계산
    optimizer.step() ## model.parameters()에서 리턴되는 변수들의 기울기에 lr을 곱해서 step 진행

    if i%10 == 0:
        print(loss.data)

param_list = list(model.parameters())
print(param_list[0].item(),param_list[1].item())
## 1.9939701557159424 2.592924118041992