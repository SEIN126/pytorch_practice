import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data
import torchvision.models as models
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# 컨텐츠 손실을 맞출 layer number
content_layer_num = 1
image_size = 512
epoch = 5000

content_dir = "./images/content/Tuebingen_Neckarfront.jpg"
style_dir = "./images/style/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

# image 전처리
# pretrain 된 resnet은 이미지넷으로 학습된 모델 => 이 양식에 맞게 정규화해줘야함
def image_preprocess(img_dir):
    img = Image.open(img_dir)
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                         std=[1,1,1]) # 이미지 수치 범위를 mean, std 범위로 조정
                                    ])
    img = transform(img).view((-1,3,image_size,image_size))
    return img

# torch.tranpose(p,q,r)
# p=> 대상 img
# q ,r => 서로 바꿀 dimension
'''
ex
output = [100, 12, 28, 28]인 경우
torch.transpose(output,1,3)
결과 : [100, 28, 28, 12]
'''

# image 후처리
# 정규화 된 상태로 연산을 진행하고 다시 이미지화 => 뺐던 값을 다시 더해줌
# 이미지가 0~1 사이의 값을 가지게 해줌
def image_postprocess(tensor):
    transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                     std=[1,1,1])
    img = transform(tensor.clone())
    img = img.clamp(0,1)
    img = torch.transpose(img,0,1)
    img = torch.transpose(img,1,2)
    return img

#Resnet-50
resnet = models.resnet50(pretrained=True)
for name, module in resnet.named_children():
    print(name)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self, x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

# gram matrix
class GramMatix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        return G

# 그람 행렬간의 손실을 계산하는 클래스 및 함수
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatix()(input),target)
        return out

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    resnet = Resnet().to(device)
    for param in resnet.parameters():
        param.requires_grad = False # pre train 된 resnet의 weight는 학습 x

    content = image_preprocess(content_dir).to(device)
    style = image_preprocess(style_dir).to(device)
    generated = content.clone().requires_grad_().to(device)

    print(content.requires_grad, style.requires_grad,generated.requires_grad)
    '''
    # 시각화
    plt.imshow(image_postprocess(content[0].cpu()))
    plt.show()

    plt.imshow(image_postprocess(style[0].cpu()))
    plt.show()

    gen_img = image_postprocess(generated[0].cpu()).data.numpy()
    plt.imshow(gen_img)
    plt.show()
    '''

    # 목표값 설정, 행렬의 크기에 따른 가중치 정의 => ??
    style_target = list(GramMatix().to(device)(i) for i in resnet(style))
    content_target = resnet(content)[content_layer_num]
    style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

    # LBFGS 최적화 함수 사용
    # 학습의 대상 => 모델의 가중치가 아닌 이미지 자체
    optimizer = optim.LBFGS([generated])
    iteration = [0]

    while iteration[0] < epoch:
        def closure():
            optimizer.zero_grad()
            out = resnet(generated)

            # 스타일 손실을 각각의 목표값에 따라 계산, 리스트로 저장
            style_loss = [GramMSELoss().to(device)(out[i],style_target[i])*style_weight[i] for i in range(len(style_target))]
            # 컨텐츠 손실은 지정한 위치에서만 저장 -> 하나의 값으로
            content_loss = nn.MSELoss().to(device)(out[content_layer_num],content_target)
            # 스타일 : 컨텐츠 = 1000 : 1 으로 총 손실 계산
            total_loss = 1000*sum(style_loss) + torch.sum(content_loss)
            total_loss.backward()

            if iteration[0] %100 == 0:
                print(total_loss)
            iteration[0] +=1
            return total_loss

        optimizer.step(closure) # 터짐

    gen_img = image_postprocess(generated[0].cpu()).data.numpy()

    plt.figure(figsize=(10,10))
    plt.imshow(gen_img)
    plt.show()