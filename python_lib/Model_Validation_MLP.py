import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import scipy.io
import matplotlib.pyplot as plt
from plot_MLP import plotting 

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)


mat_file = scipy.io.loadmat('ridefluxdata_gt12_fourthdynamics.mat')

for i in mat_file:
        print(i)

State_value = mat_file["state"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
print("State size :",len(State_value), "X", len(State_value[0]))
print("State size of mat_file_value:",State_value.shape)
print("State type of mat_file_value:",type(State_value))

WSA_value = mat_file["WSA_rad"]   # gt1의 WSA와 gt2의 WSA를 각각 100의 배수로 맞추고 합친것, 마지막 time의 WSA는 없음
print("WSA size :",len(WSA_value), "X", len(WSA_value[0]))
print("WSA size of mat_file_value:",WSA_value.shape)
print("WSA type of mat_file_value:",type(WSA_value))

State_diff = mat_file["state_diff"]  # gt1의 state_diff와 gt2의 state_diff를 각각 100의 배수로 맞추고 합친것
print("State_diff size :",len(State_diff), "X", len(State_diff[0]))
print("State_diff size of mat_file_value:",State_diff.shape)
print("State_diff type of mat_file_value:",type(State_diff))

Time = mat_file["time"]    # gt1의 time와 gt2의 time를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
print("SWA size :",len(Time), "X", len(Time[0]))
print("SWA size of mat_file_value:",Time.shape)
print("SWA type of mat_file_value:",type(Time))


X = np.concatenate((State_value, WSA_value), axis=1)
#X = np.concatenate((State_value, SWA_value), axis=1)
# y = np.diff(State_value,axis=0)
Target = State_diff
print('X.shape : ',X.shape)
print('Target.shape : ',Target.shape)
print('Target[0:10,:] : ',Target[0:10,:])


X_tensor = torch.Tensor(X)
Target_tensor = torch.Tensor(Target)
print('type(X_tensor) : ',type(X_tensor))
print('type(Target_tensor) : ',type(Target_tensor))
print('X_tensor.shape : ',X_tensor.shape)
print('Target_tensor.shape : ',Target_tensor.shape)

X_tensor_gpu = X_tensor.to(device)
Target_tensor_gpu = Target_tensor.to(device)

train = torch.utils.data.TensorDataset(X_tensor_gpu, Target_tensor_gpu)

batch_size = 1

train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)

class MultiLayerPerceptron(nn.Module):

    def __init__(self,device):
        # __init__ 메소드: 모델의 레이어들을 초기화한다.
        # 즉, 모델의 레이어의 구성 요소들을 정의한다.
        # __init__ 메소드는 사용자 정의 클래스의 객체가 생성될 때 자동으로 호출된다. 
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(5, 32, bias=True)
        self.fc2 = nn.Linear(32, 16, bias=True)
        self.fc3 = nn.Linear(16, 8, bias=True)
        self.output_layer = nn.Linear(8, 4, bias=True)

        # 가중치와 바이어스 초기화
        self._init_weights()

    def _init_weights(self):
        # Glorot 초기화 적용
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.output_layer.weight)

        # 바이어스를 0으로 초기화
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # forward 메소드 : 모델에서 실행되어야 하는 계산을 정의한다.
        # 즉, 데이터를 입력받아서 어떤 계산을 진행하여야 출력이 될 것인가를 정의해준다.
        # 사용자 정의 클래스의 객체 생성 후 호출하면 자동으로 실행된다. 
        x = torch.tanh(self.fc1(x))   #Liear 계산 후 활성화 함수 tanh 적용
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.output_layer(x)
        return x


model = MultiLayerPerceptron(device).to(device)

#optimizer = optim.Adam(model.parameters(), lr = learning_rate)


model.load_state_dict(torch.load('rideflux_gt12_fourthdynamics_32168_batch100_onlystatedict.pt'))
#model.load_state_dict(checkpoint['model_state_dict'])

plotting(train_loader, model, State_value, Time)