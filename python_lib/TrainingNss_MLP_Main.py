import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
from torch.nn import init
from plot_MLP import plotting 

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)


mat_file = scipy.io.loadmat('ridefluxdata_gt34_fourthdynamics.mat')


# 파이토치로 딥러닝 모델을 구현하는 데는 nn.Module 클래스를 상속받아 구현한다. 
# 파이토치를 이용하여 nn.Module 클래스를 상속받고 딥러닝을 구현하기 위해서는 __init__ 메소드와 forward 메소드를 메소드 오버라이딩(method overriding, 메소드 재정의) 해야한다.

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
print("state_diff size :",len(State_diff), "X", len(State_diff[0]))
print("state_diff size of mat_file_value:",State_diff.shape)
print("state_diff type of mat_file_value:",type(State_diff))

Time = mat_file["time"]    # gt1의 time와 gt2의 time를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
print("SWA size :",len(Time), "X", len(Time[0]))
print("SWA size of mat_file_value:",Time.shape)
print("SWA type of mat_file_value:",type(Time))

idx = []
for i in range(0, len(State_value)):
    idx = np.append(idx, i)

print('x축:',len(idx))
print('y축:',len(State_value))

print('type(State_value) : ',type(State_value))
print('State_value.shape : ',State_value.shape)

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

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)

# 학습 파라미터
num_epochs = 10000
learning_rate = 0.001

model = MultiLayerPerceptron(device = device).to(device)



# regression 문제이기 때문에 loss function을 RMSE로 설정
# 이를 위해 우선 nn.MSELoss()를 이용
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    input, target = data # 배치 데이터.
    dx = model(input)   # 모델에 넣고,
    #h_n = h_n[-1, :, :]
    #print('h_n.size() : ',h_n.size())
    #print('target.size() : ',target.size())
    loss = torch.sqrt(criterion(dx, target)) # nn.MSELoss() 값의 제곱근을 구함으로써 RMSE를 구하고, 이 RMSE를 loss함수로 설정

    optimizer.zero_grad() # gradient를 0으로 안만들어주면 기존 gradient 연산 결과가 축적됨
    loss.backward() # loss가 최소가 되게하는
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()

model_path1 = 'rideflux_gt34_fourthdynamics_32168_batch100_onlystatedict.pt'
torch.save(model.state_dict(), model_path1)

model_path2 = 'rideflux_gt34_fourthdynamics_32168_batch100_allcheckpoint.pt'
torch.save({
    'epoch':epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dixt': optimizer.state_dict(),
    'loss' : loss,
    }, model_path2)
#print('State_value.shape : ',State_value.shape)

#plotting(train_loader, model, State_value, time)