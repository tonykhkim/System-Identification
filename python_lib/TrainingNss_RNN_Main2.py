import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
from plot import plotting 
from data_seq import sliding_windows

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)


mat_file = scipy.io.loadmat('TrainingData_seconddynamics.mat')

class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        #self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)  # 초기 hidden state 설정
        out, h_n = self.rnn(x, h0)  # out : 모든 타임스텝에 대한 결과. hn: t=n인 마지막 타임스텝에 대한 hidden state를 반환
        #out = out.reshape(out.shape[0], -1)  # many to many 전략
        #out = self.fc(out)
        #return out
        h_n = self.fc(h_n.squeeze(0))  # 마지막 hidden state를 선형 레이어에 전달
        return out, h_n


for i in mat_file:
        print(i)

State_value = mat_file["State"]
print("State size :",len(State_value), "X", len(State_value[0]))
print("State size of mat_file_value:",State_value.shape)
print("State type of mat_file_value:",type(State_value))

WSA_value = mat_file["WSA_input"]
print("WSA size :",len(WSA_value), "X", len(WSA_value[0]))
print("WSA size of mat_file_value:",WSA_value.shape)
print("WSA type of mat_file_value:",type(WSA_value))

SWA_value = mat_file["SWA_input"]
print("SWA size :",len(SWA_value), "X", len(SWA_value[0]))
print("SWA size of mat_file_value:",SWA_value.shape)
print("SWA type of mat_file_value:",type(SWA_value))

idx = []
for i in range(0, len(State_value)):
    idx = np.append(idx, i)

print('x축:',len(idx))
print('y축:',len(State_value))

print('type(State_value) : ',type(State_value))
print('State_value.shape : ',State_value.shape)

X = np.concatenate((State_value, WSA_value), axis=1)
#X = np.concatenate((State_value, SWA_value), axis=1)
y = np.diff(State_value,axis=0)
print('X.shape : ',X.shape)
print('y.shape : ',y.shape)
print('y[0:10,:] : ',y[0:10,:])

sequence_length = 10

x_seq= sliding_windows(X,sequence_length)
print('type(x_seq) : ',type(x_seq))
x_seq = np.array(x_seq)
print('type(x_seq) : ',type(x_seq))
print('x_seq.shape : ',x_seq.shape)
#print('y_seq.shape : ',y_seq.shape)
y_seq = y[9:,:]
print('y_seq.shape : ',y_seq.shape)

X_seq = torch.Tensor(x_seq)
Y_seq = torch.Tensor(y_seq)
print('type(X_seq) : ',type(X_seq))
print('type(Y_seq) : ',type(Y_seq))
print('X_seq.shape : ',X_seq.shape)
print('Y_seq.shape : ',Y_seq.shape)

X_seq_gpu = X_seq.to(device)
Y_seq_gpu = Y_seq.to(device)

train = torch.utils.data.TensorDataset(X_seq_gpu, Y_seq_gpu)

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)

# 학습 파라미터
num_epochs = 10000
learning_rate = 0.001
input_size = 3    # input_size (Vy, YawRate, delta)
hidden_size = 32   # hidden state의 개수
num_layers = 1

model = VanillaRNN(input_size = input_size,
                  hidden_size = hidden_size,
                  sequence_length = sequence_length,
                  num_layers = num_layers,
                  device = device).to(device)



# regression 문제이기 때문에 loss function을 RMSE로 설정
# 이를 위해 우선 nn.MSELoss()를 이용
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    seq, target = data # 배치 데이터.
    out,h_n = model(seq)   # 모델에 넣고,
    #h_n = h_n[-1, :, :]
    #print('h_n.size() : ',h_n.size())
    #print('target.size() : ',target.size())
    loss = torch.sqrt(criterion(h_n, target)) # nn.MSELoss() 값의 제곱근을 구함으로써 RMSE를 구하고, 이 RMSE를 loss함수로 설정

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

model_path = 'second_model_seconddynamics_inputWSA_hiddensize32.pt'
torch.save(model.state_dict(), model_path)

#print('State_value.shape : ',State_value.shape)

#time = mat_file["time"]

#plotting(train_loader, model, State_value, time)