import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
from plot_RNN import plotting 
from data_seq import sliding_windows
from VehMathModelCalculation import VehMathModelCal, fVehParameter, con2dis_seconddynamics

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)


mat_file = scipy.io.loadmat('TrainingData_seconddynamics.mat')

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

time = mat_file["time"]

X = np.concatenate((State_value, WSA_value), axis=1)
y = np.diff(State_value,axis=0)

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


# 학습 파라미터
num_epochs = 10000
learning_rate = 0.001
input_size = 3    # input_size (Vy, YawRate, delta)
#hidden_size = 64   # hidden state의 개수
hidden_size = 32   # hidden state의 개수
num_layers = 1

model = VanillaRNN(input_size = input_size,
                  hidden_size = hidden_size,
                  sequence_length = sequence_length,
                  num_layers = num_layers,
                  device = device).to(device)


#model.load_state_dict(torch.load('second_model_seconddynamics_inputWSA_hiddensize64.pth'))
model.load_state_dict(torch.load('second_model_seconddynamics_inputWSA_hiddensize32.pt'))

mathmodelpred_file = scipy.io.loadmat('VehMathModelPrediction.mat')

for i in mathmodelpred_file:
        print(i)

MathModelpred = mathmodelpred_file["x2_k"]
print('MathModelpred.shape : ',MathModelpred.shape)

# vehicleParams = fVehParameter()
# Ad, Bd = con2dis_seconddynamics(vehicleParams)
# C_bar, A_bar = VehMathModelCal(Ad,Bd,State_value[:,0],WSA_value)
# xG = C_bar*WSA_value + A_bar*State_value[:,0]
# print('xG.shape : ',xG.shape)

# # xG의 길이의 절반만큼의 열을 가지고, 2개의 행을 가진 행렬을 초기화합니다.
# mathmodelpred = np.zeros((2, len(xG) // 2))

# # xG를 순회하며 필요한 원소들을 x2_k에 할당합니다.
# for ii in range(1, len(xG) + 1):  # Python은 0부터 인덱싱을 시작하지만, MATLAB 코드의 로직을 따르기 위해 1부터 시작합니다.
#     if ii % 2 == 0:
#         mathmodelpred[:, ii // 2 - 1] = xG[ii-2:ii]  # Python은 0부터 인덱싱을 시작하므로 ii-2:ii를 사용합니다.

# print('mathmodelpred.shape : ',mathmodelpred.shape)

plotting(train_loader, model, State_value, MathModelpred, time)