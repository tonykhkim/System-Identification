import numpy as np
import matplotlib.pylab as plt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from anal_loggingdata_module import logging_info, sliding_windows

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)

mat_file_name = "TrainingData_seconddynamics.mat"

Vy, YawRate, WSA_value, SWA_value = logging_info(mat_file_name)




#print('type of State_value :',type(State_value))
print('type of WSA_value :',type(WSA_value))
print('type of SWA_value :',type(SWA_value))

#Network_input = np.hstack((State_value,SWA_value))   # 두 배열을 가로로 결합

seq_length = 10
print('shape of Vy : ',Vy.shape)
print('shape of YawRate : ',YawRate.shape)
print('type of Vy: ', type(Vy))
print('type of YawRate: ',type(YawRate))
X1_hist, X2_hist, U_hist, X1_dot, X2_dot,dX1_history, dX2_history = sliding_windows(Vy, YawRate,SWA_value,seq_length)
# print('X1_hist[0,:] : ',X1_hist[0,:])
# print('X1_hist[1,:] : ', X1_hist[1,:])

# GPU를 이용하여 수치 연산을 가속화 하기 위해서 numpy 배열을 파이토치의 텐서로 변환
# torch.Tensor 클래스의 생성자 함수를 이용하여 텐서 생성
# torch.Tensor 클래스는 정수형 데이터 타입이더라도 생성된 텐서의 데이터 타입은 실수형으로 변환한다.

# numpy의 배열로부터 텐서 생성
X1_seq = torch.Tensor(X1_hist)   # 현재 state1
X1_next = torch.Tensor(X1_dot)   # 다음 state1
U_seq = torch.Tensor(U_hist)       # 현재 SWA
X2_seq = torch.Tensor(X2_hist)   # 현재 state2
X2_next = torch.Tensor(X2_dot)   # 다음 state2
dX1_seq = torch.Tensor(dX1_history)   # derivative state1
dX2_seq = torch.Tensor(dX2_history)   # derivative state2

print('type(X1_seq) : ',type(X1_seq))
print('type(X2_seq) : ',type(X2_seq))
print('type(U_seq) : ',type(U_seq))
print('type(X1_next) : ',type(X1_next))
print('type(X2_next) : ',type(X2_next))
print('type(dX1_seq) : ',type(dX1_seq))
print('type(dX2_seq) : ',type(dX2_seq))

print('X1_seq.shape : ',X1_seq.shape)
print('X2_seq.shape : ',X2_seq.shape)
print('U_seq.shape : ',U_seq.shape)
print('X1_next.shape : ',X1_next.shape)
print('X2_next.shape : ',X2_next.shape)
print('dX1_seq.shape : ',dX1_seq.shape)
print('dX2_seq.shape : ',dX2_seq.shape)

# 해당 텐서에 GPU 설정
X1_seq_gpu = X1_seq.to(device)    
X2_seq_gpu = X2_seq.to(device)    
U_seq_gpu = U_seq.to(device)    
X1_next_gpu = X1_next.to(device)
X2_next_gpu = X2_next.to(device)
U_seq_gpu = U_seq.to(device)
dX1_seq_gpu = dX1_seq.to(device)
dX2_seq_gpu = dX2_seq.to(device)

# 파이토치에서 지원하는 TensorDataset과 DataLoader 기능을 이용함으로써
# 미니배치 학습, 데이터 셔플, 병렬처리와 같은 데이터 처리를 간단하게 수행 가능
train = torch.utils.data.TensorDataset(X1_seq_gpu,X1_next_gpu)
batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

# 학습 파라미터
num_epochs = 10000
learning_rate = 0.001
input_size = 3    # input_size를 3으로 할 수 있음?
hidden_size = 32   # hidden state의 크기가 무엇을 의미?
num_layers = 1
                  # mini-batch 학습 하려면 어떻게 해야함?

# RNN 모델 구축
class VanillaRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)  # 초기 hidden state 설정
        out, _ = self.rnn(x, h0)  # out : RNN의 마지막 레이어로부터 나온 output feature를 반환. hn: hidden state를 반환
        out = out.reshape(out.shape[0], -1)  # many to many 전략
        out = self.fc(out)
        return out
    
model = VanillaRNN(input_size = input_size,
                  hidden_size = hidden_size,
                  sequence_length = seq_length,
                  num_layers = num_layers,
                  device = device).to(device)

# regression 문제이기 때문에 loss function을 MSE로 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

# # RNN 모델 학습
# loss_graph = [] # 그래프 그릴 목적인 loss.
# n = len(train_loader)

# for epoch in range(num_epochs):
#   running_loss = 0.0

#   for data in train_loader:

#     seq, target = data # 배치 데이터.
#     out = model(seq)   # 모델에 넣고,
#     loss = criterion(out, target) # output 가지고 loss 구하고,

#     optimizer.zero_grad() # 
#     loss.backward() # loss가 최소가 되게하는 
#     optimizer.step() # 가중치 업데이트 해주고,
#     running_loss += loss.item() # 한 배치의 loss 더해주고,

#   loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
#   if epoch % 100 == 0:
#     print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

# # loss graph 
# plt.figure(figsize=(20,10))
# plt.plot(loss_graph)
# plt.show()

# # validation
# # 실제값과 train_loader, test_loader를 넣었을때 예측값들을 모두 죽 뽑음
# # 실제값 vs train+test 예측값을 모두 확인.
# def plotting(train_loader, test_loader, actual):
#   with torch.no_grad():
#     train_pred = []
#     test_pred = []

#     for data in train_loader:
#       seq, target = data
#       out = model(seq)
#       train_pred += out.cpu().numpy().tolist()

#     for data in test_loader:
#       seq, target = data
#       out = model(seq)
#       test_pred += out.cpu().numpy().tolist()
      
#   total = train_pred + test_pred
#   plt.figure(figsize=(20,10))
#   plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
#   plt.plot(actual, '--')
#   plt.plot(total, 'b', linewidth=0.6)

#   plt.legend(['train boundary', 'actual', 'prediction'])
#   plt.show()

# plotting(train_loader, test_loader, df['Close'][sequence_length:])