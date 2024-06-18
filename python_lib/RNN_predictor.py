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
hidden_size = 32   # hidden state의 개수
num_layers = 1
sequence_length = 10
batch_size = 1    # predictor는 RNN을 학습하는 용도가 아니기 때문에 batch_size = 1로 해야함.

model = VanillaRNN(input_size = input_size,
                  hidden_size = hidden_size,
                  sequence_length = sequence_length,
                  num_layers = num_layers,
                  device = device).to(device)


#model.load_state_dict(torch.load('second_model_seconddynamics_inputWSA_hiddensize64.pth'))
model.load_state_dict(torch.load('second_model_seconddynamics_inputWSA_hiddensize32.pth'))


mat_file = scipy.io.loadmat('TrainingData_seconddynamics.mat')

State_value = mat_file["State"]
WSA_value = mat_file["WSA_input"]
time = mat_file["time"]

X = np.concatenate((State_value, WSA_value), axis=1)
y = np.diff(State_value,axis=0)


x_seq= sliding_windows(X,sequence_length)
x_seq = np.array(x_seq)
y_seq = y[9:,:]

X_seq = torch.Tensor(x_seq)
Y_seq = torch.Tensor(y_seq)

print('X_seq.shape : ',X_seq.shape)   # torch.Size([16091, 10, 3])


X_seq_gpu = X_seq.to(device)
Y_seq_gpu = Y_seq.to(device)
print('X_seq_gpu.shape : ',X_seq_gpu.shape)   # torch.Size([16091, 10, 3])
print('type(X_seq_gpu) : ',type(X_seq_gpu))   # <class 'torch.Tensor'>

print('X_seq_gpu[0,:,:].shape : ',X_seq_gpu[0,:,:].shape)  # torch.Size([10, 3])
print('X_seq_gpu[0].shape : ',X_seq_gpu[0].shape)      # torch.Size([10, 3])

train = torch.utils.data.TensorDataset(X_seq_gpu, Y_seq_gpu)


train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=False)

with torch.no_grad():
    model_input = X_seq_gpu[0]      # 2D tensor
    model_input_3d = model_input.view(1,10,3)   #3D tensor

    out,h_n = model(model_input_3d)
    dx = h_n.cpu().numpy()
    print('model_input.shape : ',model_input.shape)   # torch.Size([10, 3])
    print('type(model_input) : ',type(model_input))   # <class 'torch.Tensor'>
    print('model_input_3d.shape : ',model_input_3d.shape)  # torch.Size([1, 10, 3])
    print('type(model_input_3d) : ',type(model_input_3d))  # <class 'torch.Tensor'>
    print('h_n.shape : ',h_n.shape)   # torch.Size([1, 2])
    print('type(h_n) : ',type(h_n))   # <class 'torch.Tensor'>
    print('dx.shape : ',dx.shape)   # (1, 2)
    print('type(dx) : ',type(dx))   # <class 'numpy.ndarray'>
    print('-------------------------')

# def plotting(train_loader, model, state, mathmodelpred, time):
#   with torch.no_grad():
#     train_pred = []
#     dx_pred = []

#     for data in train_loader:
#       seq, target = data
#       out,h_n = model(seq)
#       dx_pred += h_n.cpu().numpy().tolist()

  
#   dx = np.array(dx_pred)
#   final_pred = dx*0.01 + state[9:-1,:]

# with torch.no_grad():
#     dx_pred = []

#     for data in train_loader:
#       seq, target = data
#       out,h_n = model(seq)
#       dx = h_n.cpu().numpy()
#       print('seq.shape : ',seq.shape)
#       print('type(seq) : ',type(seq))
#       print('h_n.shape : ',h_n.shape)
#       print('type(h_n) : ',type(h_n))
#       print('dx.shape : ',dx.shape)
#       print('type(dx) : ',type(dx))
#       print('-------------------------')
##       dx_pred += h_n.cpu().numpy().tolist()
  
##     dx = np.array(dx_pred)
##     final_pred = dx*0.01 + state[9:-1,:]