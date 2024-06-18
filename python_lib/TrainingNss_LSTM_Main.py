import numpy as np
import matplotlib.pylab as plt
import scipy.io
import torch
import torch.nn as nn
from anal_loggingdata_module import logging_info

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)

mat_file_name = "TrainingData_seconddynamics.mat"

State_value, WSA_value, SWA_value = logging_info(mat_file_name)

print('type of State_value :',type(State_value))
print('type of WSA_value :',type(WSA_value))
print('type of SWA_value :',type(SWA_value))

Network_input = np.hstack((State_value,SWA_value))   # 두 배열을 가로로 결합

# GPU를 이용하여 수치 연산을 가속화 하기 위해서 numpy 배열을 파이토치의 텐서로 변환
# torch.Tensor 클래스의 생성자 함수를 이용하여 텐서 생성
# torch.Tensor 클래스는 정수형 데이터 타입이더라도 생성된 텐서의 데이터 타입은 실수형으로 변환한다.

# numpy의 배열로부터 텐서 생성
X = torch.Tensor(State_value[:-1,:])   # 현재 state
X_dot = torch.Tensor(State_value[1:,:])   # 다음 state
SWA = torch.Tensor(SWA_value[:-1,:])       # 현재 SWA
lstm_input = torch.Tensor(Network_input[:-1,:])

# 해당 텐서에 GPU 설정
X_gpu = X.to(device)    
X_dot_gpu = X_dot.to(device)
SWA_gpu = SWA.to(device)
Input_gpu = lstm_input.to(device)

# 학습 파라미터
num_epochs = 10000
learning_rate = 0.001

input_size = 1       # 입력에 대한 features의 수
hidden_size = 2      # 은닉상태의 features의 수
num_layers = 1       # LSTM을 스택킹(stacking)하는 수
batch_first = False  # 입출력 텐서의 현태가 다음과 같음(기본값은 False) : (seq, batch, input_size)
num_classes = 1       

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,device=device)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,device=device)
        
        # Propagate input through LSTM
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        h_n = h_n.view(-1, self.hidden_size)
        
        result = self.fc(h_n)
        
        return result
    
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm = lstm.to(device)  

loss_function = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(Input_gpu)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = loss_function(outputs, X_dot_gpu)
    
    loss.backward()
    
    optimizer.step()
    if (epoch+1)== 1:
        print("Epoch: %d, loss: %1.5f" % (epoch+1, loss.item()))
    if (epoch+1) % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch+1, loss.item()))

# predict
#rnn.eval()
Y_predict = lstm(Input_gpu)
Y_predict = Y_predict.data.cpu().numpy()
Y_data = Y_data.data.cpu().numpy()

inverse_scaled_Y_predict = scaler.inverse_transform(Y_predict)
inverse_scaled_Y_data = scaler.inverse_transform(Y_data)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(inverse_scaled_Y_data, label='original')
plt.plot(inverse_scaled_Y_predict,label='predicted' )
plt.suptitle('Time-Series Prediction')
plt.legend(loc='best')
plt.show()