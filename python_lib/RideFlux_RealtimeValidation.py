import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import scipy.io
import matplotlib.pyplot as plt
from Validation_Module import plotting , LoadMatData, DataLoaderPlot
import MLPClass
from scipy.stats import norm

print(torch.__version__)
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)

mat_file = scipy.io.loadmat('gt1_forPython.mat')
# mat_file = scipy.io.loadmat('gt2_forPython.mat')
# mat_file = scipy.io.loadmat('gt3_forPython.mat')
# mat_file = scipy.io.loadmat('gt4_forPython.mat')

Input_tensor_gpu, State_tensor_gpu, Vel_X_tensor_gpu, Acc_Y_tensor_gpu, Time = LoadMatData(mat_file,device)
model_high = MLPClass.MultiLayerPerceptron(device).to(device)  # 고속주회로 네트워크 모델 객체
model_low = MLPClass.MultiLayerPerceptron(device).to(device)   # K-city 네트워크 모델 객체

#optimizer = optim.Adam(model.parameters(), lr = learning_rate)

NetworkInput = torch.utils.data.TensorDataset(Input_tensor_gpu, State_tensor_gpu)
batch_size = 1
NetworkInput_loader = torch.utils.data.DataLoader(dataset=NetworkInput,batch_size=batch_size,shuffle=False)

model_high.load_state_dict(torch.load('rideflux_gt12_fourthdynamics_32168_batch100_onlystatedict.pt'))    # 고속주회로 모델 네트워크 파라미터
model_low.load_state_dict(torch.load('rideflux_gt34_fourthdynamics_32168_batch100_onlystatedict.pt'))     # K-city 모델 네트워크 파라미터

#model.load_state_dict(checkpoint['model_state_dict'])

#plotting(model, Input_tensor_gpu, State_tensor_gpu, Vel_X_tensor_gpu, Acc_Y_tensor_gpu, Time)
#DataLoaderPlot(model,NetworkInput_loader,Time)

with torch.no_grad():
    high_pred_list = []
    low_pred_list = []
    state_list = []

    for index, data in enumerate(NetworkInput_loader):
      input, state = data # data는 input과 state를 원소로 갖는 리스트
      high_dx = model_high(input)   
      low_dx = model_low(input) 
      high_pred = high_dx*0.01+state   # torch.Size([1, 4])   # <class 'torch.Tensor'>
      low_pred = low_dx*0.01+state

      high_pred_list += high_pred .cpu().numpy().tolist()     # + 연산자로 high_pred_list 리스트에 원소를 리스트로 추가
      low_pred_list += low_pred .cpu().numpy().tolist()
      state_list += state.cpu().numpy().tolist()

      if index >= 499:
        # numpy array로 변환
        high_pred_array = np.array(high_pred_list)
        low_pred_array = np.array(low_pred_list)
        state_array = np.array(state_list)
            
        # (index-499)번째부터 index번째까지의 값들을 슬라이싱
        high_pred_subarray = high_pred_array[(index-499):index+1]  #Python에서 리스트나 배열의 슬라이싱을 할 때, 끝 인덱스는 포함되지 않기 때문에 index+1을 한것임.
        low_pred_subarray = low_pred_array[(index-499):index+1]
        state_subarray = state_array[(index-499):index+1]
            
        # error 계산
        high_error = high_pred_subarray - state_subarray   # (500, 4)  # <class 'numpy.ndarray'>
        low_error = low_pred_subarray - state_subarray

        # error의 평균(μ) 계산
        high_mean = [np.mean(high_error[:, i]) for i in range(4)]  # <class 'list'>  ex) [1,2,3,4]
        low_mean = [np.mean(low_error[:, i]) for i in range(4)] 

        # error의 표준편차(σ) 계산
        high_std = [np.std(high_error[:, i]) for i in range(4)]  # <class 'list'>  ex) [1,2,3,4]
        low_std = [np.std(low_error[:, i]) for i in range(4)]  

        # error의 정규분포 객체 생성
        high_norm = [norm(loc=high_mean[i],scale=high_std[i]) for i in range(4)]  # loc=평균, scale=표준편차
        low_norm = [norm(loc=low_mean[i],scale=low_std[i]) for i in range(4)]

        # error의 확률밀도함수의 확률 변수 범위 지정
        high_x = [np.linspace(high_mean[i]-3*high_std[i],high_mean[i]+3*high_std[i],1000) for i in range(4)]   # 리스트 high_x의 각각의 원소는 numpy.ndarray 형식
        low_x = [np.linspace(low_mean[i]-3*low_std[i],low_mean[i]+3*low_std[i],1000) for i in range(4)]
        
        # error의 확률밀도 함수의 확률밀도 값
        high_y = [high_norm[i].pdf(high_x[i]) for i in range(4)]   # 리스트 high_y의 각각의 원소는 numpy.ndarray 형식
        low_y = [low_norm[i].pdf(low_x[i]) for i in range(4)]

        # print('high_x : ',high_x)
        print('len(high_y) : ',len(high_y))
        print('type(high_y[1]) : ',type(high_y[1]))
        # high_y = high_norm[0].pdf(high_x)
        # print('high_y : ',high_y)