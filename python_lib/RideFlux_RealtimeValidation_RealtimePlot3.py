import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Validation_Module import plotting , LoadMatData, DataLoaderPlot
from scipy.stats import norm
import MLPClass

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

NetworkInput = torch.utils.data.TensorDataset(Input_tensor_gpu, State_tensor_gpu)
batch_size = 1
NetworkInput_loader = torch.utils.data.DataLoader(dataset=NetworkInput,batch_size=batch_size,shuffle=False)

model_high.load_state_dict(torch.load('rideflux_gt12_fourthdynamics_32168_batch100_onlystatedict.pt'))    # 고속주회로 모델 네트워크 파라미터
model_low.load_state_dict(torch.load('rideflux_gt34_fourthdynamics_32168_batch100_onlystatedict.pt'))     # K-city 모델 네트워크 파라미터

plt.ion()

with torch.no_grad():
    high_pred_list = []
    low_pred_list = []
    state_list = []

    # 2x2 subplot grid (axs) 내의 각 subplot에 대해 두 개의 선 객체(line1과 line2)를 초기화
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    lines = []

    for i in range(4):
      line1, = axs[i // 2, i % 2].plot([0], [0], 'r-', label='High')  # 빈 데이터([])를 가진 초기 line객체를 생성
      line2, = axs[i // 2, i % 2].plot([0], [0], 'b-', label='Low')
      lines.append((line1, line2))
      axs[i // 2, i % 2].legend()

    for ax in axs.flat:
      ax.set_xlim(-10, 10)
      ax.set_ylim(0, 2)

    for index, data in enumerate(NetworkInput_loader):
      input, state = data
      high_dx = model_high(input)
      low_dx = model_low(input)
      high_pred = high_dx * 0.01 + state
      low_pred = low_dx * 0.01 + state

      high_pred_list.extend(high_pred.cpu().numpy().tolist())
      low_pred_list.extend(low_pred.cpu().numpy().tolist())
      state_list.extend(state.cpu().numpy().tolist())

      if len(high_pred_list) > 500:
        high_pred_list.pop(0)
        low_pred_list.pop(0)
        state_list.pop(0)

      if index >= 499:
        high_pred_array = np.array(high_pred_list)
        low_pred_array = np.array(low_pred_list)
        state_array = np.array(state_list)

        high_pred_subarray = high_pred_array[-500:]
        low_pred_subarray = low_pred_array[-500:]
        state_subarray = state_array[-500:]

        high_error = high_pred_subarray - state_subarray
        low_error = low_pred_subarray - state_subarray

        high_mean = [np.mean(high_error[:, i]) for i in range(4)]
        low_mean = [np.mean(low_error[:, i]) for i in range(4)]

        high_std = [np.std(high_error[:, i]) for i in range(4)]
        low_std = [np.std(low_error[:, i]) for i in range(4)]

        high_norm = [norm(loc=high_mean[i], scale=high_std[i]) for i in range(4)]
        low_norm = [norm(loc=low_mean[i], scale=low_std[i]) for i in range(4)]

        high_x = [np.linspace(high_mean[i] - 3 * high_std[i], high_mean[i] + 3 * high_std[i], 1000) for i in range(4)]
        low_x = [np.linspace(low_mean[i] - 3 * low_std[i], low_mean[i] + 3 * low_std[i], 1000) for i in range(4)]

        high_y = [high_norm[i].pdf(high_x[i]) for i in range(4)]
        low_y = [low_norm[i].pdf(low_x[i]) for i in range(4)]

        print('ok')

        for i in range(4):
          lines[i][0].set_data(high_x[i], high_y[i])
          lines[i][1].set_data(low_x[i], low_y[i])
          axs[i // 2, i % 2].relim()  # 데이터 범위 재계산
          axs[i // 2, i % 2].autoscale_view()  # 축 범위 자동 설정
          fig.canvas.draw_idle()
          #fig.canvas.flush_events()
          plt.pause(0.1)

plt.show()  # 최종 플롯을 보여줌