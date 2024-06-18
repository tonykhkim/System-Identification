import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt

def plotting(train_loader, model, state, mathmodelpred, time):
  with torch.no_grad():
    train_pred = []
    dx_pred = []

    for data in train_loader:
      seq, target = data
      out,h_n = model(seq)
      dx_pred += h_n.cpu().numpy().tolist()

  print('type(dx_pred) : ',type(dx_pred))
  dx = np.array(dx_pred)
  print('dx.shape : ',dx.shape)
  #print('dx_pred.shape : ',dx_pred.shape)
  print('state.shape : ',state.shape)
  final_pred = dx*0.01 + state[9:-1,:]
  ground_truth = state[10:,:]
  print('final_pred.shape : ',final_pred.shape)
  print('ground_truth.shape : ',ground_truth.shape)
  MathModelpred = mathmodelpred[9:,:]
  print('mathmodelpred.shape : ',mathmodelpred.shape)
  print('MathModelpred.shape : ',MathModelpred.shape)

  file_path = 'RNN_pred.mat'  # 저장할 파일 이름
  scipy.io.savemat(file_path, {'RNN_pred': final_pred})

  plt.figure(figsize=(10,5))
  #plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
  plt.plot(time[10:,0],ground_truth[:,0], 'r--')
  plt.plot(time[10:,0],final_pred[:,0], 'b--')
  plt.plot(time[10:,0],MathModelpred[:,0], 'g--')
  plt.legend(['Ground Truth', 'RNN','Math Model'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Lateral Velocity', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  plt.ylabel('dy/dt [m/s]', fontsize=15)  # y축 레이블 설정

  plt.figure(figsize=(10,5))
  plt.plot(time[10:,0],ground_truth[:,1], 'r--')
  plt.plot(time[10:,0],final_pred[:,1], 'b--')
  plt.plot(time[10:,0],MathModelpred[:,1], 'g--')
  plt.legend(['Ground Truth', 'RNN','Math Model'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Yaw Rate', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  #plt.ylabel(r'$\frac{d\psi}{dt}$ [rad/s]', fontsize=15)  # y축 레이블 설정
  plt.ylabel(r'$d\psi$/dt [rad/s]', fontsize=15)  # y축 레이블 설정

  plt.show()

  #ground truth와 prediction value와의 동등성 검사(다르게 나와야 정상)
  isequal_gt_RNN_state1 = np.all(ground_truth[:,0] == final_pred[:,0])
  isequal_gt_RNN_state2 = np.all(ground_truth[:,1] == final_pred[:,1])
  print('isequal_gt_RNN_state1 : ',isequal_gt_RNN_state1)
  print('isequal_gt_RNN_state2 : ',isequal_gt_RNN_state2)

  # RMSE 구하기
  rmse_gt_RNN_state1 = np.sqrt(np.mean((ground_truth[:,0] - final_pred[:,0]) ** 2))
  rmse_gt_RNN_state2 = np.sqrt(np.mean((ground_truth[:,1] - final_pred[:,1]) ** 2))
  rmse_gt_MathModel_state1 = np.sqrt(np.mean((ground_truth[:,0] - MathModelpred[:,0]) ** 2))
  rmse_gt_MathModel_state2 = np.sqrt(np.mean((ground_truth[:,1] - MathModelpred[:,1]) ** 2))
  print('rmse_gt_RNN_state1 : ',rmse_gt_RNN_state1)
  print('rmse_gt_RNN_state2 : ',rmse_gt_RNN_state2)
  print('rmse_gt_MathModel_state1 : ',rmse_gt_MathModel_state1)
  print('rmse_gt_MathModel_state2 : ',rmse_gt_MathModel_state2)

  # 표준편차 구하기
  std_gt_RNN_state1 = np.std(ground_truth[:,0] - final_pred[:,0])
  std_gt_RNN_state2 = np.std(ground_truth[:,1] - final_pred[:,1])
  std_gt_MathModel_state1 = np.std(ground_truth[:,0] - MathModelpred[:,0])
  std_gt_MathModel_state2 = np.std(ground_truth[:,1] - MathModelpred[:,1])
  print('std_gt_RNN_state1 : ',std_gt_RNN_state1)
  print('std_gt_RNN_state2 : ',std_gt_RNN_state2)
  print('std_gt_MathModel_state1 : ',std_gt_MathModel_state1)
  print('std_gt_MathModel_state2 : ',std_gt_MathModel_state2)