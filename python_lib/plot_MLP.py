import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotting(train_loader, model, state, time):
  with torch.no_grad():
    train_pred = []
    dx_pred = []

    for data in train_loader:
      input, target = data # 배치 데이터.
      dx = model(input)   # 모델에 넣고,
      dx_pred += dx .cpu().numpy().tolist()

  print('type(dx_pred) : ',type(dx_pred))
  dx = np.array(dx_pred)
  print('dx.shape : ',dx.shape)
  print('state.shape : ',state.shape)
  final_pred = dx*0.01 + state[:,:]
  final_pred = final_pred[:44599,:]     # gt1
  ground_truth = state[1:44600,:]       # gt1
  final_pred = final_pred[:44599,:]     # gt2
  ground_truth = state[1:44600,:]       # gt2
  time = time[1:44600,:]
  print('final_pred.shape : ',final_pred.shape)
  print('ground_truth.shape : ',ground_truth.shape)
  print('time.shape : ',time.shape)

  file_path = 'MLP_pred.mat'  # 저장할 파일 이름
  scipy.io.savemat(file_path, {'MLP_pred': final_pred})

  plt.figure(figsize=(10,5))
  #plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
  plt.plot(time[:,0],ground_truth[:,0], 'r-')
  plt.plot(time[:,0],final_pred[:,0], 'b--')
  plt.legend(['Ground Truth', 'MLP'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Lateral Position', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  plt.ylabel('y [m]', fontsize=15)  # y축 레이블 설정

  plt.figure(figsize=(10,5))
  #plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
  plt.plot(time[:,0],ground_truth[:,1], 'r-')
  plt.plot(time[:,0],final_pred[:,1], 'b--')
  plt.legend(['Ground Truth', 'MLP'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Lateral Velocity', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  plt.ylabel('dy/dt [m/s]', fontsize=15)  # y축 레이블 설정

  plt.figure(figsize=(10,5))
  plt.plot(time[:,0],ground_truth[:,2], 'r-')
  plt.plot(time[:,0],final_pred[:,2], 'b--')
  plt.legend(['Ground Truth', 'MLP'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Yaw', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  #plt.ylabel(r'$\frac{d\psi}{dt}$ [rad/s]', fontsize=15)  # y축 레이블 설정
  plt.ylabel(r'$\psi$ [rad]', fontsize=15)  # y축 레이블 설정

  plt.figure(figsize=(10,5))
  plt.plot(time[:,0],ground_truth[:,3], 'r-')
  plt.plot(time[:,0],final_pred[:,3], 'b--')
  plt.legend(['Ground Truth', 'MLP'], fontsize=15)
  plt.grid(True)  # 그리드 추가
  plt.title('Yaw Rate', fontsize=15)  # 제목 설정
  plt.xlabel('time', fontsize=15)  # x축 레이블 설정
  #plt.ylabel(r'$\frac{d\psi}{dt}$ [rad/s]', fontsize=15)  # y축 레이블 설정
  plt.ylabel(r'$d\psi$/dt [rad/s]', fontsize=15)  # y축 레이블 설정

  plt.show()

  #ground truth와 prediction value와의 동등성 검사(다르게 나와야 정상)
  isequal_gt_MLP_state1 = np.all(ground_truth[:,0] == final_pred[:,0])
  isequal_gt_MLP_state2 = np.all(ground_truth[:,1] == final_pred[:,1])
  isequal_gt_MLP_state3 = np.all(ground_truth[:,2] == final_pred[:,2])
  isequal_gt_MLP_state4 = np.all(ground_truth[:,3] == final_pred[:,3])
  print('isequal_gt_MLP_state1 : ',isequal_gt_MLP_state1)
  print('isequal_gt_MLP_state2 : ',isequal_gt_MLP_state2)
  print('isequal_gt_MLP_state3 : ',isequal_gt_MLP_state3)
  print('isequal_gt_MLP_state4 : ',isequal_gt_MLP_state4)

  # RMSE 구하기
  rmse_gt_MLP_state1 = np.sqrt(np.mean((ground_truth[:,0] - final_pred[:,0]) ** 2))
  rmse_gt_MLP_state2 = np.sqrt(np.mean((ground_truth[:,1] - final_pred[:,1]) ** 2))
  rmse_gt_MLP_state3 = np.sqrt(np.mean((ground_truth[:,2] - final_pred[:,2]) ** 2))
  rmse_gt_MLP_state4 = np.sqrt(np.mean((ground_truth[:,3] - final_pred[:,3]) ** 2))
  print('rmse_gt_MLP_state1 : ',rmse_gt_MLP_state1)
  print('rmse_gt_MLP_state2 : ',rmse_gt_MLP_state2)
  print('rmse_gt_MLP_state2 : ',rmse_gt_MLP_state3)
  print('rmse_gt_MLP_state3 : ',rmse_gt_MLP_state4)

  # 표준편차 구하기
  std_gt_MLP_state1 = np.std(ground_truth[:,0] - final_pred[:,0])
  std_gt_MLP_state2 = np.std(ground_truth[:,1] - final_pred[:,1])
  std_gt_MLP_state3 = np.std(ground_truth[:,2] - final_pred[:,2])
  std_gt_MLP_state4 = np.std(ground_truth[:,3] - final_pred[:,3])
  print('std_gt_MLP_state1 : ',std_gt_MLP_state1)
  print('std_gt_MLP_state2 : ',std_gt_MLP_state2)
  print('std_gt_MLP_state3 : ',std_gt_MLP_state3)
  print('std_gt_MLP_state4 : ',std_gt_MLP_state4)