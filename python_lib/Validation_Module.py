import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt

def LoadMatData(mat_file,device):
    for i in mat_file:
            print(i)

    Pos_Y = mat_file["Pos_Y"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Pos_Y size :",len(Pos_Y), "X", len(Pos_Y[0]))
    print("Pos_Y size of mat_file_value:",Pos_Y.shape)
    print("Pos_Y type of mat_file_value:",type(Pos_Y))

    Vel_Y = mat_file["Vel_Y"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Vel_Y size :",len(Vel_Y), "X", len(Vel_Y[0]))
    print("Vel_Y size of mat_file_value:",Vel_Y.shape)
    print("Vel_Y type of mat_file_value:",type(Vel_Y))

    Yaw = mat_file["Yaw"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Yaw size :",len(Yaw), "X", len(Yaw[0]))
    print("Yaw size of mat_file_value:",Yaw.shape)
    print("Yaw type of mat_file_value:",type(Yaw))

    Yaw_Rate = mat_file["Yaw_Rate"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Yaw_Rate size :",len(Yaw_Rate), "X", len(Yaw_Rate[0]))
    print("Yaw_Rate size of mat_file_value:",Yaw_Rate.shape)
    print("Yaw_Rate type of mat_file_value:",type(Yaw_Rate))

    Vel_X = mat_file["Vel_X"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Vel_X size :",len(Vel_X), "X", len(Vel_X[0]))
    print("Vel_X size of mat_file_value:",Vel_X.shape)
    print("Vel_X type of mat_file_value:",type(Vel_X))

    Acc_Y = mat_file["Acc_Y"]     # gt1의 state와 gt2의 state를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("Acc_Y size :",len(Acc_Y), "X", len(Acc_Y[0]))
    print("Acc_Y size of mat_file_value:",Acc_Y.shape)
    print("Acc_Y type of mat_file_value:",type(Acc_Y))

    WSA_value = mat_file["WSA_rad"]   # gt1의 WSA와 gt2의 WSA를 각각 100의 배수로 맞추고 합친것, 마지막 time의 WSA는 없음
    print("WSA size :",len(WSA_value), "X", len(WSA_value[0]))
    print("WSA size of mat_file_value:",WSA_value.shape)
    print("WSA type of mat_file_value:",type(WSA_value))

    Time = mat_file["time"]    # gt1의 time와 gt2의 time를 각각 100의 배수로 맞추고 합친것, 마지막 time의 state는 없음
    print("SWA size :",len(Time), "X", len(Time[0]))
    print("SWA size of mat_file_value:",Time.shape)
    print("SWA type of mat_file_value:",type(Time))

    Input = np.concatenate((Pos_Y, Vel_Y, Yaw, Yaw_Rate, WSA_value), axis=1)
    State = np.concatenate((Pos_Y, Vel_Y, Yaw, Yaw_Rate), axis=1)
    
    print('Input.shape : ',Input.shape)
    print('State.shape : ',State.shape)

    Input_tensor = torch.Tensor(Input)
    State_tensor = torch.Tensor(State)
    Vel_X_tensor = torch.Tensor(Vel_X)
    Acc_Y_tensor = torch.Tensor(Acc_Y)
    print('type(Input_tensor) : ',type(Input_tensor))
    print('type(State_tensor) : ',type(State_tensor))
    print('type(Vel_X_tensor) : ',type(Vel_X_tensor))
    print('type(Acc_Y_tensor) : ',type(Acc_Y_tensor))
    print('Input_tensor.shape : ',Input_tensor.shape)
    print('State_tensor.shape : ',State_tensor.shape)
    print('Vel_X_tensor.shape : ',Vel_X_tensor.shape)
    print('Acc_Y_tensor.shape : ',Acc_Y_tensor.shape)

    Input_tensor_gpu = Input_tensor.to(device)
    State_tensor_gpu = State_tensor.to(device)
    Vel_X_tensor_gpu = Vel_X_tensor.to(device)
    Acc_Y_tensor_gpu = Acc_Y_tensor.to(device)

    return Input_tensor_gpu, State_tensor_gpu, Vel_X_tensor_gpu, Acc_Y_tensor_gpu, Time

def DataLoaderPlot(model,NetworkInput_Loader,time):
  with torch.no_grad():
    X_pred = []

    for data in NetworkInput_Loader:
      input, state = data # 배치 데이터.
      dx = model(input)   # 모델에 넣고,
      x_next = dx*0.01+state
      X_pred += x_next .cpu().numpy().tolist()


  print('type(X_pred) : ',type(X_pred))
  X_pred = np.array(X_pred)
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

def plotting(model, Input_tensor_gpu, State_tensor_gpu, Vel_X_tensor_gpu, Acc_Y_tensor_gpu, Time):
  with torch.no_grad():
    dx_pred = []

    for input in Input_tensor_gpu:
      print('input.shape: ',input.shape)
      dx = model(input)   # 모델에 넣고,
      print('dx.shape: ',dx.shape)
      dx_pred += dx.cpu().numpy().tolist()
      #print('dx_pred: ',dx_pred,sep='\n')

  print('type(dx_pred) : ',type(dx_pred))
  dx = np.array(dx_pred)
  print('dx.shape : ',dx.shape)
  print('state.shape : ',State_tensor_gpu.shape)
  final_pred = dx*0.01 + State_tensor_gpu[:,:]
  final_pred = final_pred[:44599,:]     # gt1
  ground_truth = State_tensor_gpu[1:44600,:]       # gt1
  final_pred = final_pred[:44599,:]     # gt2
  ground_truth = State_tensor_gpu[1:44600,:]       # gt2
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

  # #ground truth와 prediction value와의 동등성 검사(다르게 나와야 정상)
  # isequal_gt_MLP_state1 = np.all(ground_truth[:,0] == final_pred[:,0])
  # isequal_gt_MLP_state2 = np.all(ground_truth[:,1] == final_pred[:,1])
  # isequal_gt_MLP_state3 = np.all(ground_truth[:,2] == final_pred[:,2])
  # isequal_gt_MLP_state4 = np.all(ground_truth[:,3] == final_pred[:,3])
  # print('isequal_gt_MLP_state1 : ',isequal_gt_MLP_state1)
  # print('isequal_gt_MLP_state2 : ',isequal_gt_MLP_state2)
  # print('isequal_gt_MLP_state3 : ',isequal_gt_MLP_state3)
  # print('isequal_gt_MLP_state4 : ',isequal_gt_MLP_state4)

  # # RMSE 구하기
  # rmse_gt_MLP_state1 = np.sqrt(np.mean((ground_truth[:,0] - final_pred[:,0]) ** 2))
  # rmse_gt_MLP_state2 = np.sqrt(np.mean((ground_truth[:,1] - final_pred[:,1]) ** 2))
  # rmse_gt_MLP_state3 = np.sqrt(np.mean((ground_truth[:,2] - final_pred[:,2]) ** 2))
  # rmse_gt_MLP_state4 = np.sqrt(np.mean((ground_truth[:,3] - final_pred[:,3]) ** 2))
  # print('rmse_gt_MLP_state1 : ',rmse_gt_MLP_state1)
  # print('rmse_gt_MLP_state2 : ',rmse_gt_MLP_state2)
  # print('rmse_gt_MLP_state2 : ',rmse_gt_MLP_state3)
  # print('rmse_gt_MLP_state3 : ',rmse_gt_MLP_state4)

  # # 표준편차 구하기
  # std_gt_MLP_state1 = np.std(ground_truth[:,0] - final_pred[:,0])
  # std_gt_MLP_state2 = np.std(ground_truth[:,1] - final_pred[:,1])
  # std_gt_MLP_state3 = np.std(ground_truth[:,2] - final_pred[:,2])
  # std_gt_MLP_state4 = np.std(ground_truth[:,3] - final_pred[:,3])
  # print('std_gt_MLP_state1 : ',std_gt_MLP_state1)
  # print('std_gt_MLP_state2 : ',std_gt_MLP_state2)
  # print('std_gt_MLP_state3 : ',std_gt_MLP_state3)
  # print('std_gt_MLP_state4 : ',std_gt_MLP_state4)