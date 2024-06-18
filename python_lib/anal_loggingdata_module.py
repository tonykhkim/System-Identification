import numpy as np
import matplotlib.pylab as plt
import scipy.io

def logging_info(mat_file_name):

    mat_file = scipy.io.loadmat(mat_file_name)
    print(type(mat_file))

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

    # plt.figure(0)
    # plt.title("Lateral Velocity")
    # plt.xlabel("time")
    # plt.plot(idx,State_value[:,0])

    # plt.figure(1)
    # plt.title("Yaw Rate")
    # plt.plot(idx,State_value[:,1])

    # plt.figure(3)
    # plt.title("Wheel Steering Angle")
    # plt.plot(idx,WSA_value[:,:])

    # plt.figure(4)
    # plt.title("Steering Wheel Angle")
    # plt.plot(idx,SWA_value[:,:])
    # plt.show()

    Vy = mat_file["y_dot"]
    YawRate = mat_file["yaw_dot"]

    return State_value, WSA_value, SWA_value

def sliding_windows(state1, state2,input,seq_length):
    # 시계열 데이터이므로 몇 개의 타임 스텝을 이용하여 에측할 것인가를 정해야 함
    # 이를 위해 sliding_windows함수를 만들어 슬라이딩 윈도우 데이터셋을 생성함.

    X1_history = []    # 첫번째 state에 대한 state history
    X2_history = []    # 두번째 state에 대한 state history
    #dX1_history = []   # 첫번째 state에 대한 dX1 (derivative)
    #dX2_history = []   # 두번째 state에 대한 dX2 (derivative)
    U_history = []     # input U에 대한 input history
    X1_dot = []        # 예측 state 값 (첫번째 ground Truth 값)
    X2_dot = []        # 예측 state 값 (두번째 ground Truth 값)

    for i in range(len(state1[:,0])-seq_length):
        _X1_his = state1[i:i+seq_length,0]
        _X2_his = state2[i:i+seq_length,0]
        _U_his = input[i:i+seq_length,0]
        _X1_dot = state1[i+seq_length,0]
        _X2_dot = state2[i+seq_length,0]

        X1_history.append(_X1_his)
        X2_history.append(_X2_his)
        U_history.append(_U_his)
        X1_dot.append(_X1_dot)
        X2_dot.append(_X2_dot)

    X1_history = np.array(X1_history)
    X2_history = np.array(X2_history)
    U_history = np.array(U_history)
    dX1_history = np.diff(state1,axis=0)   # 첫번째 state에 대한 dX1 (derivative)
    dX2_history = np.diff(state2,axis=0)   # 두번째 state에 대한 dX2 (derivative)
    X1_dot = np.array(X1_dot)
    X2_dot = np.array(X2_dot)
    
    X1_dot = X1_dot.reshape(len(X1_dot),1)
    X2_dot = X2_dot.reshape(len(X2_dot),1)
    

    print('type(X1_history) :',type(X1_history))
    print('type(X2_history) :',type(X2_history))
    print('type(U_history) :',type(U_history))
    print('type(X1_dot) :',type(X1_dot))
    print('type(X2_dot) :',type(X2_dot))
    print('type(dX1_history) :',type(dX1_history))
    print('type(dX2_history) :',type(dX2_history))

    print('X1_history.shape :',X1_history.shape)
    print('X2_history.shape :',X2_history.shape)
    print('U_history.shape :',U_history.shape)
    print('X1_dot.shape :',X1_dot.shape)
    print('X2_dot.shape :',X2_dot.shape)
    print('dX1_history.shape :',dX1_history.shape)
    print('dX2_history.shape :',dX2_history.shape)
    print('state1.shape: ',state1.shape)
    print('state2.shape: ',state2.shape)


    return X1_history, X2_history, U_history, X1_dot, X2_dot, dX1_history, dX2_history