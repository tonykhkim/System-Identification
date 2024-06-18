import numpy as np

def VehMathModelCal(Ad, Bd, x0, uG):
    nx = Ad.shape[1]  # number of state
    nu = Bd.shape[1]  # number of input
    ns = uG.shape[0]  # number of step

    C_bar = np.zeros((ns*nx, ns*nu))
    A_bar = np.zeros((ns*nx, nx))
    
    # A_bar
    for i in range(1, ns+1):
        A_bar[(i-1)*nx:i*nx, :] = np.linalg.matrix_power(Ad, i)
    
    # C_bar
    for i in range(1, ns+2):      # 행
        for j in range(1, ns+1):  # 열
            if i > j:
                C_bar[(i-1)*nx:i*nx, (j-1)*nu:j*nu] = np.linalg.matrix_power(Ad, i-j-1) @ Bd
            else:
                C_bar[(i-1)*nx:i*nx, (j-1)*nu:j*nu] = np.zeros((nx, nu))
    
    C_bar = C_bar[nx:, :]
    
    return C_bar, A_bar

def fVehParameter():
    # CarSim's Vehicle(C-Class, Hatchback) Parameter
    params = {
        'm': 1274,
        'Iz': 1523,
        'Caf': 96415,
        'Car': 80417,
        'lf': 1.016,
        'lr': 1.562,
        'Ts': 0.01,  # Controller Cycle Time
        'Vx': 13.8889  # 50km/h = 13.8889 m/s
    }
    
    return params

def con2dis_seconddynamics(param):
    # load param
    Caf = param['Caf']
    Car = param['Car']
    m = param['m']
    lf = param['lf']
    lr = param['lr']
    Iz = param['Iz']
    Vx = param['Vx']

    Ac = np.array([[-1 * (2 * Caf + 2 * Car) / (m * Vx), -Vx - 1 * (2 * Caf * lf - 2 * Car * lr) / (m * Vx)],
                   [-1 * (2 * Caf * lf - 2 * Car * lr) / (Iz * Vx), -1 * (2 * Caf * (lf**2) + 2 * Car * (lr**2)) / (Iz * Vx)]])

    Bc = np.array([[2 * Caf / m],
                   [2 * Caf * lf / Iz]])

    Cc = np.array([[1, 0],
                   [0, 1]])

    # Discretization by Euler Method
    n = Ac.shape[1]

    Ts = 0.01

    Ad = np.eye(n) + Ac * Ts
    Bd = Bc * Ts

    return Ad, Bd