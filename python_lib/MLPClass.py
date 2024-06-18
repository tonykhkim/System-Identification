import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

class MultiLayerPerceptron(nn.Module):

    def __init__(self,device):
        # __init__ 메소드: 모델의 레이어들을 초기화한다.
        # 즉, 모델의 레이어의 구성 요소들을 정의한다.
        # __init__ 메소드는 사용자 정의 클래스의 객체가 생성될 때 자동으로 호출된다. 
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(5, 32, bias=True)
        self.fc2 = nn.Linear(32, 16, bias=True)
        self.fc3 = nn.Linear(16, 8, bias=True)
        self.output_layer = nn.Linear(8, 4, bias=True)

        # 가중치와 바이어스 초기화
        self._init_weights()

    def _init_weights(self):
        # Glorot 초기화 적용
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.output_layer.weight)

        # 바이어스를 0으로 초기화
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # forward 메소드 : 모델에서 실행되어야 하는 계산을 정의한다.
        # 즉, 데이터를 입력받아서 어떤 계산을 진행하여야 출력이 될 것인가를 정의해준다.
        # 사용자 정의 클래스의 객체 생성 후 호출하면 자동으로 실행된다. 
        x = torch.tanh(self.fc1(x))   #Liear 계산 후 활성화 함수 tanh 적용
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.output_layer(x)
        return x