%close all; clear; clc;

%load('C:\DMD_data.mat')
%load('C:\0707_DMD_model.mat')
%open data_generation.m
%gt1으로 학습한 모델에다가 gt2를 추가적으로 학습 
rng(0)

%각각 다른 초기 상태에서 시작하여 1초 동안 지속되는 1000개의 시뮬레이션을 실행한다. 
%각 실험은 동일한 시점을 사용해야 한다.

%Version1
Ts=0.001;

%%%%%%%%%%%%%%%%%%%%%%%%Generate Data Set for Validation%%%%%%%%%%%%%%%%%%
% input2_data(45801:end)=[];
% time2(45801:end)=[];
% state2_data(45801:end,:)=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U=cell(904,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
Y=cell(904,1);
time1_1 = time1(1:end-1);
time1_2 = time1(2:end);

x1 = state1_data(1:end-1,:);
x_dot1 = state1_data(2:end,:);

input1 = input1_data(1:end-1);
input_1 = cat(2,x1,input1);

x1_inverse = state1_data_inverse(1:end-1,:);
x_dot1_inverse = state1_data_inverse(2:end,:);

input1_inverse = input1_data_inverse(1:end-1);
input_1_inverse = cat(2,x1_inverse,input1_inverse);

for i=0:445
    
    U{i+1} = array2timetable(input_1(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_1(100*i+1:1:100*(i+1),1)));
    Y{i+1} = array2timetable(x_dot1(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_2(100*i+1:1:100*(i+1),1)));
    
end
% 
% for i=0:445
%     
%     U{i+447} = array2timetable(input_1_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_1(100*i+1:1:100*(i+1),1)));
%     Y{i+447} = array2timetable(x_dot1_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_2(100*i+1:1:100*(i+1),1)));
%     
% end
% % U{447} = array2timetable(input(1:1:44600,:),RowTimes=seconds(time1_1(1:1:44600,1)));
% % Y{447} = array2timetable(x_dot(1:1:44600,:),RowTimes=seconds(time1_2(1:1:44600,1)));

%%%%%%%%%%%%%%%%%%%%%gt6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(1620,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(1620,1);
time2_1 = time2(1:end-1);
time2_2 = time2(2:end);

x2 = state2_data(1:end-1,:);
x_dot2 = state2_data(2:end,:);

input2 = input2_data(1:end-1);
input_2 = cat(2,x2,input2);

x2_inverse = state2_data_inverse(1:end-1,:);
x_dot2_inverse = state2_data_inverse(2:end,:);

input2_inverse = input2_data_inverse(1:end-1);
input_2_inverse = cat(2,x2_inverse,input2_inverse);

for i=0:457
    
    U{i+447} = array2timetable(input_2(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_1(100*i+1:1:100*(i+1),1)));
    Y{i+447} = array2timetable(x_dot2(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_2(100*i+1:1:100*(i+1),1)));
    
end
% 
% for i=0:457
%     
%     U{i+1351} = array2timetable(input_2_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1351} = array2timetable(x_dot2_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_2(100*i+1:1:100*(i+1),1)));
%     
% end
% U{459} = array2timetable(input(1:1:45800,:),RowTimes=seconds(time2_1(1:1:45800,1)));
% Y{459} = array2timetable(x_dot(1:1:45800,:),RowTimes=seconds(time2_2(1:1:45800,1)));

%%%%%%%%%%%%%%%%%%%%gt7%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(647,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(647,1);
time3_1 = time3(1:end-1);
time3_2 = time3(2:end);

x3 = state3_data(1:end-1,:);
x_dot3 = state3_data(2:end,:);

input3 = input3_data(1:end-1);
input_3 = cat(2,x3,input3);

x3_inverse = state3_data_inverse(1:end-1,:);
x_dot3_inverse = state3_data_inverse(2:end,:);

input3_inverse = input3_data_inverse(1:end-1);
input_3_inverse = cat(2,x3_inverse,input3_inverse);

% for i=0:351
%     
%     U{i+1} = array2timetable(input_3(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1} = array2timetable(x_dot3(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:351
%     
%     U{i+353} = array2timetable(input_3_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot3_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_2(100*i+1:1:100*(i+1),1)));
%     
% end
% % U{353} = array2timetable(input(1:1:35200,:),RowTimes=seconds(time3_1(1:1:35200,1)));
% % Y{353} = array2timetable(x_dot(1:1:35200,:),RowTimes=seconds(time3_2(1:1:35200,1)));

%%%%%%%%%%%%%%%%%%%%gt8%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

x4_inverse = state4_data_inverse(1:end-1,:);
x_dot4_inverse = state4_data_inverse(2:end,:);

input4_inverse = input4_data_inverse(1:end-1);
input_4_inverse = cat(2,x4_inverse,input4_inverse);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:294
%     
%     U{i+1000} = array2timetable(input_4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1000} = array2timetable(x_dot4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end
% U{296} = array2timetable(input(23601:1:29542,:),RowTimes=seconds(time4_1(23601:1:29542,1)));
% Y{296} = array2timetable(x_dot(23601:1:29542,:),RowTimes=seconds(time4_1(23601:1:29542,1)));


%%%%%%%%%%%%%%%%%%%%gt9%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

x4_inverse = state4_data_inverse(1:end-1,:);
x_dot4_inverse = state4_data_inverse(2:end,:);

input4_inverse = input4_data_inverse(1:end-1);
input_4_inverse = cat(2,x4_inverse,input4_inverse);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:294
%     
%     U{i+1000} = array2timetable(input_4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1000} = array2timetable(x_dot4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt10%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

x4_inverse = state4_data_inverse(1:end-1,:);
x_dot4_inverse = state4_data_inverse(2:end,:);

input4_inverse = input4_data_inverse(1:end-1);
input_4_inverse = cat(2,x4_inverse,input4_inverse);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:294
%     
%     U{i+1000} = array2timetable(input_4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1000} = array2timetable(x_dot4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt11%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

x4_inverse = state4_data_inverse(1:end-1,:);
x_dot4_inverse = state4_data_inverse(2:end,:);

input4_inverse = input4_data_inverse(1:end-1);
input_4_inverse = cat(2,x4_inverse,input4_inverse);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:294
%     
%     U{i+1000} = array2timetable(input_4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1000} = array2timetable(x_dot4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt12%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

x4_inverse = state4_data_inverse(1:end-1,:);
x_dot4_inverse = state4_data_inverse(2:end,:);

input4_inverse = input4_data_inverse(1:end-1);
input_4_inverse = cat(2,x4_inverse,input4_inverse);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

% for i=0:294
%     
%     U{i+1000} = array2timetable(input_4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1000} = array2timetable(x_dot4_inverse(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

%Version2
% Ts=0.001;
% U=cell(81,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(81,1);
%  
% % %%%%%%%%%%%%%%%Generate Data Set for Validation%%%%%%%%%%%%%%%%%%
% for i=0:79
%    
%     U{i+1} = array2timetable(input1_data(447*i+1:1:447*(i+1),1),RowTimes=seconds(time1(447*i+1:1:447*(i+1),1)));
%     Y{i+1} = array2timetable(state1_data(447*i+1:1:447*(i+1),:),RowTimes=seconds(time1(447*i+1:1:447*(i+1),1)));
%     
% end
% 
% U{81} = array2timetable(input2_data(35761:1:44700,1),RowTimes=seconds(time2(35761:1:44700,1)));
% Y{81} = array2timetable(state2_data(35761:1:44700,:),RowTimes=seconds(time2(35761:1:44700,:)));

% %%Version3
% Ts=0.001;
% u1=input1_data(1:1:36660);
% t1=time1(1:1:36660);
% y1=state1_data(1:1:36660,:);
% u2=input1_data(36661:1:44700);
% t2=time1(36661:1:44700);
% y2=state1_data(36661:1:44700,:);
% U=cell(2,1);   
% Y=cell(2,1);
% U{1,1}=array2timetable(u1(:),RowTimes=seconds(t1(:)));
% Y{1,1}=array2timetable(y1(:,:),RowTimes=seconds(t1(:)));
% U{2,1}=array2timetable(u2(:),RowTimes=seconds(t2(:)));
% Y{2,1}=array2timetable(y2(:,:),RowTimes=seconds(t2(:)));

%%Version4
% Ts=0.001;
% U=cell(2,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(2,1);
% U{1,1} = array2timetable(input1_data(1:1:36660,1),RowTimes=seconds(time1(1:1:36660,1)));
% Y{1,1} = array2timetable(state1_data(1:1:36660,:),RowTimes=seconds(time1(1:1:36660,1)));
% U{2,1} = array2timetable(input1_data(36661:1:44700,1),RowTimes=seconds(time1(36661:1:44700,1)));
% Y{2,1} = array2timetable(state1_data(36661:1:44700,:),RowTimes=seconds(time1(36661:1:44700,1)));

% %%Version5
% Ts=0.001;
% U=cell(1,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(1,1);
% U{1} = array2timetable(input1_data(1:1:36660,1),RowTimes=seconds(time1(1:1:36660,1)));
% Y{1} = array2timetable(state1_data(1:1:36660,:),RowTimes=seconds(time1(1:1:36660,1)));

%%Version6
% Ts=0.001;
% 
% U = array2timetable(input1_data(1:1:36660,1),RowTimes=seconds(time1(1:1:36660,1)));
% Y = array2timetable(state1_data(1:1:36660,:),RowTimes=seconds(time1(1:1:36660,1)));

%%%%%%%%%%%%%Create a Neural State-Space Object%%%%%%%%%%%%%%%%%%%%%%%
%출력과 동일한 하나의 state, 하나의 입력 및 샘플 시간 Ts를 가진 time-invariant discrete-time neural state-space 객체를 생성한다.
%idNeuralStateSpace를 사용하여 식별 가능한(추정 가능한) 네트워크 가중치 및 편향을 가지고 블랙박스 연속 시간 또는
% 이산 시간 신경 상태 공간 모델을 생성한다.
nss=idNeuralStateSpace(4,NumInputs=5,NumOutputs=4,Ts=Ts);   %idNeuralStateSpace(NumStates,NumInputs=1,Ts) creates a neural state-space object with 1 states, 1 inputs, and sample time 0.1.
% 
nss.StateNetwork = createMLPNetwork(nss,'state',...
    LayerSizes=[32 16 8],...    ##[32 32]
    Activations="tanh",...
    WeightsInitializer="glorot",...
    BiasInitializer="zeros");

summary(nss.StateNetwork)

%상태 네트워크에 대한 훈련 옵션을 지정한다. 
% Adam 알고리즘을 사용하고 최대 에포크 수를 300으로 지정한다.(에포크는 전체 훈련 세트에 대한 훈련 알고리즘의 전체 통과하는 것) 
% 알고리즘이 1000개의 실험 데이터 전체 세트를 배치 세트로 사용하여 각각의 iteration 마다 gradient를 계산하도록 한다. 
opt=nssTrainingOptions('adam');
opt.MaxEpochs=10000;
opt.MiniBatchSize=647;
% opt.MiniBatchSize=810;
% opt.MiniBatchSize=352;
% opt.MiniBatchSize=236;


%또한 InputInterSample 옵션을 지정하여 두 샘플링 간격 사이에 입력 상수를 유지한다.
% 마지막으로 학습률을 지정한다.
opt.InputInterSample="zoh";
opt.LearnRate=0.001;       
% 
% %%%%%%%%%%%%%%%%%%Estimate the Neural State-space system%%%%%%%%%%%%%%
% %식별 데이터 세트와 미리 정의된 최적화 옵션 세트를 사용하여 nlssest를 사용하여 nss의 state-network를 훈련시킨다.
% %nss=nlssest(U,Y,nss,opt);
% nss=nlssest(U,Y,nss,opt,'UseLastExperimentForValidation',true);
nss=nlssest(U,Y,nss,opt);

%linearize(nss,[0 0 0 0]',0)

%%%%%%%%%%%%Perform an Extra Validation Check%%%%%%%%%%%%%%%%%%%%%%
% %input time series와 random initial state를 생성
% 
% Var1=input3_data;
% x0_1=state3_data(1,:)';
% 
% %동일한 initial state로부터 동일한 input data를 가지고 linear state space system과 neural state-space system을 시뮬레이션한다.
% 
% %Simulate original system from  x0
% ylin1=state3_data;
% 
% t3=(0:0.001:35.258)';
% u1=array2timetable(Var1,RowTimes=seconds(time3));
% u2=array2timetable(Var1,RowTimes=seconds(t3));
% 
% %Simulate neural state-space system from x0
% simOpt1=simOptions('InitialCondition',x0_1);
% yn1 = sim(nss,u1,simOpt1);
% 
% %두 시스템의 output을 plot하고, plot title에 difference의 norm을 보이게 힌다.
% %stairs(t,[ylin yn.Variables]);
% subplot(4,1,1);
% stairs(t3,[ylin1(:,1) yn1.Var1])
% xlabel("Time"); ylabel("y [m]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin1(:,1)-yn1.Var1))])
% 
% subplot(4,1,2);
% stairs(t3,[ylin1(:,2) yn1.Var2])
% xlabel("Time"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin1(:,2)-yn1.Var2))])
% 
% subplot(4,1,3);
% stairs(t3,[ylin1(:,3) yn1.Var3])
% xlabel("Time"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin1(:,3)-yn1.Var3))])
% 
% subplot(4,1,4);
% stairs(t3,[ylin1(:,4) yn1.Var4])
% xlabel("Time"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin1(:,4)-yn1.Var4))])

% figure
% compare(input3_data,state3_data,nss)


% u2=input4_data;
% x0_2=state4_data(1,:);
% 
% %동일한 initial state로부터 동일한 input data를 가지고 linear state space system과 neural state-space system을 시뮬레이션한다.
% 
% %Simulate original system from  x0
% ylin2=state4_data;
% 
% %Simulate neural state-space system from x0
% simOpt2=simOptions('InitialCondition',x0_2);
% yn2=sim(nss,array2timetable(u2,RowTimes=seconds(time4)),simOpt2);
% 
% 
% %두 시스템의 output을 plot하고, plot title에 difference의 norm을 보이게 힌다.
% %stairs(t,[ylin yn.Variables]);
% subplot(4,1,1);
% stairs(time4,[ylin2(:,1) yn2.x1])
% xlabel("Time"); ylabel("y [m]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin2(:,1)-yn2.x1))])
% 
% subplot(4,1,2);
% stairs(time4,[ylin2(:,2) yn2.x2])
% xlabel("Time"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin2(:,2)-yn2.x2))])
% 
% subplot(4,1,3);
% stairs(time4,[ylin2(:,3) yn2.x3])
% xlabel("Time"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin2(:,3)-yn2.x3))])
% 
% subplot(4,1,4);
% stairs(time4,[ylin2(:,4) yn2.x4])
% xlabel("Time"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title(['Approximation error= ' num2str(norm(ylin2(:,4)-yn2.x4))])

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt1%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simOpt = simOptions('InitialCondition',x0);
% yn = sim(nss,array2timetable(input,RowTimes=seconds(time1_1)),simOpt);
sz1=size(time1_1);
%x_kh=x.';
%input_kh=input.';
x1_0 = x1(1,:).';

%x_k = x_k.';
x1_k = zeros(4,sz1(1)); 
x1_k(:,1)=x1_0;
x1 = x1.';
input_1 = input_1.';

for k = 1:length(time1_1)-1
    x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
end

%stairs(time1_1,[x_dot x_k.']);
x1_k = x1_k.';

ylin1_1 = x_dot1(:,1);
yn1_1 = x1_k(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% figure
% plot(time1_1,ylin1_1,'r',time1_1,yn1_1,'b--')
% grid on
% xlabel("Time [s]"); ylabel("y [m]");
% legend("Original","Estimated");
% title('lateral position')
%title({'lateral position'; ['Approximation error =  ' num2str(norm(ylin1-yn1)) ]})

% hold on

ylin1_2 = x_dot1(:,2);
yn1_2 = x1_k(:,2);
% figure
% plot(time1_1,ylin1_2,'r',time1_1,yn1_2,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title('lateral velocity')
%title({'lateral velocity';['Approximation error =  ' num2str(norm(ylin2-yn2)) ]})

ylin1_3 = x_dot1(:,3);
yn1_3 = x1_k(:,3);
% figure
% plot(time1_1,ylin1_3,'r',time1_1,yn1_3,'b--')
% grid on
% xlabel("Time [s]"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title('yaw angle')
%title({'yaw angle';['Approximation error =  ' num2str(norm(ylin3-yn3)) ]})

ylin1_4 = x_dot1(:,4);
yn1_4 = x1_k(:,4);
% figure
% plot(time1_1,ylin1_4,'r',time1_1,yn1_4,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title('yaw angle rate')
%title({'yaw angle rate';['Approximation error =  ' num2str(norm(ylin4-yn4)) ]})

%%%%%%%%%%%%%%%%%%%%%%gt1 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt1');
subplot(2,2,1);
plot(time1_1,ylin1_1,'r',time1_1,yn1_1,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin1_1-yn1_1).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time1_1,ylin1_2,'r',time1_1,yn1_2,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin1_2-yn1_2).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time1_1,ylin1_3,'r',time1_1,yn1_3,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin1_3-yn1_3).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time1_1,ylin1_4,'r',time1_1,yn1_4,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin1_4-yn1_4).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

%%%%%%%%%%%%%%%%%%%%%%gt1 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution1');
% subplot(2,2,1);
% error1_1 = ylin1_1-yn1_1;
% [V1_1,M1_1] = var(error1_1);
% std1_1 = sqrt(V1_1);
% normal_distribution1_1 = makedist('Normal', 'mu', M1_1, 'sigma', std1_1);
% x1_1 = linspace(M1_1 - 3 * std1_1, M1_1 + 3 * std1_1, 1000); % 플롯을 위한 x 값 범위
% y1_1 = pdf(normal_distribution1_1, x1_1); % 확률 밀도 함수 계산
% plot(x1_1, y1_1,'r', 'LineWidth', 2);
% xline(M1_1, 'k--');
% title({'Normal distribution of Lateral position error';['mean= ' num2str(M1_1) ', std= ' num2str(std1_1)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2);
% error1_2 = ylin1_2-yn1_2;
% [V1_2,M1_2] = var(error1_2);
% std1_2 = sqrt(V1_2);
% normal_distribution1_2 = makedist('Normal', 'mu', M1_2, 'sigma', std1_2);
% x1_2 = linspace(M1_2 - 3 * std1_2, M1_2 + 3 * std1_2, 1000); % 플롯을 위한 x 값 범위
% y1_2 = pdf(normal_distribution1_2, x1_2); % 확률 밀도 함수 계산
% plot(x1_2, y1_2,'g', 'LineWidth', 2);
% xline(M1_2, 'k--');
% title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M1_2) ', std= ' num2str(std1_2)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3);
% error1_3 = ylin1_3-yn1_3;
% [V1_3,M1_3] = var(error1_3);
% std1_3 = sqrt(V1_3);
% normal_distribution1_3 = makedist('Normal', 'mu', M1_3, 'sigma', std1_3);
% x1_3 = linspace(M1_3 - 3 * std1_3, M1_3 + 3 * std1_3, 1000); % 플롯을 위한 x 값 범위
% y1_3 = pdf(normal_distribution1_3, x1_3); % 확률 밀도 함수 계산
% plot(x1_3, y1_3,'b', 'LineWidth', 2);
% xline(M1_3, 'k--');
% title({'Normal distribution of yaw error';['mean= ' num2str(M1_3) ', std= ' num2str(std1_3)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4);
% error1_4 = ylin1_4-yn1_4;
% [V1_4,M1_4] = var(error1_4);
% std1_4 = sqrt(V1_4);
% normal_distribution1_4 = makedist('Normal', 'mu', M1_4, 'sigma', std1_4);
% x1_4 = linspace(M1_4 - 3 * std1_4, M1_4 + 3 * std1_4, 1000); % 플롯을 위한 x 값 범위
% y1_4 = pdf(normal_distribution1_4, x1_4); % 확률 밀도 함수 계산
% plot(x1_4, y1_4,'c', 'LineWidth', 2);
% xline(M1_4, 'k--');
% title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M1_4) ', std= ' num2str(std1_4)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%gt2%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simOpt = simOptions('InitialCondition',x0);
% yn = sim(nss,array2timetable(input,RowTimes=seconds(time1_1)),simOpt);
sz2=size(time2_1);
%x_kh=x.';
%input_kh=input.';
x2_0 = x2(1,:).';

%x_k = x_k.';
x2_k = zeros(4,sz2(1)); 
x2_k(:,1)=x2_0;
x2 = x2.';
input_2 = input_2.';

for k = 1:length(time2_1)-1
    x2_k(:,k+1) = evaluate(nss,x2(:,k),input_2(:,k)); 
end

%stairs(time1_1,[x_dot x_k.']);
x2_k = x2_k.';

ylin2_1 = x_dot2(:,1);
yn2_1 = x2_k(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% figure
% plot(time2_1,ylin2_1,'r',time2_1,yn2_1,'b--')
% grid on
% xlabel("Time [s]"); ylabel("y [m]");
% legend("Original","Estimated");
% title('lateral position')
%title({'lateral position'; ['Approximation error =  ' num2str(norm(ylin1-yn1)) ]})

% hold on

ylin2_2 = x_dot2(:,2);
yn2_2 = x2_k(:,2);
% figure
% plot(time2_1,ylin2_2,'r',time2_1,yn2_2,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title('lateral velocity')
%title({'lateral velocity';['Approximation error =  ' num2str(norm(ylin2-yn2)) ]})

ylin2_3 = x_dot2(:,3);
yn2_3 = x2_k(:,3);
% figure
% plot(time2_1,ylin2_3,'r',time2_1,yn2_3,'b--')
% grid on
% xlabel("Time [s]"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title('yaw angle')
%title({'yaw angle';['Approximation error =  ' num2str(norm(ylin3-yn3)) ]})

ylin2_4 = x_dot2(:,4);
yn2_4 = x2_k(:,4);
% figure
% plot(time2_1,ylin2_4,'r',time2_1,yn2_4,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title('yaw angle rate')
%title({'yaw angle rate';['Approximation error =  ' num2str(norm(ylin4-yn4)) ]})

%%%%%%%%%%%%%%%%%%%%%%gt2 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt2');
subplot(2,2,1);
plot(time2_1,ylin2_1,'r',time2_1,yn2_1,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin2_1-yn2_1).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time2_1,ylin2_2,'r',time2_1,yn2_2,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin2_2-yn2_2).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time2_1,ylin2_3,'r',time2_1,yn2_3,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin2_3-yn2_3).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time2_1,ylin2_4,'r',time2_1,yn2_4,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin2_4-yn2_4).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

%%%%%%%%%%%%%%%%%%%%%gt2 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution2');
% subplot(2,2,1);
% error2_1 = ylin2_1-yn2_1;
% [V2_1,M2_1] = var(error2_1);
% std2_1 = sqrt(V2_1);
% normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
% x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
% y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산
% plot(x2_1, y2_1,'r', 'LineWidth', 2);
% xline(M2_1, 'k--');
% title({'Normal distribution of Lateral position error';['mean= ' num2str(M2_1) ', std= ' num2str(std2_1)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2);
% error2_2 = ylin2_2-yn2_2;
% [V2_2,M2_2] = var(error2_2);
% std2_2 = sqrt(V2_2);
% normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
% x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
% y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산
% plot(x2_2, y2_2,'g', 'LineWidth', 2);
% xline(M2_2, 'k--');
% title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M2_2) ', std= ' num2str(std2_2)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3);
% error2_3 = ylin2_3-yn2_3;
% [V2_3,M2_3] = var(error2_3);
% std2_3 = sqrt(V2_3);
% normal_distribution2_3 = makedist('Normal', 'mu', M2_3, 'sigma', std2_3);
% x2_3 = linspace(M2_3 - 3 * std2_3, M2_3 + 3 * std2_3, 1000); % 플롯을 위한 x 값 범위
% y2_3 = pdf(normal_distribution2_3, x2_3); % 확률 밀도 함수 계산
% plot(x2_3, y2_3,'b', 'LineWidth', 2);
% xline(M2_3, 'k--');
% title({'Normal distribution of yaw error';['mean= ' num2str(M2_3) ', std= ' num2str(std2_3)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4);
% error2_4 = ylin2_4-yn2_4;
% [V2_4,M2_4] = var(error2_4);
% std2_4 = sqrt(V2_4);
% normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
% x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
% y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산
% plot(x2_4, y2_4,'c', 'LineWidth', 2);
% xline(M2_4, 'k--');
% title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M2_4) ', std= ' num2str(std2_4)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;

%%%%%%%%%%%%%%%%%%%%%gt1 & gt2 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','Normal distribution gt1 & gt2');
subplot(2,2,1);
error1_1 = ylin1_1-yn1_1;
[V1_1,M1_1] = var(error1_1);
std1_1 = sqrt(V1_1);
normal_distribution1_1 = makedist('Normal', 'mu', M1_1, 'sigma', std1_1);
x1_1 = linspace(M1_1 - 3 * std1_1, M1_1 + 3 * std1_1, 1000); % 플롯을 위한 x 값 범위
y1_1 = pdf(normal_distribution1_1, x1_1); % 확률 밀도 함수 계산

error2_1 = ylin2_1-yn2_1;
[V2_1,M2_1] = var(error2_1);
std2_1 = sqrt(V2_1);
normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산

plot(x1_1, y1_1,'r--', 'LineWidth', 2);
hold on;
plot(x2_1, y2_1,'b--', 'LineWidth', 2);
xline(M1_1, 'r--');
xline(M2_1, 'b--');
legend("gt1","gt2");
title('Normal distribution of Lateral position error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,2);
error1_2 = ylin1_2-yn1_2;
[V1_2,M1_2] = var(error1_2);
std1_2 = sqrt(V1_2);
normal_distribution1_2 = makedist('Normal', 'mu', M1_2, 'sigma', std1_2);
x1_2 = linspace(M1_2 - 3 * std1_2, M1_2 + 3 * std1_2, 1000); % 플롯을 위한 x 값 범위
y1_2 = pdf(normal_distribution1_2, x1_2); % 확률 밀도 함수 계산

error2_2 = ylin2_2-yn2_2;
[V2_2,M2_2] = var(error2_2);
std2_2 = sqrt(V2_2);
normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산

plot(x1_2, y1_2,'r--', 'LineWidth', 2);
hold on;
plot(x2_2, y2_2,'b--', 'LineWidth', 2);
xline(M1_2, 'r--');
xline(M2_2, 'b--');
legend("gt1","gt2");
title('Normal distribution of Lateral velocity error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,3);
error1_3 = ylin1_3-yn1_3;
[V1_3,M1_3] = var(error1_3);
std1_3 = sqrt(V1_3);
normal_distribution1_3 = makedist('Normal', 'mu', M1_3, 'sigma', std1_3);
x1_3 = linspace(M1_3 - 3 * std1_3, M1_3 + 3 * std1_3, 1000); % 플롯을 위한 x 값 범위
y1_3 = pdf(normal_distribution1_3, x1_3); % 확률 밀도 함수 계산

error2_3 = ylin2_3-yn2_3;
[V2_3,M2_3] = var(error2_3);
std2_3 = sqrt(V2_3);
normal_distribution2_3 = makedist('Normal', 'mu', M2_3, 'sigma', std2_3);
x2_3 = linspace(M2_3 - 3 * std2_3, M2_3 + 3 * std2_3, 1000); % 플롯을 위한 x 값 범위
y2_3 = pdf(normal_distribution2_3, x2_3); % 확률 밀도 함수 계산

plot(x1_3, y1_3,'r--', 'LineWidth', 2);
hold on;
plot(x2_3, y2_3,'b--', 'LineWidth', 2);
xline(M1_3, 'r--');
xline(M2_3, 'b--');
legend("gt1","gt2");
title('Normal distribution of yaw error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,4);
error1_4 = ylin1_4-yn1_4;
[V1_4,M1_4] = var(error1_4);
std1_4 = sqrt(V1_4);
normal_distribution1_4 = makedist('Normal', 'mu', M1_4, 'sigma', std1_4);
x1_4 = linspace(M1_4 - 3 * std1_4, M1_4 + 3 * std1_4, 1000); % 플롯을 위한 x 값 범위
y1_4 = pdf(normal_distribution1_4, x1_4); % 확률 밀도 함수 계산

error2_4 = ylin2_4-yn2_4;
[V2_4,M2_4] = var(error2_4);
std2_4 = sqrt(V2_4);
normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산

plot(x1_4, y1_4,'r--', 'LineWidth', 2);
hold on;
plot(x2_4, y2_4,'b--', 'LineWidth', 2);
xline(M1_4, 'r--');
xline(M2_4, 'b--');
legend("gt1","gt2");
title('Normal distribution of yaw angle rate error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simOpt = simOptions('InitialCondition',x0);
% yn = sim(nss,array2timetable(input,RowTimes=seconds(time1_1)),simOpt);
sz3=size(time3_1);
%x_kh=x.';
%input_kh=input.';
x3_0 = x3(1,:).';

%x_k = x_k.';
x3_k = zeros(4,sz3(1)); 
x3_k(:,1)=x3_0;
x3 = x3.';
input_3 = input_3.';

for k = 1:length(time3_1)-1
    x3_k(:,k+1) = evaluate(nss,x3(:,k),input_3(:,k)); 
end

%stairs(time1_1,[x_dot x_k.']);
x3_k = x3_k.';

ylin3_1 = x_dot3(:,1);
yn3_1 = x3_k(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% figure
% plot(time3_1,ylin3_1,'r',time3_1,yn3_1,'b--')
% grid on
% xlabel("Time [s]"); ylabel("y [m]");
% legend("Original","Estimated");
% title('lateral position')
%title({'lateral position'; ['Approximation error =  ' num2str(norm(ylin1-yn1)) ]})

% hold on

ylin3_2 = x_dot3(:,2);
yn3_2 = x3_k(:,2);
% figure
% plot(time3_1,ylin3_2,'r',time3_1,yn3_2,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title('lateral velocity')
%title({'lateral velocity';['Approximation error =  ' num2str(norm(ylin2-yn2)) ]})

ylin3_3 = x_dot3(:,3);
yn3_3 = x3_k(:,3);
% figure
% plot(time3_1,ylin3_3,'r',time3_1,yn3_3,'b--')
% grid on
% xlabel("Time [s]"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title('yaw angle')
%title({'yaw angle';['Approximation error =  ' num2str(norm(ylin3-yn3)) ]})

ylin3_4 = x_dot3(:,4);
yn3_4 = x3_k(:,4);
% figure
% plot(time3_1,ylin3_4,'r',time3_1,yn3_4,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title('yaw angle rate')
%title({'yaw angle rate';['Approximation error =  ' num2str(norm(ylin4-yn4)) ]})

%%%%%%%%%%%%%%%%%%%%%%gt3 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt3');
subplot(2,2,1);
plot(time3_1,ylin3_1,'r',time3_1,yn3_1,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin3_1-yn3_1).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time3_1,ylin3_2,'r',time3_1,yn3_2,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin3_2-yn3_2).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time3_1,ylin3_3,'r',time3_1,yn3_3,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin3_3-yn3_3).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time3_1,ylin3_4,'r',time3_1,yn3_4,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin3_4-yn3_4).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

% %%%%%%%%%%%%%%%%%%%%%gt3 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution3');
% subplot(2,2,1);
% error3_1 = ylin3_1-yn3_1;
% [V3_1,M3_1] = var(error3_1);
% std3_1 = sqrt(V3_1);
% normal_distribution3_1 = makedist('Normal', 'mu', M3_1, 'sigma', std3_1);
% x3_1 = linspace(M3_1 - 3 * std3_1, M3_1 + 3 * std3_1, 1000); % 플롯을 위한 x 값 범위
% y3_1 = pdf(normal_distribution3_1, x3_1); % 확률 밀도 함수 계산
% plot(x3_1, y3_1,'r', 'LineWidth', 2);
% xline(M3_1, 'k--');
% title({'Normal distribution of Lateral position error';['mean= ' num2str(M3_1) ', std= ' num2str(std3_1)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2);
% error3_2 = ylin3_2-yn3_2;
% [V3_2,M3_2] = var(error3_2);
% std3_2 = sqrt(V3_2);
% normal_distribution3_2 = makedist('Normal', 'mu', M3_2, 'sigma', std3_2);
% x3_2 = linspace(M3_2 - 3 * std3_2, M3_2 + 3 * std3_2, 1000); % 플롯을 위한 x 값 범위
% y3_2 = pdf(normal_distribution3_2, x3_2); % 확률 밀도 함수 계산
% plot(x3_2, y3_2,'g', 'LineWidth', 2);
% xline(M3_2, 'k--');
% title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M3_2) ', std= ' num2str(std3_2)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3);
% error3_3 = ylin3_3-yn3_3;
% [V3_3,M3_3] = var(error3_3);
% std3_3 = sqrt(V3_3);
% normal_distribution3_3 = makedist('Normal', 'mu', M3_3, 'sigma', std3_3);
% x3_3 = linspace(M3_3 - 3 * std3_3, M3_3 + 3 * std3_3, 1000); % 플롯을 위한 x 값 범위
% y3_3 = pdf(normal_distribution3_3, x3_3); % 확률 밀도 함수 계산
% plot(x3_3, y3_3,'b', 'LineWidth', 2);
% xline(M3_3, 'k--');
% title({'Normal distribution of yaw error';['mean= ' num2str(M3_3) ', std= ' num2str(std3_3)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4);
% error3_4 = ylin3_4-yn3_4;
% [V3_4,M3_4] = var(error3_4);
% std3_4 = sqrt(V3_4);
% normal_distribution3_4 = makedist('Normal', 'mu', M3_4, 'sigma', std3_4);
% x3_4 = linspace(M3_4 - 3 * std3_4, M3_4 + 3 * std3_4, 1000); % 플롯을 위한 x 값 범위
% y3_4 = pdf(normal_distribution3_4, x3_4); % 확률 밀도 함수 계산
% plot(x3_4, y3_4,'c', 'LineWidth', 2);
% xline(M3_4, 'k--');
% title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M3_4) ', std= ' num2str(std3_4)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simOpt = simOptions('InitialCondition',x0);
% yn = sim(nss,array2timetable(input,RowTimes=seconds(time1_1)),simOpt);
sz4=size(time4_1);
%x_kh=x.';
%input_kh=input.';
x4_0 = x4(1,:).';

%x_k = x_k.';
x4_k = zeros(4,sz4(1)); 
x4_k(:,1)=x4_0;
x4 = x4.';
input_4 = input_4.';

for k = 1:length(time4_1)-1
    x4_k(:,k+1) = evaluate(nss,x4(:,k),input_4(:,k)); 
end

%stairs(time1_1,[x_dot4 x_k.']);
x4_k = x4_k.';

ylin4_1 = x_dot4(:,1);
yn4_1 = x4_k(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% figure
% plot(time4_1,ylin4_1,'r',time4_1,yn4_1,'b--')
% grid on
% xlabel("Time [s]"); ylabel("y [m]");
% legend("Original","Estimated");
% title('lateral position')
%title({'lateral position'; ['Approximation error =  ' num2str(norm(ylin1-yn1)) ]})

% hold on

ylin4_2 = x_dot4(:,2);
yn4_2 = x4_k(:,2);
% figure
% plot(time4_1,ylin4_2,'r',time4_1,yn4_2,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dy/dt [m/s]");
% legend("Original","Estimated");
% title('lateral velocity')
%title({'lateral velocity';['Approximation error =  ' num2str(norm(ylin2-yn2)) ]})

ylin4_3 = x_dot4(:,3);
yn4_3 = x4_k(:,3);
% figure
% plot(time4_1,ylin4_3,'r',time4_1,yn4_3,'b--')
% grid on
% xlabel("Time [s]"); ylabel("ψ [rad]");
% legend("Original","Estimated");
% title('yaw angle')
%title({'yaw angle';['Approximation error =  ' num2str(norm(ylin3-yn3)) ]})

ylin4_4 = x_dot4(:,4);
yn4_4 = x4_k(:,4);
% figure
% plot(time4_1,ylin4_4,'r',time4_1,yn4_4,'b--')
% grid on
% xlabel("Time [s]"); ylabel("dψ/dt [rad/s]");
% legend("Original","Estimated");
% title('yaw angle rate')
%title({'yaw angle rate';['Approximation error =  ' num2str(norm(ylin4-yn4)) ]})

%%%%%%%%%%%%%%%%%%%%%%gt4 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt4');
subplot(2,2,1);
plot(time4_1,ylin4_1,'r',time4_1,yn4_1,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin4_1-yn4_1).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time4_1,ylin4_2,'r',time4_1,yn4_2,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin4_2-yn4_2).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time4_1,ylin4_3,'r',time4_1,yn4_3,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin4_3-yn4_3).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time4_1,ylin4_4,'r',time4_1,yn4_4,'b--')
legend("Original","Estimated",'FontSize',15);
grid on
title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin4_4-yn4_4).^2)))]},'FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

%%%%%%%%%%%%%%%%%%%%%gt4 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution4');
% subplot(2,2,1);
% error4_1 = ylin4_1-yn4_1;
% [V4_1,M4_1] = var(error4_1);
% std4_1 = sqrt(V4_1);
% normal_distribution4_1 = makedist('Normal', 'mu', M4_1, 'sigma', std4_1);
% x4_1 = linspace(M4_1 - 3 * std4_1, M4_1 + 3 * std4_1, 1000); % 플롯을 위한 x 값 범위
% y4_1 = pdf(normal_distribution4_1, x4_1); % 확률 밀도 함수 계산
% plot(x4_1, y4_1,'r', 'LineWidth', 2);
% xline(M4_1, 'k--');
% title({'Normal distribution of Lateral position error';['mean= ' num2str(M4_1) ', std= ' num2str(std4_1)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2);
% error4_2 = ylin4_2-yn4_2;
% [V4_2,M4_2] = var(error4_2);
% std4_2 = sqrt(V4_2);
% normal_distribution4_2 = makedist('Normal', 'mu', M4_2, 'sigma', std4_2);
% x4_2 = linspace(M4_2 - 3 * std4_2, M4_2 + 3 * std4_2, 1000); % 플롯을 위한 x 값 범위
% y4_2 = pdf(normal_distribution4_2, x4_2); % 확률 밀도 함수 계산
% plot(x4_2, y4_2,'g', 'LineWidth', 2);
% xline(M4_2, 'k--');
% title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M4_2) ', std= ' num2str(std4_2)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3);
% error4_3 = ylin4_3-yn4_3;
% [V4_3,M4_3] = var(error4_3);
% std4_3 = sqrt(V4_3);
% normal_distribution4_3 = makedist('Normal', 'mu', M4_3, 'sigma', std4_3);
% x4_3 = linspace(M4_3 - 3 * std4_3, M4_3 + 3 * std4_3, 1000); % 플롯을 위한 x 값 범위
% y4_3 = pdf(normal_distribution4_3, x4_3); % 확률 밀도 함수 계산
% plot(x4_3, y4_3,'b', 'LineWidth', 2);
% xline(M4_3, 'k--');
% title({'Normal distribution of yaw error';['mean= ' num2str(M4_3) ', std= ' num2str(std4_3)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4);
% error4_4 = ylin4_4-yn4_4;
% [V4_4,M4_4] = var(error4_4);
% std4_4 = sqrt(V4_4);
% normal_distribution4_4 = makedist('Normal', 'mu', M4_4, 'sigma', std4_4);
% x4_4 = linspace(M4_4 - 3 * std4_4, M4_4 + 3 * std4_4, 1000); % 플롯을 위한 x 값 범위
% y4_4 = pdf(normal_distribution4_4, x4_4); % 확률 밀도 함수 계산
% plot(x4_4, y4_4,'c', 'LineWidth', 2);
% xline(M4_4, 'k--');
% title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M4_4) ', std= ' num2str(std4_4)]},'FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;


%%%%%%%%%%%%%%%%%%%%%gt3 & gt4 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','Normal distribution gt3 & gt4');
subplot(2,2,1);
error3_1 = ylin3_1-yn3_1;
[V3_1,M3_1] = var(error3_1);
std3_1 = sqrt(V3_1);
normal_distribution3_1 = makedist('Normal', 'mu', M3_1, 'sigma', std3_1);
x3_1 = linspace(M3_1 - 3 * std3_1, M3_1 + 3 * std3_1, 1000); % 플롯을 위한 x 값 범위
y3_1 = pdf(normal_distribution3_1, x3_1); % 확률 밀도 함수 계산

error4_1 = ylin4_1-yn4_1;
[V4_1,M4_1] = var(error4_1);
std4_1 = sqrt(V4_1);
normal_distribution4_1 = makedist('Normal', 'mu', M4_1, 'sigma', std4_1);
x4_1 = linspace(M4_1 - 3 * std4_1, M4_1 + 3 * std4_1, 1000); % 플롯을 위한 x 값 범위
y4_1 = pdf(normal_distribution4_1, x4_1); % 확률 밀도 함수 계산

plot(x3_1, y3_1,'r--', 'LineWidth', 2);
hold on;
plot(x4_1, y4_1,'b--', 'LineWidth', 2);
xline(M3_1, 'r--');
xline(M4_1, 'b--');
legend("gt3","gt4");
title('Normal distribution of Lateral position error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,2);
error3_2 = ylin3_2-yn3_2;
[V3_2,M3_2] = var(error3_2);
std3_2 = sqrt(V3_2);
normal_distribution3_2 = makedist('Normal', 'mu', M3_2, 'sigma', std3_2);
x3_2 = linspace(M3_2 - 3 * std3_2, M3_2 + 3 * std3_2, 1000); % 플롯을 위한 x 값 범위
y3_2 = pdf(normal_distribution3_2, x3_2); % 확률 밀도 함수 계산

error4_2 = ylin4_2-yn4_2;
[V4_2,M4_2] = var(error4_2);
std4_2 = sqrt(V4_2);
normal_distribution4_2 = makedist('Normal', 'mu', M4_2, 'sigma', std4_2);
x4_2 = linspace(M4_2 - 3 * std4_2, M4_2 + 3 * std4_2, 1000); % 플롯을 위한 x 값 범위
y4_2 = pdf(normal_distribution4_2, x4_2); % 확률 밀도 함수 계산

plot(x3_2, y3_2,'r--', 'LineWidth', 2);
hold on;
plot(x4_2, y4_2,'b--', 'LineWidth', 2);
xline(M3_2, 'r--');
xline(M4_2, 'b--');
legend("gt3","gt4");
title('Normal distribution of Lateral velocity error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,3);
error3_3 = ylin3_3-yn3_3;
[V3_3,M3_3] = var(error3_3);
std3_3 = sqrt(V3_3);
normal_distribution3_3 = makedist('Normal', 'mu', M3_3, 'sigma', std3_3);
x3_3 = linspace(M3_3 - 3 * std3_3, M3_3 + 3 * std3_3, 1000); % 플롯을 위한 x 값 범위
y3_3 = pdf(normal_distribution3_3, x3_3); % 확률 밀도 함수 계산

error4_3 = ylin4_3-yn4_3;
[V4_3,M4_3] = var(error4_3);
std4_3 = sqrt(V4_3);
normal_distribution4_3 = makedist('Normal', 'mu', M4_3, 'sigma', std4_3);
x4_3 = linspace(M4_3 - 3 * std4_3, M4_3 + 3 * std4_3, 1000); % 플롯을 위한 x 값 범위
y4_3 = pdf(normal_distribution4_3, x4_3); % 확률 밀도 함수 계산

plot(x3_3, y3_3,'r--', 'LineWidth', 2);
hold on;
plot(x4_3, y4_3,'b--', 'LineWidth', 2);
xline(M3_3, 'r--');
xline(M4_3, 'b--');
legend("gt3","gt4");
title('Normal distribution of yaw error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

subplot(2,2,4);
error3_4 = ylin3_4-yn3_4;
[V3_4,M3_4] = var(error3_4);
std3_4 = sqrt(V3_4);
normal_distribution3_4 = makedist('Normal', 'mu', M3_4, 'sigma', std3_4);
x3_4 = linspace(M3_4 - 3 * std3_4, M3_4 + 3 * std3_4, 1000); % 플롯을 위한 x 값 범위
y3_4 = pdf(normal_distribution3_4, x3_4); % 확률 밀도 함수 계산

error4_4 = ylin4_4-yn4_4;
[V4_4,M4_4] = var(error4_4);
std4_4 = sqrt(V4_4);
normal_distribution4_4 = makedist('Normal', 'mu', M4_4, 'sigma', std4_4);
x4_4 = linspace(M4_4 - 3 * std4_4, M4_4 + 3 * std4_4, 1000); % 플롯을 위한 x 값 범위
y4_4 = pdf(normal_distribution4_4, x4_4); % 확률 밀도 함수 계산

plot(x3_4, y3_4,'r--', 'LineWidth', 2);
hold on;
plot(x4_4, y4_4,'b--', 'LineWidth', 2);
xline(M3_4, 'r--');
xline(M4_4, 'b--');
legend("gt3","gt4");
title('Normal distribution of yaw angle rate error','FontSize',15)
xlabel('error');
ylabel('probability density');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%compare%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% compare(input(1:1:44600,:),x_dot(1:1:44600,:),nss)

%%%%%%%%%%%%%%%%%%%%%%%%%%%gt2%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% compare(input(1:45800,:),x_dot(1:45800,:),nss)

%%%%%%%%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% compare(input(28201:35258,:),x_dot(28201:35258,:),nss)

%%%%%%%%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% compare(input(23601:29542,:),x_dot(23601:29542,:),nss)