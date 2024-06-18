%close all; clear; clc;

%각각 다른 초기 상태에서 시작하여 1초 동안 지속되는 1000개의 시뮬레이션을 실행한다. 
%각 실험은 동일한 시점을 사용해야 한다.

run("anal.m");

%Neural Network model 불러오기
% load("newgt1plusgt2plusinverse_32168_nss_delayed.mat");
% load("newgt3plusgt4plusinverse_32168_nss_delayed.mat");
% load("newgt5plusgt6plusgt7plusinverse_32168_nss_delayed.mat");
% load("newgt8plusgt9plusinverse_32168_nss_delayed.mat");
load("noinverse_gtattyaw_gt1plusgt2_32168_nss_delayed.mat");
nss_gt12 = nss;

load("noinverse_gtattyaw_gt3plusgt4_32168_nss_delayed.mat");
nss_gt34 = nss;

Ts=0.01;

%%% 2023.12.21 : nlssest에 정의하는 Y에 x1을 넣었을때보다 x_dot1을 넣었을 때 오차가 더 적음 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(904,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(904,1);
time1_1 = time1(1:end-1);
time1_2 = time1(2:end);

x1 = state1_data(1:end-1,:);
x_dot1 = state1_data(2:end,:);

input1 = input1_data(1:end-1);
input_1 = cat(2,x1,input1);

% for i=0:445
% 
%     U{i+1} = array2timetable(input_1(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1} = array2timetable(x_dot1(100*i+1:1:100*(i+1),:),RowTimes=seconds(time1_2(100*i+1:1:100*(i+1),1)));
% 
% end

%%%%%%%%%%%%%%%%%%%%%gt2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(1620,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(1620,1);
time2_1 = time2(1:end-1);
time2_2 = time2(2:end);

x2 = state2_data(1:end-1,:);
x_dot2 = state2_data(2:end,:);

input2 = input2_data(1:end-1);
input_2 = cat(2,x2,input2);

% for i=0:457
% 
%     U{i+447} = array2timetable(input_2(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_1(100*i+1:1:100*(i+1),1)));
%     Y{i+447} = array2timetable(x_dot2(100*i+1:1:100*(i+1),:),RowTimes=seconds(time2_2(100*i+1:1:100*(i+1),1)));
% 
% end

%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(647,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(647,1);
time3_1 = time3(1:end-1);
time3_2 = time3(2:end);

x3 = state3_data(1:end-1,:);
x_dot3 = state3_data(2:end,:);

input3 = input3_data(1:end-1);
input_3 = cat(2,x3,input3);

% for i=0:351
%     
%     U{i+1} = array2timetable(input_3(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1} = array2timetable(x_dot3(100*i+1:1:100*(i+1),:),RowTimes=seconds(time3_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(296,1);

time4_1 = time4(1:end-1);
time4_2 = time4(2:end);

x4 = state4_data(1:end-1,:);
x_dot4 = state4_data(2:end,:);

input4 = input4_data(1:end-1);
input_4 = cat(2,x4,input4);

% for i=0:294
%     
%     U{i+353} = array2timetable(input_4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_1(100*i+1:1:100*(i+1),1)));
%     Y{i+353} = array2timetable(x_dot4(100*i+1:1:100*(i+1),:),RowTimes=seconds(time4_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% U=cell(8202,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% Y=cell(8202,1);

% time5_1 = time5(1:end-1);
% time5_2 = time5(2:end);
% 
% x5 = state5_data(1:end-1,:);
% x_dot5 = state5_data(2:end,:);
% 
% input5 = input5_data(1:end-1);
% input_5 = cat(2,x5,input5);
% 
% % for i=0:2654
% %     
% %     U{i+1} = array2timetable(input_5(100*i+1:1:100*(i+1),:),RowTimes=seconds(time5_1(100*i+1:1:100*(i+1),1)));
% %     Y{i+1} = array2timetable(x_dot5(100*i+1:1:100*(i+1),:),RowTimes=seconds(time5_2(100*i+1:1:100*(i+1),1)));
% %     
% % end
% 
% %%%%%%%%%%%%%%%%%%%%gt6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(11094,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(11094,1);
% 
% time6_1 = time6(1:end-1);
% time6_2 = time6(2:end);
% 
% x6 = state6_data(1:end-1,:);
% x_dot6 = state6_data(2:end,:);
% 
% input6 = input6_data(1:end-1);
% input_6 = cat(2,x6,input6);
% 
% % for i=0:2707
% %     
% %     U{i+2656} = array2timetable(input_6(100*i+1:1:100*(i+1),:),RowTimes=seconds(time6_1(100*i+1:1:100*(i+1),1)));
% %     Y{i+2656} = array2timetable(x_dot6(100*i+1:1:100*(i+1),:),RowTimes=seconds(time6_2(100*i+1:1:100*(i+1),1)));
% %     
% % end
% 
% %%%%%%%%%%%%%%%%%%%%gt7%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(5678,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(5678,1);
% 
% time7_1 = time7(1:end-1);
% time7_2 = time7(2:end);
% 
% x7 = state7_data(1:end-1,:);
% x_dot7 = state7_data(2:end,:);
% 
% input7 = input7_data(1:end-1);
% input_7 = cat(2,x7,input7);
% 
% % for i=0:2838
% % 
% %     U{i+5364} = array2timetable(input_7(100*i+1:1:100*(i+1),:),RowTimes=seconds(time7_1(100*i+1:1:100*(i+1),1)));
% %     Y{i+5364} = array2timetable(x_dot7(100*i+1:1:100*(i+1),:),RowTimes=seconds(time7_2(100*i+1:1:100*(i+1),1)));
% % 
% % end
% 
% %%%%%%%%%%%%%%%%%%%%gt8%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(3737,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(3737,1);
% 
% time8_1 = time8(1:end-1);
% time8_2 = time8(2:end);
% 
% x8 = state8_data(1:end-1,:);
% x_dot8 = state8_data(2:end,:);
% 
% input8 = input8_data(1:end-1);
% input_8 = cat(2,x8,input8);
% 
% % for i=0:1461
% %     
% %     U{i+1} = array2timetable(input_8(100*i+1:1:100*(i+1),:),RowTimes=seconds(time8_1(100*i+1:1:100*(i+1),1)));
% %     Y{i+1} = array2timetable(x_dot8(100*i+1:1:100*(i+1),:),RowTimes=seconds(time8_2(100*i+1:1:100*(i+1),1)));
% %     
% % end
% 
% %%%%%%%%%%%%%%%%%%%%gt9%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(296,1);
% 
% time9_1 = time9(1:end-1);
% time9_2 = time9(2:end);
% 
% x9 = state9_data(1:end-1,:);
% x_dot9 = state9_data(2:end,:);
% 
% input9 = input9_data(1:end-1);
% input_9 = cat(2,x9,input9);

% for i=0:2274
%     
%     U{i+1463} = array2timetable(input_9(100*i+1:1:100*(i+1),:),RowTimes=seconds(time9_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1463} = array2timetable(x_dot9(100*i+1:1:100*(i+1),:),RowTimes=seconds(time9_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%%%%%%%%gt10%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(11430,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(11430,1);
% 
% time10_1 = time10(1:end-1);
% time10_2 = time10(2:end);
% 
% x10 = state10_data(1:end-1,:);
% x_dot10 = state10_data(2:end,:);
% 
% input10 = input10_data(1:end-1);
% input_10 = cat(2,x10,input10);
% 
% for i=0:1873
%     
%     U{i+1} = array2timetable(input_10(100*i+1:1:100*(i+1),:),RowTimes=seconds(time10_1(100*i+1:1:100*(i+1),1)));
%     Y{i+1} = array2timetable(x_dot10(100*i+1:1:100*(i+1),:),RowTimes=seconds(time10_2(100*i+1:1:100*(i+1),1)));
%     
% end
% 
% %%%%%%%%%%%%%%%%%%%%gt11%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(296,1);
% 
% time11_1 = time11(1:end-1);
% time11_2 = time11(2:end);
% 
% x11 = state11_data(1:end-1,:);
% x_dot11 = state11_data(2:end,:);
% 
% input11 = input11_data(1:end-1);
% input_11 = cat(2,x11,input11);
% 
% for i=0:1924
%     
%     U{i+3749} = array2timetable(input_11(100*i+1:1:100*(i+1),:),RowTimes=seconds(time11_1(100*i+1:1:100*(i+1),1)));
%     Y{i+3749} = array2timetable(x_dot11(100*i+1:1:100*(i+1),:),RowTimes=seconds(time11_2(100*i+1:1:100*(i+1),1)));
%     
% end
% 
% %%%%%%%%%%%%%%%%%%%%gt12%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % U=cell(296,1);   %cell: 셀형 배열은 셀이라는 인덱싱된 데이터 컨테이너를 사용하는 데이터형    
% % Y=cell(296,1);
% 
% time12_1 = time12(1:end-1);
% time12_2 = time12(2:end);
% 
% x12 = state12_data(1:end-1,:);
% x_dot12 = state12_data(2:end,:);
% 
% input12 = input12_data(1:end-1);
% input_12 = cat(2,x12,input12);
% 
% for i=0:1915
%     
%     U{i+7599} = array2timetable(input_12(100*i+1:1:100*(i+1),:),RowTimes=seconds(time12_1(100*i+1:1:100*(i+1),1)));
%     Y{i+7599} = array2timetable(x_dot12(100*i+1:1:100*(i+1),:),RowTimes=seconds(time12_2(100*i+1:1:100*(i+1),1)));
%     
% end

%%%%%%%%%%%%%Create a Neural State-Space Object%%%%%%%%%%%%%%%%%%%%%%%
%출력과 동일한 하나의 state, 하나의 입력 및 샘플 시간 Ts를 가진 time-invariant discrete-time neural state-space 객체를 생성한다.
%idNeuralStateSpace를 사용하여 식별 가능한(추정 가능한) 네트워크 가중치 및 편향을 가지고 블랙박스 연속 시간 또는
% 이산 시간 신경 상태 공간 모델을 생성한다.
% nss=idNeuralStateSpace(4,NumInputs=5,NumOutputs=4,Ts=Ts);   %idNeuralStateSpace(NumStates,NumInputs=1,Ts) creates a neural state-space object with 1 states, 1 inputs, and sample time 0.1.
% % 
% nss.StateNetwork = createMLPNetwork(nss,'state',...
%     LayerSizes=[32 16 8],...    ##[32 32]
%     Activations="tanh",...
%     WeightsInitializer="glorot",...
%     BiasInitializer="zeros");
% 
% summary(nss.StateNetwork)

%상태 네트워크에 대한 훈련 옵션을 지정한다. 
% Adam 알고리즘을 사용하고 최대 에포크 수를 300으로 지정한다.(에포크는 전체 훈련 세트에 대한 훈련 알고리즘의 전체 통과하는 것) 
% 알고리즘이 1000개의 실험 데이터 전체 세트를 배치 세트로 사용하여 각각의 iteration 마다 gradient를 계산하도록 한다. 
% opt=nssTrainingOptions('adam');
% opt.MaxEpochs=10000;
% opt.MiniBatchSize=904;      %gt1+gt2
% opt.MiniBatchSize=647;        %gt3+gt4
% opt.MiniBatchSize=8202;         %gt5+gt6+gt7
% opt.MiniBatchSize=3737;          %gt8+gt9
% opt.MiniBatchSize=11430;

%또한 InputInterSample 옵션을 지정하여 두 샘플링 간격 사이에 입력 상수를 유지한다.
% 마지막으로 학습률을 지정한다.
% opt.InputInterSample="zoh";
% opt.LearnRate=0.001;       
% 
% %%%%%%%%%%%%%%%%%%Estimate the Neural State-space system%%%%%%%%%%%%%%
% %식별 데이터 세트와 미리 정의된 최적화 옵션 세트를 사용하여 nlssest를 사용하여 nss의 state-network를 훈련시킨다.
% nss=nlssest(U,Y,nss,opt,'UseLastExperimentForValidation',true);
% nss=nlssest(U,Y,nss,opt);

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt1%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sz1=size(time1_2);
% x1_0 = x1(1,:).';

x1_k = zeros(4,sz1(1)); 
% x1_k(:,1)=x1_0;
x1 = x1.';
input_1 = input_1.';

% gt1_std_1 = zeros(1,sz1(1));
% gt1_std_2 = zeros(1,sz1(1));
% gt1_std_3 = zeros(1,sz1(1));
% gt1_std_4 = zeros(1,sz1(1));

% error=zeros(sz1(1),4);

for k = 1:length(time1_1)
    x1_k(:,k) = evaluate(nss_gt12,x1(:,k),input_1(:,k)); 
% %     gt1_std_1(1,k) = sqrt(mean((x_dot1(1:k,1).'-x1_k(1,1:k)).^2));
% %     gt1_std_2(1,k) = sqrt(mean((x_dot1(1:k,2).'-x1_k(2,1:k)).^2));
% %     gt1_std_3(1,k) = sqrt(mean((x_dot1(1:k,3).'-x1_k(3,1:k)).^2));
% %     gt1_std_4(1,k) = sqrt(mean((x_dot1(1:k,4).'-x1_k(4,1:k)).^2));
%     error(k,1) = x_dot1(1:k,1) - x1_k(1,1:k).';
%     error(k,2) = x_dot1(1:k,2) - x1_k(2,1:k).';
%     error(k,3) = x_dot1(1:k,3) - x1_k(3,1:k).';
%     error(k,4) = x_dot1(1:k,4) - x1_k(4,1:k).';
    
    
end

% min_gt1_std_1 = min(gt1_std_1);
% max_gt1_std_1 = max(gt1_std_1);
% 
% min_gt1_std_2 = min(gt1_std_2);
% max_gt1_std_2 = max(gt1_std_2);
% 
% min_gt1_std_3 = min(gt1_std_3);
% max_gt1_std_3 = max(gt1_std_3);
% 
% min_gt1_std_4 = min(gt1_std_4);
% max_gt1_std_4 = max(gt1_std_4);

% for k = 1:length(time1_1)-1
%     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% end

x1_k = x1_k.';

ylin1_1 = x_dot1(:,1);
yn1_1 = x1_k(:,1);

ylin1_2 = x_dot1(:,2);
yn1_2 = x1_k(:,2);

ylin1_3 = x_dot1(:,3);
yn1_3 = x1_k(:,3);

ylin1_4 = x_dot1(:,4);
yn1_4 = x1_k(:,4);


%%%%%%%%%%%%%%%%%%%%gt1 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt1');
subplot(2,2,1);
plot(time1_2,ylin1_1,'r',time1_2,yn1_1,'b--')
legend("Ground Truth","Estimation",'FontSize',8,'FontName','Times New Roman');
grid on
title('Lateral position, y','FontSize',10,'FontName','Times New Roman')
xlabel('Time [s]','FontSize',8,'FontName','Times New Roman')
ylabel('Position [m]','FontSize',8,'FontName','Times New Roman')

subplot(2,2,2);
plot(time1_2,ylin1_2,'r',time1_2,yn1_2,'b--')
legend("Ground Truth","Estimation",'FontSize',8,'FontName','Times New Roman');
grid on
title('Lateral velocity, $\dot{y}$','FontSize',10,'FontName','Times New Roman','Interpreter','latex')
xlabel('Time [s]','FontSize',8,'FontName','Times New Roman')
ylabel('Velocity [m/s]','FontSize',8,'FontName','Times New Roman')

subplot(2,2,3);
plot(time1_2,ylin1_3,'r',time1_2,yn1_3,'b--')
legend("Ground Truth","Estimation",'FontSize',8,'FontName','Times New Roman');
grid on
title('Yaw, \psi','FontSize',10,'FontName','Times New Roman')
xlabel('Time[s]','FontSize',8,'FontName','Times New Roman')
ylabel('Angle [rad]','FontSize',8,'FontName','Times New Roman')

subplot(2,2,4);
plot(time1_2,ylin1_4,'r',time1_2,yn1_4,'b--')
legend("Ground Truth","Estimation",'FontSize',8);
grid on
title('Yaw Rate, $\dot{\psi}$','FontSize',10,'FontName','Times New Roman','Interpreter','latex')
xlabel('Time[s]','FontSize',8)
ylabel('Velocity [rad/s]','FontSize',8)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name','validation gt1');
subplot(4,1,1);
plot(time1_2,ylin1_1,'r',time1_2,yn1_1,'b--')
legend("Ground Truth","Estimation",'FontSize',10);
grid on
title('Lateral position','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(4,1,2);
plot(time1_2,ylin1_2,'r',time1_2,yn1_2,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('Lateral velocity','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(4,1,3);
plot(time1_2,ylin1_3,'r',time1_2,yn1_3,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('yaw','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(4,1,4);
plot(time1_2,ylin1_4,'r',time1_2,yn1_4,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('yaw angle rate','FontSize',15)
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
sz2=size(time2_2);
% x2_0 = x2(1,:).';

x2_k = zeros(4,sz2(1)); 
% x2_k(:,1)=x2_0;
x2 = x2.';
input_2 = input_2.';

% gt2_rmse_1 = zeros(1,sz2(1));
% gt2_rmse_2 = zeros(1,sz2(1));
% gt2_rmse_3 = zeros(1,sz2(1));
% gt2_rmse_4 = zeros(1,sz2(1));

for k = 1:length(time2_1)
    x2_k(:,k) = evaluate(nss,x2(:,k),input_2(:,k)); 
%     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
%     gt2_rmse_1(1,k) = sqrt(mean((x_dot2(1:k,1).'-x2_k(1,1:k)).^2));
%     gt2_rmse_2(1,k) = sqrt(mean((x_dot2(1:k,2).'-x2_k(2,1:k)).^2));
%     gt2_rmse_3(1,k) = sqrt(mean((x_dot2(1:k,3).'-x2_k(3,1:k)).^2));
%     gt2_rmse_4(1,k) = sqrt(mean((x_dot2(1:k,4).'-x2_k(4,1:k)).^2));
    
end

% min_gt2_rmse_1 = min(gt2_rmse_1);
% max_gt2_rmse_1 = max(gt2_rmse_1);
% 
% min_gt2_rmse_2 = min(gt2_rmse_2);
% max_gt2_rmse_2 = max(gt2_rmse_2);
% 
% min_gt2_rmse_3 = min(gt2_rmse_3);
% max_gt2_rmse_3 = max(gt2_rmse_3);
% 
% min_gt2_rmse_4 = min(gt2_rmse_4);
% max_gt2_rmse_4 = max(gt2_rmse_4);
% for k = 1:length(time2_1)-1
%     x2_k(:,k+1) = evaluate(nss,x2(:,k),input_2(:,k)); 
% end

x2_k = x2_k.';

%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
ylin2_1 = x_dot2(:,1);
yn2_1 = x2_k(:,1);

ylin2_2 = x_dot2(:,2);
yn2_2 = x2_k(:,2);

ylin2_3 = x_dot2(:,3);
yn2_3 = x2_k(:,3);

ylin2_4 = x_dot2(:,4);
yn2_4 = x2_k(:,4);

%%%%%%%%%%%%%%%%%%%%gt2 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt2');
subplot(2,2,1);
plot(time2_2,ylin2_1,'r',time2_2,yn2_1,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin2_1-yn2_1).^2)))]},'FontSize',15)
title('Lateral position','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time2_2,ylin2_2,'r',time2_2,yn2_2,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin2_2-yn2_2).^2)))]},'FontSize',15)
title('Lateral velocity','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time2_2,ylin2_3,'r',time2_2,yn2_3,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin2_3-yn2_3).^2)))]},'FontSize',15)
title('yaw','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time2_2,ylin2_4,'r',time2_2,yn2_4,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin2_4-yn2_4).^2)))]},'FontSize',15)
title('yaw angle rate','FontSize',15)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sz3=size(time3_1);
% x3_0 = x3(1,:).';

x3_k = zeros(4,sz3(1)); 
% x3_k(:,1)=x3_0;
x3 = x3.';
input_3 = input_3.';

% gt3_rmse_1 = zeros(1,sz3(1));
% gt3_rmse_2 = zeros(1,sz3(1));
% gt3_rmse_3 = zeros(1,sz3(1));
% gt3_rmse_4 = zeros(1,sz3(1));

for k = 1:length(time3_1)
    x3_k(:,k) = evaluate(nss_gt34,x3(:,k),input_3(:,k)); 
%     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
%     gt3_rmse_1(1,k) = sqrt(mean((x_dot3(1:k,1).'-x3_k(1,1:k)).^2));
%     gt3_rmse_2(1,k) = sqrt(mean((x_dot3(1:k,2).'-x3_k(2,1:k)).^2));
%     gt3_rmse_3(1,k) = sqrt(mean((x_dot3(1:k,3).'-x3_k(3,1:k)).^2));
%     gt3_rmse_4(1,k) = sqrt(mean((x_dot3(1:k,4).'-x3_k(4,1:k)).^2));
%     
end

% min_gt3_rmse_1 = min(gt3_rmse_1);
% max_gt3_rmse_1 = max(gt3_rmse_1);
% 
% min_gt3_rmse_2 = min(gt3_rmse_2);
% max_gt3_rmse_2 = max(gt3_rmse_2);
% 
% min_gt3_rmse_3 = min(gt3_rmse_3);
% max_gt3_rmse_3 = max(gt3_rmse_3);
% 
% min_gt3_rmse_4 = min(gt3_rmse_4);
% max_gt3_rmse_4 = max(gt3_rmse_4);

% for k = 1:length(time3_1)-1
%     x3_k(:,k+1) = evaluate(nss,x3(:,k),input_3(:,k)); 
% end

x3_k = x3_k.';

%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
ylin3_1 = x_dot3(:,1);
yn3_1 = x3_k(:,1);

ylin3_2 = x_dot3(:,2);
yn3_2 = x3_k(:,2);

ylin3_3 = x_dot3(:,3);
yn3_3 = x3_k(:,3);

ylin3_4 = x_dot3(:,4);
yn3_4 = x3_k(:,4);

%%%%%%%%%%%%%%%%%%%%%%gt3 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt3');
subplot(2,2,1);
plot(time3_2,ylin3_1,'r',time3_2,yn3_1,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin3_1-yn3_1).^2)))]},'FontSize',15)
title('Lateral position','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time3_2,ylin3_2,'r',time3_2,yn3_2,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin3_2-yn3_2).^2)))]},'FontSize',15)
title('Lateral velocity','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time3_2,ylin3_3,'r',time3_2,yn3_3,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin3_3-yn3_3).^2)))]},'FontSize',15)
title('yaw','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time3_2,ylin3_4,'r',time3_2,yn3_4,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin3_4-yn3_4).^2)))]},'FontSize',15)
title('yaw angle rate','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

figure('Name','validation gt3');
subplot(4,1,1);
plot(time3_2,ylin3_1,'r',time3_2,yn3_1,'b--')
legend("Ground Truth","Estimation",'FontSize',10);
grid on
title('Lateral position','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(4,1,2);
plot(time3_2,ylin3_2,'r',time3_2,yn3_2,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('Lateral velocity','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(4,1,3);
plot(time3_2,ylin3_3,'r',time3_2,yn3_3,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('yaw','FontSize',15)
% xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(4,1,4);
plot(time3_2,ylin3_4,'r',time3_2,yn3_4,'b--')
% legend("Ground Truth","Estimation",'FontSize',15);
grid on
title('yaw angle rate','FontSize',15)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sz4=size(time4_1);
% x4_0 = x4(1,:).';

x4_k = zeros(4,sz4(1)); 
% x4_k(:,1)=x4_0;
x4 = x4.';
input_4 = input_4.';

% gt4_rmse_1 = zeros(1,sz4(1));
% gt4_rmse_2 = zeros(1,sz4(1));
% gt4_rmse_3 = zeros(1,sz4(1));
% gt4_rmse_4 = zeros(1,sz4(1));

for k = 1:length(time4_1)
    x4_k(:,k) = evaluate(nss,x4(:,k),input_4(:,k)); 
%     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
%     gt4_rmse_1(1,k) = sqrt(mean((x_dot4(1:k,1).'-x4_k(1,1:k)).^2));
%     gt4_rmse_2(1,k) = sqrt(mean((x_dot4(1:k,2).'-x4_k(2,1:k)).^2));
%     gt4_rmse_3(1,k) = sqrt(mean((x_dot4(1:k,3).'-x4_k(3,1:k)).^2));
%     gt4_rmse_4(1,k) = sqrt(mean((x_dot4(1:k,4).'-x4_k(4,1:k)).^2));
    
end

% min_gt4_rmse_1 = min(gt4_rmse_1);
% max_gt4_rmse_1 = max(gt4_rmse_1);
% 
% min_gt4_rmse_2 = min(gt4_rmse_2);
% max_gt4_rmse_2 = max(gt4_rmse_2);
% 
% min_gt4_rmse_3 = min(gt4_rmse_3);
% max_gt4_rmse_3 = max(gt4_rmse_3);
% 
% min_gt4_rmse_4 = min(gt4_rmse_4);
% max_gt4_rmse_4 = max(gt4_rmse_4);

% for k = 1:length(time4_1)-1
%     x4_k(:,k+1) = evaluate(nss,x4(:,k),input_4(:,k)); 
% end

x4_k = x4_k.';

%%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
ylin4_1 = x_dot4(:,1);
yn4_1 = x4_k(:,1);

ylin4_2 = x_dot4(:,2);
yn4_2 = x4_k(:,2);

ylin4_3 = x_dot4(:,3);
yn4_3 = x4_k(:,3);

ylin4_4 = x_dot4(:,4);
yn4_4 = x4_k(:,4);

%%%%%%%%%%%%%%%%%%%%%%gt4 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','validation gt4');
subplot(2,2,1);
plot(time4_2,ylin4_1,'r',time4_2,yn4_1,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin4_1-yn4_1).^2)))]},'FontSize',15)
title('Lateral position','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time4_2,ylin4_2,'r',time4_2,yn4_2,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin4_2-yn4_2).^2)))]},'FontSize',15)
title('Lateral velocity','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time4_2,ylin4_3,'r',time4_2,yn4_3,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin4_3-yn4_3).^2)))]},'FontSize',15)
title('yaw','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time4_2,ylin4_4,'r',time4_2,yn4_4,'b--')
legend("Ground Truth","Estimation",'FontSize',15);
grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin4_4-yn4_4).^2)))]},'FontSize',15)
title('yaw angle rate','FontSize',15)
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

%%%%%%%%%%%%%%%%%%%%%gt1 & gt2 & gt3 & gt4정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','Normal distribution gt1 & gt2 & gt3 & gt4');
subplot(2,2,1);
error1_1 = ylin1_1-yn1_1;
[V1_1,M1_1] = var(error1_1);
std1_1 = sqrt(V1_1);
normal_distribution1_1 = makedist('Normal', 'mu', M1_1, 'sigma', std1_1);
x1_1 = linspace(M1_1 - 3 * std1_1, M1_1 + 3 * std1_1, 1000); % 플롯을 위한 x 값 범위
y1_1 = pdf(normal_distribution1_1, x1_1); % 확률 밀도 함수 계산

% error2_1 = ylin2_1-yn2_1;
% [V2_1,M2_1] = var(error2_1);
% std2_1 = sqrt(V2_1);
% normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
% x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
% y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산

error3_1 = ylin3_1-yn3_1;
[V3_1,M3_1] = var(error3_1);
std3_1 = sqrt(V3_1);
normal_distribution3_1 = makedist('Normal', 'mu', M3_1, 'sigma', std3_1);
x3_1 = linspace(M3_1 - 3 * std3_1, M3_1 + 3 * std3_1, 1000); % 플롯을 위한 x 값 범위
y3_1 = pdf(normal_distribution3_1, x3_1); % 확률 밀도 함수 계산

% error4_1 = ylin4_1-yn4_1;
% [V4_1,M4_1] = var(error4_1);
% std4_1 = sqrt(V4_1);
% normal_distribution4_1 = makedist('Normal', 'mu', M4_1, 'sigma', std4_1);
% x4_1 = linspace(M4_1 - 3 * std4_1, M4_1 + 3 * std4_1, 1000); % 플롯을 위한 x 값 범위
% y4_1 = pdf(normal_distribution4_1, x4_1); % 확률 밀도 함수 계산

plot(x1_1, y1_1,'r--', 'LineWidth', 1);
hold on;
% plot(x2_1, y2_1,'g--', 'LineWidth', 1);
% hold on;
plot(x3_1, y3_1,'b--', 'LineWidth', 1);
% plot(x4_1, y4_1,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
legend("high-speed","K-city",'FontSize',15);
title('Normal distribution of Lateral position error','FontSize',15)
xlabel('error','FontSize',15);
ylabel('probability density','FontSize',15);
grid on;

subplot(2,2,2);
error1_2 = ylin1_2-yn1_2;
[V1_2,M1_2] = var(error1_2);
std1_2 = sqrt(V1_2);
normal_distribution1_2 = makedist('Normal', 'mu', M1_2, 'sigma', std1_2);
x1_2 = linspace(M1_2 - 3 * std1_2, M1_2 + 3 * std1_2, 1000); % 플롯을 위한 x 값 범위
y1_2 = pdf(normal_distribution1_2, x1_2); % 확률 밀도 함수 계산

% error2_2 = ylin2_2-yn2_2;
% [V2_2,M2_2] = var(error2_2);
% std2_2 = sqrt(V2_2);
% normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
% x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
% y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산

error3_2 = ylin3_2-yn3_2;
[V3_2,M3_2] = var(error3_2);
std3_2 = sqrt(V3_2);
normal_distribution3_2 = makedist('Normal', 'mu', M3_2, 'sigma', std3_2);
x3_2 = linspace(M3_2 - 3 * std3_2, M3_2 + 3 * std3_2, 1000); % 플롯을 위한 x 값 범위
y3_2 = pdf(normal_distribution3_2, x3_2); % 확률 밀도 함수 계산

% error4_2 = ylin4_2-yn4_2;
% [V4_2,M4_2] = var(error4_2);
% std4_2 = sqrt(V4_2);
% normal_distribution4_2 = makedist('Normal', 'mu', M4_2, 'sigma', std4_2);
% x4_2 = linspace(M4_2 - 3 * std4_2, M4_2 + 3 * std4_2, 1000); % 플롯을 위한 x 값 범위
% y4_2 = pdf(normal_distribution4_2, x4_2); % 확률 밀도 함수 계산

plot(x1_2, y1_2,'r--', 'LineWidth', 1);
hold on;
% plot(x2_2, y2_2,'g--', 'LineWidth', 1);
plot(x3_2, y3_2,'b--', 'LineWidth', 1);
% plot(x4_2, y4_2,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
legend("high-speed","K-city",'FontSize',15);
title('Normal distribution of Lateral velocity error','FontSize',15)
xlabel('error','FontSize',15);
ylabel('probability density','FontSize',15);
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

plot(x1_3, y1_3,'r--', 'LineWidth', 1);
hold on;
% plot(x2_3, y2_3,'g--', 'LineWidth', 1);
plot(x3_3, y3_3,'b--', 'LineWidth', 1);
% plot(x4_3, y4_3,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
legend("high-speed","K-city",'FontSize',15);
title('Normal distribution of yaw error','FontSize',15)
xlabel('error','FontSize',15);
ylabel('probability density','FontSize',15);
grid on;

subplot(2,2,4);
error1_4 = ylin1_4-yn1_4;
[V1_4,M1_4] = var(error1_4);
std1_4 = sqrt(V1_4);
normal_distribution1_4 = makedist('Normal', 'mu', M1_4, 'sigma', std1_4);
x1_4 = linspace(M1_4 - 3 * std1_4, M1_4 + 3 * std1_4, 1000); % 플롯을 위한 x 값 범위
y1_4 = pdf(normal_distribution1_4, x1_4); % 확률 밀도 함수 계산

% error2_4 = ylin2_4-yn2_4;
% [V2_4,M2_4] = var(error2_4);
% std2_4 = sqrt(V2_4);
% normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
% x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
% y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산

error3_4 = ylin3_4-yn3_4;
[V3_4,M3_4] = var(error3_4);
std3_4 = sqrt(V3_4);
normal_distribution3_4 = makedist('Normal', 'mu', M3_4, 'sigma', std3_4);
x3_4 = linspace(M3_4 - 3 * std3_4, M3_4 + 3 * std3_4, 1000); % 플롯을 위한 x 값 범위
y3_4 = pdf(normal_distribution3_4, x3_4); % 확률 밀도 함수 계산

% error4_4 = ylin4_4-yn4_4;
% [V4_4,M4_4] = var(error4_4);
% std4_4 = sqrt(V4_4);
% normal_distribution4_4 = makedist('Normal', 'mu', M4_4, 'sigma', std4_4);
% x4_4 = linspace(M4_4 - 3 * std4_4, M4_4 + 3 * std4_4, 1000); % 플롯을 위한 x 값 범위
% y4_4 = pdf(normal_distribution4_4, x4_4); % 확률 밀도 함수 계산

plot(x1_4, y1_4,'r--', 'LineWidth', 1);
hold on;
% plot(x2_4, y2_4,'g--', 'LineWidth', 1);
plot(x3_4, y3_4,'b--', 'LineWidth', 1);
% plot(x4_4, y4_4,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
legend("high-speed","K-city",'FontSize',15);
title('Normal distribution of yaw angle rate error','FontSize',15)
xlabel('error','FontSize',15);
ylabel('probability density','FontSize',15);
grid on;

rmse1_1=sqrt(mean((error1_1).^2));
% rmse2_1=sqrt(mean((error2_1).^2));
% mean_rmse12_1=mean([rmse1_1 rmse2_1]);

rmse3_1=sqrt(mean((error3_1).^2));
% rmse4_1=sqrt(mean((error4_1).^2));
% mean_rmse34_1=mean([rmse3_1 rmse4_1]);

rmse1_2=sqrt(mean((error1_2).^2));
% rmse2_2=sqrt(mean((error2_2).^2));
% mean_rmse12_2=mean([rmse1_2 rmse2_2]);

rmse3_2=sqrt(mean((error3_2).^2));
% rmse4_2=sqrt(mean((error4_2).^2));
% mean_rmse34_2=mean([rmse3_2 rmse4_2]);

rmse1_3=sqrt(mean((error1_3).^2));
% rmse2_3=sqrt(mean((error2_3).^2));
% mean_rmse12_3=mean([rmse1_3 rmse2_3]);

rmse3_3=sqrt(mean((error3_3).^2));
% rmse4_3=sqrt(mean((error4_3).^2));
% mean_rmse34_3=mean([rmse3_3 rmse4_3]);

rmse1_4=sqrt(mean((error1_4).^2));
% rmse2_4=sqrt(mean((error2_4).^2));
% mean_rmse12_4=mean([rmse1_4 rmse2_4]);

rmse3_4=sqrt(mean((error3_4).^2));
% rmse4_4=sqrt(mean((error4_4).^2));
% mean_rmse34_4=mean([rmse3_4 rmse4_4]);

% mean_std12_1=mean([std1_1 std2_1]);
% mean_std34_1=mean([std3_1 std4_1]);
% mean_std12_2=mean([std1_2 std2_2]);
% mean_std34_2=mean([std3_2 std4_2]);
% mean_std12_3=mean([std1_3 std2_3]);
% mean_std34_3=mean([std3_3 std4_3]);
% mean_std12_4=mean([std1_4 std2_4]);
% mean_std34_4=mean([std3_4 std4_4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%gt5%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz5=size(time5_1);
% % x5_0 = x5(1,:).';
% 
% x5_k = zeros(4,sz5(1)); 
% % x5_k(:,1)=x5_0;
% x5 = x5.';
% input_5 = input_5.';
% 
% % gt5_rmse_1 = zeros(1,sz5(1));
% % gt5_rmse_2 = zeros(1,sz5(1));
% % gt5_rmse_3 = zeros(1,sz5(1));
% % gt5_rmse_4 = zeros(1,sz5(1));
% 
% for k = 1:length(time5_1)
%     x5_k(:,k) = evaluate(nss,x5(:,k),input_5(:,k)); 
% %     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% %     gt5_rmse_1(1,k) = sqrt(mean((x_dot5(1:k,1).'-x5_k(1,1:k)).^2));
% %     gt5_rmse_2(1,k) = sqrt(mean((x_dot5(1:k,2).'-x5_k(2,1:k)).^2));
% %     gt5_rmse_3(1,k) = sqrt(mean((x_dot5(1:k,3).'-x5_k(3,1:k)).^2));
% %     gt5_rmse_4(1,k) = sqrt(mean((x_dot5(1:k,4).'-x5_k(4,1:k)).^2));
%     
% end
% 
% % min_gt5_rmse_1 = min(gt5_rmse_1);
% % max_gt5_rmse_1 = max(gt5_rmse_1);
% % 
% % min_gt5_rmse_2 = min(gt5_rmse_2);
% % max_gt5_rmse_2 = max(gt5_rmse_2);
% % 
% % min_gt5_rmse_3 = min(gt5_rmse_3);
% % max_gt5_rmse_3 = max(gt5_rmse_3);
% % 
% % min_gt5_rmse_4 = min(gt5_rmse_4);
% % max_gt5_rmse_4 = max(gt5_rmse_4);
% 
% % for k = 1:length(time5_1)-1
% %     x5_k(:,k+1) = evaluate(nss,x5(:,k),input_5(:,k)); 
% % end
% 
% x5_k = x5_k.';
% 
% ylin5_1 = x_dot5(:,1);
% yn5_1 = x5_k(:,1);
% 
% ylin5_2 = x_dot5(:,2);
% yn5_2 = x5_k(:,2);
% 
% ylin5_3 = x_dot5(:,3);
% yn5_3 = x5_k(:,3);
% 
% ylin5_4 = x_dot5(:,4);
% yn5_4 = x5_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt5 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt5');
% subplot(2,2,1);
% plot(time5_2,ylin5_1,'r',time5_2,yn5_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin5_1-yn5_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time5_2,ylin5_2,'r',time5_2,yn5_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin5_2-yn5_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time5_2,ylin5_3,'r',time5_2,yn5_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin5_3-yn5_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time5_2,ylin5_4,'r',time5_2,yn5_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin5_4-yn5_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt6%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz6=size(time6_1);
% % x6_0 = x6(1,:).';
% 
% x6_k = zeros(4,sz6(1)); 
% % x6_k(:,1)=x6_0;
% x6 = x6.';
% input_6 = input_6.';
% 
% % gt6_rmse_1 = zeros(1,sz6(1));
% % gt6_rmse_2 = zeros(1,sz6(1));
% % gt6_rmse_3 = zeros(1,sz6(1));
% % gt6_rmse_4 = zeros(1,sz6(1));
% 
% for k = 1:length(time6_1)
%     x6_k(:,k) = evaluate(nss,x6(:,k),input_6(:,k)); 
% %     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% %     gt6_rmse_1(1,k) = sqrt(mean((x_dot6(1:k,1).'-x6_k(1,1:k)).^2));
% %     gt6_rmse_2(1,k) = sqrt(mean((x_dot6(1:k,2).'-x6_k(2,1:k)).^2));
% %     gt6_rmse_3(1,k) = sqrt(mean((x_dot6(1:k,3).'-x6_k(3,1:k)).^2));
% %     gt6_rmse_4(1,k) = sqrt(mean((x_dot6(1:k,4).'-x6_k(4,1:k)).^2));
%     
% end
% 
% % min_gt6_rmse_1 = min(gt6_rmse_1);
% % max_gt6_rmse_1 = max(gt6_rmse_1);
% % 
% % min_gt6_rmse_2 = min(gt6_rmse_2);
% % max_gt6_rmse_2 = max(gt6_rmse_2);
% % 
% % min_gt6_rmse_3 = min(gt6_rmse_3);
% % max_gt6_rmse_3 = max(gt6_rmse_3);
% % 
% % min_gt6_rmse_4 = min(gt6_rmse_4);
% % max_gt6_rmse_4 = max(gt6_rmse_4);
% 
% % for k = 1:length(time6_1)-1
% %     x6_k(:,k+1) = evaluate(nss,x6(:,k),input_6(:,k)); 
% % end
% 
% x6_k = x6_k.';
% 
% ylin6_1 = x_dot6(:,1);
% yn6_1 = x6_k(:,1);
% 
% ylin6_2 = x_dot6(:,2);
% yn6_2 = x6_k(:,2);
% 
% ylin6_3 = x_dot6(:,3);
% yn6_3 = x6_k(:,3);
% 
% ylin6_4 = x_dot6(:,4);
% yn6_4 = x6_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt6 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt6');
% subplot(2,2,1);
% plot(time6_2,ylin6_1,'r',time6_2,yn6_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin6_1-yn6_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time6_2,ylin6_2,'r',time6_2,yn6_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin6_2-yn6_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time6_2,ylin6_3,'r',time6_2,yn6_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin6_3-yn6_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time6_2,ylin6_4,'r',time6_2,yn6_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin6_4-yn6_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt7%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz7=size(time7_1);
% % x7_0 = x7(1,:).';
% 
% x7_k = zeros(4,sz7(1)); 
% % x7_k(:,1)=x7_0;
% x7 = x7.';
% input_7 = input_7.';
% 
% % gt7_rmse_1 = zeros(1,sz7(1));
% % gt7_rmse_2 = zeros(1,sz7(1));
% % gt7_rmse_3 = zeros(1,sz7(1));
% % gt7_rmse_4 = zeros(1,sz7(1));
% 
% for k = 1:length(time7_1)
%     x7_k(:,k) = evaluate(nss,x7(:,k),input_7(:,k)); 
% %     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% %     gt7_rmse_1(1,k) = sqrt(mean((x_dot7(1:k,1).'-x7_k(1,1:k)).^2));
% %     gt7_rmse_2(1,k) = sqrt(mean((x_dot7(1:k,2).'-x7_k(2,1:k)).^2));
% %     gt7_rmse_3(1,k) = sqrt(mean((x_dot7(1:k,3).'-x7_k(3,1:k)).^2));
% %     gt7_rmse_4(1,k) = sqrt(mean((x_dot7(1:k,4).'-x7_k(4,1:k)).^2));
%     
% end
% 
% % min_gt7_rmse_1 = min(gt7_rmse_1);
% % max_gt7_rmse_1 = max(gt7_rmse_1);
% % 
% % min_gt7_rmse_2 = min(gt7_rmse_2);
% % max_gt7_rmse_2 = max(gt7_rmse_2);
% % 
% % min_gt7_rmse_3 = min(gt7_rmse_3);
% % max_gt7_rmse_3 = max(gt7_rmse_3);
% % 
% % min_gt7_rmse_4 = min(gt7_rmse_4);
% % max_gt7_rmse_4 = max(gt7_rmse_4);
% 
% % for k = 1:length(time7_1)-1
% %     x7_k(:,k+1) = evaluate(nss,x7(:,k),input_7(:,k)); 
% % end
% 
% x7_k = x7_k.';
% 
% ylin7_1 = x_dot7(:,1);
% yn7_1 = x7_k(:,1);
% 
% ylin7_2 = x_dot7(:,2);
% yn7_2 = x7_k(:,2);
% 
% ylin7_3 = x_dot7(:,3);
% yn7_3 = x7_k(:,3);
% 
% ylin7_4 = x_dot7(:,4);
% yn7_4 = x7_k(:,4);
% % 
% %%%%%%%%%%%%%%%%%%%%%%gt7 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt7');
% subplot(2,2,1);
% plot(time7_2,ylin7_1,'r',time7_2,yn7_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin7_1-yn7_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time7_2,ylin7_2,'r',time7_2,yn7_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin7_2-yn7_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time7_2,ylin7_3,'r',time7_2,yn7_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin7_3-yn7_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time7_2,ylin7_4,'r',time7_2,yn7_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin7_4-yn7_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt8%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz8=size(time8_1);
% % x8_0 = x8(1,:).';
% 
% x8_k = zeros(4,sz8(1)); 
% % x8_k(:,1)=x8_0;
% x8 = x8.';
% input_8 = input_8.';
% 
% % gt8_rmse_1 = zeros(1,sz8(1));
% % gt8_rmse_2 = zeros(1,sz8(1));
% % gt8_rmse_3 = zeros(1,sz8(1));
% % gt8_rmse_4 = zeros(1,sz8(1));
% 
% for k = 1:length(time8_1)
%     x8_k(:,k) = evaluate(nss,x8(:,k),input_8(:,k)); 
% %     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% %     gt8_rmse_1(1,k) = sqrt(mean((x_dot8(1:k,1).'-x8_k(1,1:k)).^2));
% %     gt8_rmse_2(1,k) = sqrt(mean((x_dot8(1:k,2).'-x8_k(2,1:k)).^2));
% %     gt8_rmse_3(1,k) = sqrt(mean((x_dot8(1:k,3).'-x8_k(3,1:k)).^2));
% %     gt8_rmse_4(1,k) = sqrt(mean((x_dot8(1:k,4).'-x8_k(4,1:k)).^2));
%     
% end
% 
% % min_gt8_rmse_1 = min(gt8_rmse_1);
% % max_gt8_rmse_1 = max(gt8_rmse_1);
% % 
% % min_gt8_rmse_2 = min(gt8_rmse_2);
% % max_gt8_rmse_2 = max(gt8_rmse_2);
% % 
% % min_gt8_rmse_3 = min(gt8_rmse_3);
% % max_gt8_rmse_3 = max(gt8_rmse_3);
% % 
% % min_gt8_rmse_4 = min(gt8_rmse_4);
% % max_gt8_rmse_4 = max(gt8_rmse_4);
% 
% % for k = 1:length(time8_1)-1
% %     x8_k(:,k+1) = evaluate(nss,x8(:,k),input_8(:,k)); 
% % end
% 
% x8_k = x8_k.';
% 
% ylin8_1 = x_dot8(:,1);
% yn8_1 = x8_k(:,1);
% 
% ylin8_2 = x_dot8(:,2);
% yn8_2 = x8_k(:,2);
% 
% ylin8_3 = x_dot8(:,3);
% yn8_3 = x8_k(:,3);
% 
% ylin8_4 = x_dot8(:,4);
% yn8_4 = x8_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt8 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt8');
% subplot(2,2,1);
% plot(time8_2,ylin8_1,'r',time8_2,yn8_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin8_1-yn8_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time8_2,ylin8_2,'r',time8_2,yn8_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin8_2-yn8_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time8_2,ylin8_3,'r',time8_2,yn8_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin8_3-yn8_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time8_2,ylin8_4,'r',time8_2,yn8_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin8_4-yn8_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt9%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz9=size(time9_1);
% % x9_0 = x9(1,:).';
% 
% x9_k = zeros(4,sz9(1)); 
% % x9_k(:,1)=x9_0;
% x9 = x9.';
% input_9 = input_9.';
% 
% % gt9_rmse_1 = zeros(1,sz9(1));
% % gt9_rmse_2 = zeros(1,sz9(1));
% % gt9_rmse_3 = zeros(1,sz9(1));
% % gt9_rmse_4 = zeros(1,sz9(1));
% 
% for k = 1:length(time9_1)
%     x9_k(:,k) = evaluate(nss,x9(:,k),input_9(:,k)); 
% %     x1_k(:,k+1) = evaluate(nss,x1(:,k),input_1(:,k)); 
% %     gt9_rmse_1(1,k) = sqrt(mean((x_dot9(1:k,1).'-x9_k(1,1:k)).^2));
% %     gt9_rmse_2(1,k) = sqrt(mean((x_dot9(1:k,2).'-x9_k(2,1:k)).^2));
% %     gt9_rmse_3(1,k) = sqrt(mean((x_dot9(1:k,3).'-x9_k(3,1:k)).^2));
% %     gt9_rmse_4(1,k) = sqrt(mean((x_dot9(1:k,4).'-x9_k(4,1:k)).^2));
%     
% end
% 
% % min_gt9_rmse_1 = min(gt9_rmse_1);
% % max_gt9_rmse_1 = max(gt9_rmse_1);
% % 
% % min_gt9_rmse_2 = min(gt9_rmse_2);
% % max_gt9_rmse_2 = max(gt9_rmse_2);
% % 
% % min_gt9_rmse_3 = min(gt9_rmse_3);
% % max_gt9_rmse_3 = max(gt9_rmse_3);
% % 
% % min_gt9_rmse_4 = min(gt9_rmse_4);
% % max_gt9_rmse_4 = max(gt9_rmse_4);
% 
% % for k = 1:length(time9_1)-1
% %     x9_k(:,k+1) = evaluate(nss,x9(:,k),input_9(:,k)); 
% % end
% 
% x9_k = x9_k.';
% 
% ylin9_1 = x_dot9(:,1);
% yn9_1 = x9_k(:,1);
% 
% ylin9_2 = x_dot9(:,2);
% yn9_2 = x9_k(:,2);
% 
% ylin9_3 = x_dot9(:,3);
% yn9_3 = x9_k(:,3);
% 
% ylin9_4 = x_dot9(:,4);
% yn9_4 = x9_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt9 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt9');
% subplot(2,2,1);
% plot(time9_2,ylin9_1,'r',time9_2,yn9_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin9_1-yn9_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time9_2,ylin9_2,'r',time9_2,yn9_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin9_2-yn9_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time9_2,ylin9_3,'r',time9_2,yn9_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin9_3-yn9_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time9_2,ylin9_4,'r',time9_2,yn9_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin9_4-yn9_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%gt10%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % sz10=size(time10_1);
% % x10_0 = x10(1,:).';
% % 
% % x10_k = zeros(4,sz10(1)); 
% % x10_k(:,1)=x10_0;
% % x10 = x10.';
% % input_10 = input_10.';
% % 
% % for k = 1:length(time10_1)-1
% %     x10_k(:,k+1) = evaluate(nss,x10(:,k),input_10(:,k)); 
% % end
% % 
% % x10_k = x10_k.';
% % 
% % ylin10_1 = x_dot10(:,1);
% % yn10_1 = x10_k(:,1);
% % 
% % ylin10_2 = x_dot10(:,2);
% % yn10_2 = x10_k(:,2);
% % 
% % ylin10_3 = x_dot10(:,3);
% % yn10_3 = x10_k(:,3);
% % 
% % ylin10_4 = x_dot10(:,4);
% % yn10_4 = x10_k(:,4);
% % 
% % %%%%%%%%%%%%%%%%%%%%%%gt10 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure('Name','validation gt10');
% % subplot(2,2,1);
% % plot(time10_1,ylin10_1,'r',time10_1,yn10_1,'b--')
% % legend("Original","Estimated",'FontSize',15);
% % grid on
% % title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin10_1-yn10_1).^2)))]},'FontSize',15)
% % xlabel('Time[s]','FontSize',15)
% % ylabel('y [m]','FontSize',15)
% % 
% % subplot(2,2,2);
% % plot(time10_1,ylin10_2,'r',time10_1,yn10_2,'b--')
% % legend("Original","Estimated",'FontSize',15);
% % grid on
% % title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin10_2-yn10_2).^2)))]},'FontSize',15)
% % xlabel('Time[s]','FontSize',15)
% % ylabel('dy/dt [m/s]','FontSize',15)
% % 
% % subplot(2,2,3);
% % plot(time10_1,ylin10_3,'r',time10_1,yn10_3,'b--')
% % legend("Original","Estimated",'FontSize',15);
% % grid on
% % title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin10_3-yn10_3).^2)))]},'FontSize',15)
% % xlabel('Time[s]','FontSize',15)
% % ylabel('ψ [rad]','FontSize',15)
% % 
% % subplot(2,2,4);
% % plot(time10_1,ylin10_4,'r',time10_1,yn10_4,'b--')
% % legend("Original","Estimated",'FontSize',15);
% % grid on
% % title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin10_4-yn10_4).^2)))]},'FontSize',15)
% % xlabel('Time[s]','FontSize',15)
% % ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% %%%%%%%%%%%%%%%%%%%%gt5 & gt6 & gt7 & gt8 & gt9 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution gt5 & gt6 & gt7 & gt8 & gt9');
% subplot(2,2,1);
% error5_1 = ylin5_1-yn5_1;
% [V5_1,M5_1] = var(error5_1);
% std5_1 = sqrt(V5_1);
% normal_distribution5_1 = makedist('Normal', 'mu', M5_1, 'sigma', std5_1);
% x5_1 = linspace(M5_1 - 3 * std5_1, M5_1 + 3 * std5_1, 1000); % 플롯을 위한 x 값 범위
% y5_1 = pdf(normal_distribution5_1, x5_1); % 확률 밀도 함수 계산
% 
% error6_1 = ylin6_1-yn6_1;
% [V6_1,M6_1] = var(error6_1);
% std6_1 = sqrt(V6_1);
% normal_distribution6_1 = makedist('Normal', 'mu', M6_1, 'sigma', std6_1);
% x6_1 = linspace(M6_1 - 3 * std6_1, M6_1 + 3 * std6_1, 1000); % 플롯을 위한 x 값 범위
% y6_1 = pdf(normal_distribution6_1, x6_1); % 확률 밀도 함수 계산
% 
% error7_1 = ylin7_1-yn7_1;
% [V7_1,M7_1] = var(error7_1);
% std7_1 = sqrt(V7_1);
% normal_distribution7_1 = makedist('Normal', 'mu', M7_1, 'sigma', std7_1);
% x7_1 = linspace(M7_1 - 3 * std7_1, M7_1 + 3 * std7_1, 1000); % 플롯을 위한 x 값 범위
% y7_1 = pdf(normal_distribution7_1, x7_1); % 확률 밀도 함수 계산
% 
% error8_1 = ylin8_1-yn8_1;
% [V8_1,M8_1] = var(error8_1);
% std8_1 = sqrt(V8_1);
% normal_distribution8_1 = makedist('Normal', 'mu', M8_1, 'sigma', std8_1);
% x8_1 = linspace(M8_1 - 3 * std8_1, M8_1 + 3 * std8_1, 1000); % 플롯을 위한 x 값 범위
% y8_1 = pdf(normal_distribution8_1, x8_1); % 확률 밀도 함수 계산
% 
% error9_1 = ylin9_1-yn9_1;
% [V9_1,M9_1] = var(error9_1);
% std9_1 = sqrt(V9_1);
% normal_distribution9_1 = makedist('Normal', 'mu', M9_1, 'sigma', std9_1);
% x9_1 = linspace(M9_1 - 3 * std9_1, M9_1 + 3 * std9_1, 1000); % 플롯을 위한 x 값 범위
% y9_1 = pdf(normal_distribution9_1, x9_1); % 확률 밀도 함수 계산
% 
% plot(x5_1, y5_1,'r--', 'LineWidth', 1);
% hold on;
% plot(x6_1, y6_1,'g--', 'LineWidth', 1);
% hold on;
% plot(x7_1, y7_1,'b--', 'LineWidth', 1);
% hold on;
% plot(x8_1, y8_1,'c--', 'LineWidth', 1);
% hold on;
% plot(x9_1, y9_1,'k--', 'LineWidth', 1);
% legend("gt5(training)","gt6(training)","gt7(training)","gt8","gt9");
% title('Normal distribution of Lateral position error','FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2);
% error5_2 = ylin5_2-yn5_2;
% [V5_2,M5_2] = var(error5_2);
% std5_2 = sqrt(V5_2);
% normal_distribution5_2 = makedist('Normal', 'mu', M5_2, 'sigma', std5_2);
% x5_2 = linspace(M5_2 - 3 * std5_2, M5_2 + 3 * std5_2, 1000); % 플롯을 위한 x 값 범위
% y5_2 = pdf(normal_distribution5_2, x5_2); % 확률 밀도 함수 계산
% 
% error6_2 = ylin6_2-yn6_2;
% [V6_2,M6_2] = var(error6_2);
% std6_2 = sqrt(V6_2);
% normal_distribution6_2 = makedist('Normal', 'mu', M6_2, 'sigma', std6_2);
% x6_2 = linspace(M6_2 - 3 * std6_2, M6_2 + 3 * std6_2, 1000); % 플롯을 위한 x 값 범위
% y6_2 = pdf(normal_distribution6_2, x6_2); % 확률 밀도 함수 계산
% 
% error7_2 = ylin7_2-yn7_2;
% [V7_2,M7_2] = var(error7_2);
% std7_2 = sqrt(V7_2);
% normal_distribution7_2 = makedist('Normal', 'mu', M7_2, 'sigma', std7_2);
% x7_2 = linspace(M7_2 - 3 * std7_2, M7_2 + 3 * std7_2, 1000); % 플롯을 위한 x 값 범위
% y7_2 = pdf(normal_distribution7_2, x7_2); % 확률 밀도 함수 계산
% 
% error8_2 = ylin8_2-yn8_2;
% [V8_2,M8_2] = var(error8_2);
% std8_2 = sqrt(V8_2);
% normal_distribution8_2 = makedist('Normal', 'mu', M8_2, 'sigma', std8_2);
% x8_2 = linspace(M8_2 - 3 * std8_2, M8_2 + 3 * std8_2, 1000); % 플롯을 위한 x 값 범위
% y8_2 = pdf(normal_distribution8_2, x8_2); % 확률 밀도 함수 계산
% 
% 
% error9_2 = ylin9_2-yn9_2;
% [V9_2,M9_2] = var(error9_2);
% std9_2 = sqrt(V9_2);
% normal_distribution9_2 = makedist('Normal', 'mu', M9_2, 'sigma', std9_2);
% x9_2 = linspace(M9_2 - 3 * std9_2, M9_2 + 3 * std9_2, 1000); % 플롯을 위한 x 값 범위
% y9_2 = pdf(normal_distribution9_2, x9_2); % 확률 밀도 함수 계산
% 
% plot(x5_2, y5_2,'r--', 'LineWidth', 1);
% hold on;
% plot(x6_2, y6_2,'g--', 'LineWidth', 1);
% hold on;
% plot(x7_2, y7_2,'b--', 'LineWidth', 1);
% hold on;
% plot(x8_2, y8_2,'c--', 'LineWidth', 1);
% hold on;
% plot(x9_2, y9_2,'k--', 'LineWidth', 1);
% legend("gt5(training)","gt6(training)","gt7(training)","gt8","gt9");
% title('Normal distribution of Lateral velocity error','FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3);
% error5_3 = ylin5_3-yn5_3;
% [V5_3,M5_3] = var(error5_3);
% std5_3 = sqrt(V5_3);
% normal_distribution5_3 = makedist('Normal', 'mu', M5_3, 'sigma', std5_3);
% x5_3 = linspace(M5_3 - 3 * std5_3, M5_3 + 3 * std5_3, 1000); % 플롯을 위한 x 값 범위
% y5_3 = pdf(normal_distribution5_3, x5_3); % 확률 밀도 함수 계산
% 
% error6_3 = ylin6_3-yn6_3;
% [V6_3,M6_3] = var(error6_3);
% std6_3 = sqrt(V6_3);
% normal_distribution6_3 = makedist('Normal', 'mu', M6_3, 'sigma', std6_3);
% x6_3 = linspace(M6_3 - 3 * std6_3, M6_3 + 3 * std6_3, 1000); % 플롯을 위한 x 값 범위
% y6_3 = pdf(normal_distribution6_3, x6_3); % 확률 밀도 함수 계산
% 
% error7_3 = ylin7_3-yn7_3;
% [V7_3,M7_3] = var(error7_3);
% std7_3 = sqrt(V7_3);
% normal_distribution7_3 = makedist('Normal', 'mu', M7_3, 'sigma', std7_3);
% x7_3 = linspace(M7_3 - 3 * std7_3, M7_3 + 3 * std7_3, 1000); % 플롯을 위한 x 값 범위
% y7_3= pdf(normal_distribution7_3, x7_3); % 확률 밀도 함수 계산
% 
% error8_3 = ylin8_3-yn8_3;
% [V8_3,M8_3] = var(error8_3);
% std8_3 = sqrt(V8_3);
% normal_distribution8_3 = makedist('Normal', 'mu', M8_3, 'sigma', std8_3);
% x8_3 = linspace(M8_3 - 3 * std8_3, M8_3 + 3 * std8_3, 1000); % 플롯을 위한 x 값 범위
% y8_3 = pdf(normal_distribution8_3, x8_3); % 확률 밀도 함수 계산
% 
% error9_3 = ylin9_3-yn9_3;
% [V9_3,M9_3] = var(error9_3);
% std9_3 = sqrt(V9_3);
% normal_distribution9_3 = makedist('Normal', 'mu', M9_3, 'sigma', std9_3);
% x9_3 = linspace(M9_3 - 3 * std9_3, M9_3 + 3 * std9_3, 1000); % 플롯을 위한 x 값 범위
% y9_3 = pdf(normal_distribution9_3, x9_3); % 확률 밀도 함수 계산
% 
% plot(x5_3, y5_3,'r--', 'LineWidth', 1);
% hold on;
% plot(x6_3, y6_3,'g--', 'LineWidth', 1);
% hold on;
% plot(x7_3, y7_3,'b--', 'LineWidth', 1);
% hold on;
% plot(x8_3, y8_3,'c--', 'LineWidth', 1);
% hold on;
% plot(x9_3, y9_3,'k--', 'LineWidth', 1);
% legend("gt5(training)","gt6(training)","gt7(training)","gt8","gt9");
% title('Normal distribution of yaw error','FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4);
% error5_4 = ylin5_4-yn5_4;
% [V5_4,M5_4] = var(error5_4);
% std5_4 = sqrt(V5_4);
% normal_distribution5_4 = makedist('Normal', 'mu', M5_4, 'sigma', std5_4);
% x5_4 = linspace(M5_4 - 3 * std5_4, M5_4 + 3 * std5_4, 1000); % 플롯을 위한 x 값 범위
% y5_4 = pdf(normal_distribution5_4, x5_4); % 확률 밀도 함수 계산
% 
% error6_4 = ylin6_4-yn6_4;
% [V6_4,M6_4] = var(error6_4);
% std6_4 = sqrt(V6_4);
% normal_distribution6_4 = makedist('Normal', 'mu', M6_4, 'sigma', std6_4);
% x6_4 = linspace(M6_4 - 3 * std6_4, M6_4 + 3 * std6_4, 1000); % 플롯을 위한 x 값 범위
% y6_4 = pdf(normal_distribution6_4, x6_4); % 확률 밀도 함수 계산
% 
% error7_4 = ylin7_4-yn7_4;
% [V7_4,M7_4] = var(error7_4);
% std7_4 = sqrt(V7_4);
% normal_distribution7_4 = makedist('Normal', 'mu', M7_4, 'sigma', std7_4);
% x7_4 = linspace(M7_4 - 3 * std7_4, M7_4 + 3 * std7_4, 1000); % 플롯을 위한 x 값 범위
% y7_4 = pdf(normal_distribution7_4, x7_4); % 확률 밀도 함수 계산
% 
% error8_4 = ylin8_4-yn8_4;
% [V8_4,M8_4] = var(error8_4);
% std8_4 = sqrt(V8_4);
% normal_distribution8_4 = makedist('Normal', 'mu', M8_4, 'sigma', std8_4);
% x8_4 = linspace(M8_4 - 3 * std8_4, M8_4 + 3 * std8_4, 1000); % 플롯을 위한 x 값 범위
% y8_4 = pdf(normal_distribution8_4, x8_4); % 확률 밀도 함수 계산
% 
% error9_4 = ylin9_4-yn9_4;
% [V9_4,M9_4] = var(error9_4);
% std9_4 = sqrt(V9_4);
% normal_distribution9_4 = makedist('Normal', 'mu', M9_4, 'sigma', std9_4);
% x9_4 = linspace(M9_4 - 3 * std9_4, M9_4 + 3 * std9_4, 1000); % 플롯을 위한 x 값 범위
% y9_4 = pdf(normal_distribution9_4, x9_4); % 확률 밀도 함수 계산
% 
% plot(x5_4, y5_4,'r--', 'LineWidth', 1);
% hold on;
% plot(x6_4, y6_4,'g--', 'LineWidth', 1);
% hold on;
% plot(x7_4, y7_4,'b--', 'LineWidth', 1);
% hold on;
% plot(x8_4, y8_4,'c--', 'LineWidth', 1);
% hold on;
% plot(x9_4, y9_4,'k--', 'LineWidth', 1);
% legend("gt5(training)","gt6(training)","gt7(training)","gt8","gt9");
% title('Normal distribution of yaw angle rate error','FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% rmse5_1=sqrt(mean((error5_1).^2));
% rmse6_1=sqrt(mean((error6_1).^2));
% rmse7_1=sqrt(mean((error7_1).^2));
% rmse8_1=sqrt(mean((error8_1).^2));
% rmse9_1=sqrt(mean((error9_1).^2));
% mean_rmse567_1=mean([rmse5_1 rmse6_1 rmse7_1]);
% mean_rmse89_1=mean([rmse8_1 rmse9_1]);
% 
% rmse5_2=sqrt(mean((error5_2).^2));
% rmse6_2=sqrt(mean((error6_2).^2));
% rmse7_2=sqrt(mean((error7_2).^2));
% rmse8_2=sqrt(mean((error8_2).^2));
% rmse9_2=sqrt(mean((error9_2).^2));
% mean_rmse567_2=mean([rmse5_2 rmse6_2 rmse7_2]);
% mean_rmse89_2=mean([rmse8_2 rmse9_2]);
% 
% rmse5_3=sqrt(mean((error5_3).^2));
% rmse6_3=sqrt(mean((error6_3).^2));
% rmse7_3=sqrt(mean((error7_3).^2));
% rmse8_3=sqrt(mean((error8_3).^2));
% rmse9_3=sqrt(mean((error9_3).^2));
% mean_rmse567_3=mean([rmse5_3 rmse6_3 rmse7_3]);
% mean_rmse89_3=mean([rmse8_3 rmse9_3]);
% 
% rmse5_4=sqrt(mean((error5_4).^2));
% rmse6_4=sqrt(mean((error6_4).^2));
% rmse7_4=sqrt(mean((error7_4).^2));
% rmse8_4=sqrt(mean((error8_4).^2));
% rmse9_4=sqrt(mean((error9_4).^2));
% mean_rmse567_4=mean([rmse5_4 rmse6_4 rmse7_4]);
% mean_rmse89_4=mean([rmse8_4 rmse9_4]);
% 
% mean_std567_1=mean([std5_1 std6_1 std7_1]);
% mean_std89_1=mean([std8_1 std9_1]);
% mean_std567_2=mean([std5_2 std6_2 std7_2]);
% mean_std89_2=mean([std8_2 std9_2]);
% mean_std567_3=mean([std5_3 std6_3 std7_3]);
% mean_std89_3=mean([std8_3 std9_3]);
% mean_std567_4=mean([std5_4 std6_4 std7_4]);
% mean_std89_4=mean([std8_4 std9_4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%IEEE IV conference%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gt1 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
gt3 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City/
IEEEIVarticle_modelvalidation_VelxWsaAnal(gt1,gt3,input1_data,input3_data)