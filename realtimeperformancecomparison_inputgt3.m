close all; clear; clc;

%각각 다른 초기 상태에서 시작하여 1초 동안 지속되는 1000개의 시뮬레이션을 실행한다. 
%각 실험은 동일한 시점을 사용해야 한다.

run("anal.m");

%Neural Network model 불러오기
% load("newgt1plusgt2plusinverse_32168_nss_delayed.mat");
% load("newgt3plusgt4plusinverse_32168_nss_delayed.mat");
% load("newgt5plusgt6plusgt7plusinverse_32168_nss_delayed.mat");
% load("newgt8plusgt9plusinverse_32168_nss_delayed.mat");
load("noinverse_gtattyaw_gt1plusgt2_32168_nss_delayed.mat");
nss1=nss;
load("noinverse_gtattyaw_gt3plusgt4_32168_nss_delayed.mat");
nss2=nss;


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

% % %%%%%%%%%%%%%%%%%%gt3을 두 모델에 입력으로 넣음%%%%%%%%%%%%%%%%%%%%%%%%
sz1=size(time3_2);
sz3=size(time3_2);
x3 = x3.';
input_3 = input_3.';

% sz1=size(time2_2);
% sz3=size(time2_2);
% x2 = x2.';
% input_2 = input_2.';

% sz1=size(time3_2);
% sz3=size(time3_2);
% x3 = x3.';
% input_3 = input_3.';

% sz1=size(time4_2);
% sz3=size(time4_2);
% x4 = x4.';
% input_4 = input_4.';

x1_k = zeros(4,sz1(1)); 
x3_k = zeros(4,sz3(1)); 

% %100샘플씩 예측할때 
error_1 = zeros(100,4);   
error_3 = zeros(100,4); 

%700샘플씩 예측할때 
% error_1 = zeros(500,4);   
% error_3 = zeros(500,4); 


figure('Name','gt3 realtime normaldistribution')
subplot(2,2,1)
M_1_1=0;
M_3_1=0;
std_1_1=0;
std_3_1=0;
normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
normal_distribution_3_1 = makedist('Normal', 'mu', M_3_1, 'sigma', std_3_1);
x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
x_3_1 = linspace(M_3_1 - 3 * std_3_1, M_3_1 + 3 * std_3_1, 1000); % 플롯을 위한 x 값 범위
y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
y_3_1 = pdf(normal_distribution_3_1, x_3_1); % 확률 밀도 함수 계산
p_1_1=plot(x_1_1, y_1_1,'r', 'LineWidth',2);
hold on;
p_3_1=plot(x_3_1, y_3_1,'k','LineWidth',2);
xline_1_1=xline(M_1_1, 'r--');
xline_3_1=xline(M_3_1, 'k--');
dim_1 = [0.3 0.4 0.4 0.4];
str_1 = {'Initial State'};
a_1=annotation('textbox',dim_1,'String',str_1,'FitBoxToText','on');
title_1_1=title({'initial Normal distribution of Lateral position error';['mean1 = ' num2str(M_1_1) ', std1 = ' num2str(std_1_1)];['mean3 = ' num2str(M_3_1) ', std3 = ' num2str(std_3_1)]},'FontSize',15);
xlabel('error');
ylabel('probability density');
legend('고속주회로 환경 학습 모델','K-City 환경 학습 모델')
grid on;

subplot(2,2,2)
M_1_2=0;
M_3_2=0;
std_1_2=0;
std_3_2=0;
normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
normal_distribution_3_2 = makedist('Normal', 'mu', M_3_2, 'sigma', std_3_2);
x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
x_3_2 = linspace(M_3_2 - 3 * std_3_2, M_3_2 + 3 * std_3_2, 1000); % 플롯을 위한 x 값 범위
y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
y_3_2 = pdf(normal_distribution_3_2, x_3_2); % 확률 밀도 함수 계산
p_1_2=plot(x_1_2, y_1_2,'r', 'LineWidth', 2);
hold on;
p_3_2=plot(x_3_2, y_3_2,'k', 'LineWidth', 2);
xline_1_2=xline(M_1_2, 'r--');
xline_3_2=xline(M_3_2, 'k--');
dim_2 = [0.7 0.4 0.4 0.4];
str_2 = {'Initial State'};
a_2=annotation('textbox',dim_2,'String',str_2,'FitBoxToText','on');
title_1_2=title({'initial Normal distribution of Lateral velocity error';['mean1= ' num2str(M_1_2) ', std1= ' num2str(std_1_2)];['mean3= ' num2str(M_3_2) ', std3= ' num2str(std_3_2)]},'FontSize',15);
xlabel('error');
ylabel('probability density');
legend('고속주회로 환경 학습 모델','K-City 환경 학습 모델')
grid on;

subplot(2,2,3)
M_1_3=0;
M_3_3=0;
std_1_3=0;
std_3_3=0;
normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
normal_distribution_3_3 = makedist('Normal', 'mu', M_3_3, 'sigma', std_3_3);
x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
x_3_3 = linspace(M_3_3 - 3 * std_3_3, M_3_3 + 3 * std_3_3, 1000); % 플롯을 위한 x 값 범위
y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
y_3_3 = pdf(normal_distribution_3_3, x_3_3); % 확률 밀도 함수 계산
p_1_3=plot(x_1_3, y_1_3,'r', 'LineWidth', 2);
hold on;
p_3_3=plot(x_3_3, y_3_3,'k', 'LineWidth', 2);
xline_1_3=xline(M_1_3, 'r--');
xline_3_3=xline(M_3_3, 'k--');
dim_3 = [0.3 0.008 0.4 0.4];
str_3 = {'Initial State'};
a_3=annotation('textbox',dim_3,'String',str_3,'FitBoxToText','on');
title_1_3=title({'initial Normal distribution of Yaw error';['mean1= ' num2str(M_1_3) ', std1= ' num2str(std_1_3)];['mean3= ' num2str(M_3_3) ', std3= ' num2str(std_3_3)]},'FontSize',15);
xlabel('error');
ylabel('probability density');
legend('고속주회로 환경 학습 모델','K-City 환경 학습 모델')
grid on;

subplot(2,2,4)
M_1_4=0;
M_3_4=0;
std_1_4=0;
std_3_4=0;
normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
normal_distribution_3_4 = makedist('Normal', 'mu', M_3_4, 'sigma', std_3_4);
x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
x_3_4 = linspace(M_3_4 - 3 * std_3_4, M_3_4 + 3 * std_3_4, 1000); % 플롯을 위한 x 값 범위
y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
y_3_4 = pdf(normal_distribution_3_4, x_3_4); % 확률 밀도 함수 계산
p_1_4=plot(x_1_4, y_1_4,'r', 'LineWidth', 2);
hold on;
p_3_4=plot(x_3_4, y_3_4,'k', 'LineWidth', 2);
xline_1_4=xline(M_1_4, 'r--');
xline_3_4=xline(M_3_4, 'k--');
dim_4 = [0.7 0.008 0.4 0.4];
str_4 = {'Initial State'};
a_4=annotation('textbox',dim_4,'String',str_4,'FitBoxToText','on');
title_1_4=title({'initial Normal distribution of Yaw rate error';['mean1= ' num2str(M_1_4) ', std1= ' num2str(std_1_4)];['mean3= ' num2str(M_3_4) ', std1= ' num2str(std_3_4)]},'FontSize',15);
xlabel('error');
ylabel('probability density');
legend('고속주회로 환경 학습 모델','K-City 환경 학습 모델')
grid on;

%k=100일 때부터 k가 100단위씩 계산할때 
% sizestd_1 = floor(length(time1_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% % sizestd_1 = floor(length(time2_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% % sizestd_1 = floor(length(time3_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% % sizestd_1 = floor(length(time4_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% % sizestd_1 = floor(length(time1_1)/500);    %700샘플씩 계산하기 위함 
% std_1 = zeros(sizestd_1,4);
% std_3 = zeros(sizestd_1,4);

% % %k=100일 때부터 k가 1단위씩 계산할때 
sizestd_1 = length(time3_1)-99;    
std_1 = zeros(sizestd_1,4);
std_3 = zeros(sizestd_1,4);

pdf_1 = zeros(sizestd_1,4);
pdf_3 = zeros(sizestd_1,4);

x=x_dot3;
% x=x_dot2;
% x=x_dot3;
% x=x_dot4;

for k = 1:length(time3_1)

    disp(k)  

    x1_k(:,k) = evaluate(nss1,x3(:,k),input_3(:,k)); 
    x3_k(:,k) = evaluate(nss2,x3(:,k),input_3(:,k)); 

%     x1_k(:,k) = evaluate(nss1,x2(:,k),input_2(:,k)); 
%     x3_k(:,k) = evaluate(nss2,x2(:,k),input_2(:,k));

%     x1_k(:,k) = evaluate(nss1,x3(:,k),input_3(:,k)); 
%     x3_k(:,k) = evaluate(nss2,x3(:,k),input_3(:,k));

%     x1_k(:,k) = evaluate(nss1,x4(:,k),input_4(:,k)); 
%     x3_k(:,k) = evaluate(nss2,x4(:,k),input_4(:,k));

%     if mod(k,500) == 0     %k를 100으로 나눈 나머지가 0인지 확인
    if k >= 100

%%%%%%%%%%%%%%%%%%%%100샘플씩 계산할때 %%%%%%%%%%%%%%%%%%%%%%
        error_1(:,1) = x(k-99:1:k,1) - x1_k(1,k-99:1:k).';
        error_1(:,2) = x(k-99:1:k,2) - x1_k(2,k-99:1:k).';
        error_1(:,3) = x(k-99:1:k,3) - x1_k(3,k-99:1:k).';
        error_1(:,4) = x(k-99:1:k,4) - x1_k(4,k-99:1:k).';

        error_3(:,1) = x(k-99:1:k,1) - x3_k(1,k-99:1:k).';
        error_3(:,2) = x(k-99:1:k,2) - x3_k(2,k-99:1:k).';
        error_3(:,3) = x(k-99:1:k,3) - x3_k(3,k-99:1:k).';
        error_3(:,4) = x(k-99:1:k,4) - x3_k(4,k-99:1:k).';

%%%%%%%%%%%%%%%%%%%%%%%700샘플씩 계산할때 %%%%%%%%%%%%%%%%%%
%         error_1(:,1) = x(k-499:1:k,1) - x1_k(1,k-499:1:k).';
%         error_1(:,2) = x(k-499:1:k,2) - x1_k(2,k-499:1:k).';
%         error_1(:,3) = x(k-499:1:k,3) - x1_k(3,k-499:1:k).';
%         error_1(:,4) = x(k-499:1:k,4) - x1_k(4,k-499:1:k).';
% 
%         error_3(:,1) = x(k-499:1:k,1) - x3_k(1,k-499:1:k).';
%         error_3(:,2) = x(k-499:1:k,2) - x3_k(2,k-499:1:k).';
%         error_3(:,3) = x(k-499:1:k,3) - x3_k(3,k-499:1:k).';
%         error_3(:,4) = x(k-499:1:k,4) - x3_k(4,k-499:1:k).';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [V_1_1,M_1_1] = var(error_1(:,1));
        [V_1_2,M_1_2] = var(error_1(:,2));
        [V_1_3,M_1_3] = var(error_1(:,3));
        [V_1_4,M_1_4] = var(error_1(:,4));

        [V_3_1,M_3_1] = var(error_3(:,1));
        [V_3_2,M_3_2] = var(error_3(:,2));
        [V_3_3,M_3_3] = var(error_3(:,3));
        [V_3_4,M_3_4] = var(error_3(:,4));
    
        std_1_1 = sqrt(V_1_1);
        std_1_2 = sqrt(V_1_2);
        std_1_3 = sqrt(V_1_3);
        std_1_4 = sqrt(V_1_4);

        std_3_1 = sqrt(V_3_1);
        std_3_2 = sqrt(V_3_2);
        std_3_3 = sqrt(V_3_3);
        std_3_4 = sqrt(V_3_4);

%%%%%%% %k=100일 때부터 k가 100단위씩 계산할때 %%%%%%
%         std_1(k/100,1) = std_1_1;
%         std_1(k/100,2) = std_1_2;
%         std_1(k/100,3) = std_1_3;
%         std_1(k/100,4) = std_1_4;
% 
%         std_3(k/100,1) = std_3_1;
%         std_3(k/100,2) = std_3_2;
%         std_3(k/100,3) = std_3_3;
%         std_3(k/100,4) = std_3_4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% %k=500일 때부터 k가 500단위씩 계산할때 %%%%%%
%         std_1(k/500,1) = std_1_1;
%         std_1(k/500,2) = std_1_2;
%         std_1(k/500,3) = std_1_3;
%         std_1(k/500,4) = std_1_4;
% 
%         std_3(k/500,1) = std_3_1;
%         std_3(k/500,2) = std_3_2;
%         std_3(k/500,3) = std_3_3;
%         std_3(k/500,4) = std_3_4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% %k=100일 때부터 k가 1단위씩 계산할때 %%%%%%%%%%%%
        std_1(k-99,1) = std_1_1;
        std_1(k-99,2) = std_1_2;
        std_1(k-99,3) = std_1_3;
        std_1(k-99,4) = std_1_4;

        std_3(k-99,1) = std_3_1;
        std_3(k-99,2) = std_3_2;
        std_3(k-99,3) = std_3_3;
        std_3(k-99,4) = std_3_4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
        normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
        normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
        normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);

        normal_distribution_3_1 = makedist('Normal', 'mu', M_3_1, 'sigma', std_3_1);
        normal_distribution_3_2 = makedist('Normal', 'mu', M_3_2, 'sigma', std_3_2);
        normal_distribution_3_3 = makedist('Normal', 'mu', M_3_3, 'sigma', std_3_3);
        normal_distribution_3_4 = makedist('Normal', 'mu', M_3_4, 'sigma', std_3_4);
    
        x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
        y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
        x_3_1 = linspace(M_3_1 - 3 * std_3_1, M_3_1 + 3 * std_3_1, 1000); % 플롯을 위한 x 값 범위
        y_3_1 = pdf(normal_distribution_3_1, x_3_1); % 확률 밀도 함수 계산
        
        set(p_1_1,"XData",x_1_1,"YData",y_1_1);
        set(xline_1_1,'Value',M_1_1,'Color','r','LineStyle','--')
        set(title_1_1,'String',{'Normal distribution of Lateral position error';['mean1= ' num2str(M_1_1) ', std1= ' num2str(std_1_1)];['mean3= ' num2str(M_3_1) ', std3= ' num2str(std_3_1)]},'FontSize',15)

        set(p_3_1,"XData",x_3_1,"YData",y_3_1);
        set(xline_3_1,'Value',M_3_1,'Color','k','LineStyle','--')

        x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
        y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
        x_3_2 = linspace(M_3_2 - 3 * std_3_2, M_3_2 + 3 * std_3_2, 1000); % 플롯을 위한 x 값 범위
        y_3_2 = pdf(normal_distribution_3_2, x_3_2); % 확률 밀도 함수 계산

        set(p_1_2,"XData",x_1_2,"YData",y_1_2);
        set(xline_1_2,'Value',M_1_2,'Color','r','LineStyle','--')
        set(title_1_2,'String',{'Normal distribution of Lateral velocity error';['mean1= ' num2str(M_1_2) ', std1= ' num2str(std_1_2)];['mean3= ' num2str(M_3_2) ', std3= ' num2str(std_3_2)]},'FontSize',15)

        set(p_3_2,"XData",x_3_2,"YData",y_3_2);
        set(xline_3_2,'Value',M_3_2,'Color','k','LineStyle','--')

        x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
        y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
        x_3_3 = linspace(M_3_3 - 3 * std_3_3, M_3_3 + 3 * std_3_3, 1000); % 플롯을 위한 x 값 범위
        y_3_3 = pdf(normal_distribution_3_3, x_3_3); % 확률 밀도 함수 계산

        set(p_1_3,"XData",x_1_3,"YData",y_1_3);
        set(xline_1_3,'Value',M_1_3,'Color','r','LineStyle','--')
        set(title_1_3,'String',{'Normal distribution of Yaw error';['mean1= ' num2str(M_1_3) ', std1= ' num2str(std_1_3)];['mean3= ' num2str(M_3_3) ', std3= ' num2str(std_3_3)]},'FontSize',15)

        set(p_3_3,"XData",x_3_3,"YData",y_3_3);
        set(xline_3_3,'Value',M_3_3,'Color','k','LineStyle','--')

        x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
        y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
        x_3_4 = linspace(M_3_4 - 3 * std_3_4, M_3_4 + 3 * std_3_4, 1000); % 플롯을 위한 x 값 범위
        y_3_4 = pdf(normal_distribution_3_4, x_3_4); % 확률 밀도 함수 계산

        set(p_1_4,"XData",x_1_4,"YData",y_1_4);
        set(xline_1_4,'Value',M_1_4,'Color','r','LineStyle','--')
        set(title_1_4,'String',{'Normal distribution of Yaw rate error';['mean1= ' num2str(M_1_4) ', std1= ' num2str(std_1_4)];['mean3= ' num2str(M_3_4) ', std3= ' num2str(std_3_4)]},'FontSize',15)

        set(p_3_4,"XData",x_3_4,"YData",y_3_4);
        set(xline_3_4,'Value',M_3_4,'Color','k','LineStyle','--')

%         if std_1_1 < std_3_1
%             a_1.String = '고속 주회로';
%         else
%             a_1.String = 'K-city';
%         end
% 
%         if std_1_2 < std_3_2
%             a_2.String = '고속 주회로';
%         else
%             a_2.String = 'K-city';
%         end
% 
%         if std_1_3 < std_3_3
%             a_3.String = '고속 주회로';
%         else
%             a_3.String = 'K-city';
%         end
% 
%         if std_1_4 < std_3_4
%             a_4.String = '고속 주회로';
%         else
%             a_4.String = 'K-city';
%         end
        
        pdf_1_1 = pdf(normal_distribution_1_1, M_1_1); % error의 평균의 확률 밀도 함수값 계산
        pdf_1_2 = pdf(normal_distribution_1_2, M_1_2); % error의 평균의 확률 밀도 함수값 계산
        pdf_1_3 = pdf(normal_distribution_1_3, M_1_3); % error의 평균의 확률 밀도 함수값 계산
        pdf_1_4 = pdf(normal_distribution_1_4, M_1_4); % error의 평균의 확률 밀도 함수값 계산

        pdf_3_1 = pdf(normal_distribution_3_1, M_3_1); % error의 평균의 확률 밀도 함수값 계산
        pdf_3_2 = pdf(normal_distribution_3_2, M_3_2); % error의 평균의 확률 밀도 함수값 계산
        pdf_3_3 = pdf(normal_distribution_3_3, M_3_3); % error의 평균의 확률 밀도 함수값 계산
        pdf_3_4 = pdf(normal_distribution_3_4, M_3_4); % error의 평균의 확률 밀도 함수값 계산

        pdf_1(k-99,1) = pdf_1_1;
        pdf_1(k-99,2) = pdf_1_2;
        pdf_1(k-99,3) = pdf_1_3;
        pdf_1(k-99,4) = pdf_1_4;

        pdf_3(k-99,1) = pdf_3_1;
        pdf_3(k-99,2) = pdf_3_2;
        pdf_3(k-99,3) = pdf_3_3;
        pdf_3(k-99,4) = pdf_3_4;
        
        if pdf_1_1 > pdf_3_1
            a_1.String = '고속 주회로';
        else
            a_1.String = 'K-city';
        end

        if pdf_1_2 > pdf_3_2
            a_2.String = '고속 주회로';
        else
            a_2.String = 'K-city';
        end

        if pdf_1_3 > pdf_3_3
            a_3.String = '고속 주회로';
        else
            a_3.String = 'K-city';
        end

        if pdf_1_4 > pdf_3_4
            a_4.String = '고속 주회로';
        else
            a_4.String = 'K-city';
        end

        pause(0.0001)
        
    end
    
end

min_std_1_1 = min(std_1(:,1));
max_std_1_1 = max(std_1(:,1));

min_std_1_2 = min(std_1(:,2));
max_std_1_2 = max(std_1(:,2));

min_std_1_3 = min(std_1(:,3));
max_std_1_3 = max(std_1(:,3));

min_std_1_4 = min(std_1(:,4));
max_std_1_4 = max(std_1(:,4));

min_std_3_1 = min(std_3(:,1));
max_std_3_1 = max(std_3(:,1));

min_std_3_2 = min(std_3(:,2));
max_std_3_2 = max(std_3(:,2));

min_std_3_3 = min(std_3(:,3));
max_std_3_3 = max(std_3(:,3));

min_std_3_4 = min(std_3(:,4));
max_std_3_4 = max(std_3(:,4));


% %%%%%%%%%%%%%%%%%%%%gt1 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt1');
% subplot(2,2,1);
% plot(time1_2,ylin1_1,'r',time1_2,yn1_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin1_1-yn1_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time1_2,ylin1_2,'r',time1_2,yn1_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin1_2-yn1_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time1_2,ylin1_3,'r',time1_2,yn1_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin1_3-yn1_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time1_2,ylin1_4,'r',time1_2,yn1_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin1_4-yn1_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)


%%%%%%%%%%%%%%%%%%%%Original%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','Original Standard Deviation graph')
subplot(2,2,1)
plot(std_1(:,1),'r')
hold on
plot(std_3(:,1),'b')
% xline([817,3532, 6588, 6663, 6794, 7637, 20689, 21198],'k')
legend('고속 주회로 모델','K-city 모델')
title('Original standard deviation of Lateral position error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('Standard Deviation','FontSize', 15);

subplot(2,2,2)
plot(std_1(:,2),'r')
hold on
plot(std_3(:,2),'b')
% xline([74, 90, 165, 221, 1740, 1928, 1939, 2036, 2087, 2095, 3194, 3603, 3634, 3683, 3918, 4814, 10276, 10287, 10296, 10316, 11027, 11272, 11332, 11430, 12047, 12943, 12952, 17455, 17589, 18137, 24540, 24600, 24763, 24871, 25059, 26225 ],'k')
legend('고속 주회로 모델','K-city 모델')
title('Original standard deviation of Lateral velocity error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('Standard Deviation','FontSize', 15);

subplot(2,2,3)
plot(std_1(:,3),'r')
hold on
plot(std_3(:,3),'b')
% xline([3131, 3223, 4345, 4630, 9922, 10118],'k')
legend('고속 주회로 모델','K-city 모델')
title('Original standard deviation of Yaw error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('Standard Deviation','FontSize', 15);

subplot(2,2,4)
plot(std_1(:,4),'r')
hold on
plot(std_3(:,4),'b')
% xline([1006, 1254],'k')
legend('고속 주회로 모델','K-city 모델')
title('Original standard deviation of Yaw rate error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('Standard Deviation','FontSize', 15);

%K-city 1 data를 입력했기 때문에 std_1이 std_3보다 커야하는 것이 정상 = err_std가 0보다 커야하는 것이 정상
err_std_1 = std_1(:,1)-std_3(:,1);
err_std_2 = std_1(:,2)-std_3(:,2);
err_std_3 = std_1(:,3)-std_3(:,3);
err_std_4 = std_1(:,4)-std_3(:,4);

figure('Name','gt1 Model VS gt3 Model std error');
subplot(2,2,1);
plot(err_std_1)
hold on
% xline(248.741,'r')
yline(0,'k')
% text(248.741, -0.12, [' x = ' num2str(248.741)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('Lateral position std error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(err_std_2)
hold on
% xline(211.641,'r')
yline(0,'k')
% text(211.641, -0.22, [' x = ' num2str(211.641)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('Lateral velocity std error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(err_std_3)
hold on
% xline(255.311,'r')
yline(0,'k')
% text(255.311, -0.15, [' x = ' num2str(255.311)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('yaw std error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(err_std_4)
hold on
% xline(256.491,'r')
yline(0,'k')
% text(256.491, -0.1, [' x = ' num2str(256.491)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('yaw angle rate std error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

index = zeros(size(err_std_1,1),1);
for i=1:size(err_std_1,1)
    
    if (err_std_1(i) < 0) && (err_std_2(i) < 0) && (err_std_3(i) < 0) && (err_std_4(i) < 0)
        index(i) = i;
    end
end

for j=1:length(index)
    if index(j) ~= 0
        disp(j)
    end
end

%%%%%%%%%%%%%%%%%Moving average filter%%%%%%%%%%%%%%%%%%%%%%%%%
% window_size = 500;
% mov_std_1_1 = movmean(std_1(:,1),window_size);
% mov_std_1_2 = movmean(std_1(:,2),window_size);
% mov_std_1_3 = movmean(std_1(:,3),window_size);
% mov_std_1_4 = movmean(std_1(:,4),window_size);
% 
% mov_std_3_1 = movmean(std_3(:,1),window_size);
% mov_std_3_2 = movmean(std_3(:,2),window_size);
% mov_std_3_3 = movmean(std_3(:,3),window_size);
% mov_std_3_4 = movmean(std_3(:,4),window_size);
% 
% figure('Name','Standard Deviation graph after moving average filter')
% subplot(2,2,1)
% plot(mov_std_1_1,'r')
% hold on
% grid on
% plot(mov_std_3_1,'b')
% % xline(21199,'k')
% % text(21199, 2.2e-03, [' x = ' num2str(21199)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Lateral position error','FontSize', 15)
% % xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,2)
% plot(mov_std_1_2,'r')
% hold on
% grid on
% plot(mov_std_3_2,'b')
% % xline(26226,'k')
% % text(26226, 3.1e-03, [' x = ' num2str(26226)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Lateral velocity error','FontSize', 15)
% % xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,3)
% plot(mov_std_1_3,'r')
% hold on
% grid on
% plot(mov_std_3_3,'b')
% % xline(10119,'k')
% % text(10119, 2.2e-03, [' x = ' num2str(10119)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Yaw error','FontSize', 15)
% % xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,4)
% plot(mov_std_1_4,'r')
% hold on
% grid on
% plot(mov_std_3_4,'b')
% % xline(1255,'k')
% % text(1255, 2.7e-03, [' x = ' num2str(1255)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Yaw rate error','FontSize', 15)
% % xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% err_mov_std_1 = mov_std_1_1-mov_std_3_1;
% err_mov_std_2 = mov_std_1_2-mov_std_3_2;
% err_mov_std_3 = mov_std_1_3-mov_std_3_3;
% err_mov_std_4 = mov_std_1_4-mov_std_3_4;
% 
% figure('Name','gt1 Model VS gt3 Model mov std error');
% subplot(2,2,1);
% plot(err_mov_std_1)
% hold on
% % xline(248.741,'r')
% yline(0,'k')
% % text(248.741, -0.12, [' x = ' num2str(248.741)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
% grid on
% title('Lateral position mov std error','FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(err_mov_std_2)
% hold on
% % xline(211.641,'r')
% yline(0,'k')
% % text(211.641, -0.22, [' x = ' num2str(211.641)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
% grid on
% title('Lateral velocity mov std error','FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(err_mov_std_3)
% hold on
% % xline(255.311,'r')
% yline(0,'k')
% % text(255.311, -0.15, [' x = ' num2str(255.311)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
% grid on
% title('yaw mov std error','FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(err_mov_std_4)
% hold on
% % xline(256.491,'r')
% yline(0,'k')
% % text(256.491, -0.1, [' x = ' num2str(256.491)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
% grid on
% title('yaw angle rate mov std error','FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)

%%%%%%%%%%%%%%%% K-city의 std가 고속주회로의 std보다 낮은 구간 탐색 %%%%%%%%%%%
%input=고속주회로1의 state,wheel steering angle input
%output:
    %mov_std_1_1,2,3,4 : moving average filter를 거친 고속주회로 모델의 error의 std
    %mov_std_3_1,2,3,4 : moving average filter를 거친 K-city 모델의 error의 std
%description:
    %K-city의 std가 고속주회로의 std보다 낮은 구간 탐색
    %각 구간에서 시작 index와 종료 index 탐색
    %moving average filter의 sliding window size = 500
    %moving average filter index의 1번=xdot_1의 100번째=x의 99번째 

% err=mov_std_3_1-mov_std_1_1;      %err index 기준: 817, 3532, 6588, 6663, 6794, 7637, 20689, 21198
% err=mov_std_3_2-mov_std_1_2;        %err index 기준: 74, 90, 165, 221, 1740, 1928, 1939, 2036, 2087, 2095, 3194, 3603, 3634, 3683, 3918, 4814, 10276, 10287, 10296, 10316, 11027, 11272, 11332, 11430, 12047, 12943, 12952, 17455, 17589, 18137, 24540, 24600, 24763, 24871, 25059, 26225 
% err=mov_std_3_3-mov_std_1_3;       %err index 기준: 3131, 3223, 4345, 4630, 9922, 10118
% err=mov_std_3_4-mov_std_1_4;       %err index 기준: 1006, 1254err=mov_std_3_2-mov_std_1_2;
% figure
% plot(err)
% hold on
% yline(0)
% idx=zeros(1,1);
% 
% for ii = 1:length(err)
%     if err(ii)<=0
%         idx(end+1,1)=ii;
%     end
% end
% idx(1)=[];


%%%%%%%%%%%%%%%%%%%Moving Median filter%%%%%%%%%%%%%%%%%%%%%
% filter_dimension = 100;
% med_std_1_1 = medfilt1(std_1(:,1),filter_dimension);
% med_std_1_2 = medfilt1(std_1(:,2),filter_dimension);
% med_std_1_3 = medfilt1(std_1(:,3),filter_dimension);
% med_std_1_4 = medfilt1(std_1(:,4),filter_dimension);
% 
% med_std_3_1 = medfilt1(std_3(:,1),filter_dimension);
% med_std_3_2 = medfilt1(std_3(:,2),filter_dimension);
% med_std_3_3 = medfilt1(std_3(:,3),filter_dimension);
% med_std_3_4 = medfilt1(std_3(:,4),filter_dimension);
% 
% figure('Name','normaldistribution graph after moving median filter')
% subplot(2,2,1)
% plot(med_std_1_1,'r')
% hold on
% plot(med_std_3_1,'b')
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Lateral position error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,2)
% plot(med_std_1_2,'r')
% hold on
% plot(med_std_3_2,'b')
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Lateral velocity error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,3)
% plot(med_std_1_3,'r')
% hold on
% plot(med_std_3_3,'b')
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Yaw error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,4)
% plot(med_std_1_4,'r')
% hold on
% plot(med_std_3_4,'b')
% legend('고속 주회로 모델','K-city 모델')
% title('Standard Deviation of Yaw rate error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);

%%%%%%%%%%%%%%%%%%%std의 Average Filter%%%%%%%%%%%%%%%%%%%%%
% Nsamples_1 = length(std_1);
% Avgsaved_1 = zeros(Nsamples_1, 4);
% 
% Nsamples_3 = length(std_3);
% Avgsaved_3 = zeros(Nsamples_3, 4);
% 
% for k = 1:Nsamples_1
%     
% %     avg_1_1 = AvgFilter(std_1(k,1));
% %     avg_1_2 = AvgFilter(std_1(k,2));
% %     avg_1_3 = AvgFilter(std_1(k,3));
% %     avg_1_4 = AvgFilter(std_1(k,4));
% 
% %     Avgsaved_1(k,1) = avg_1_1;
% %     Avgsaved_1(k,2) = avg_1_2;
% %     Avgsaved_1(k,3) = avg_1_3;
% %     Avgsaved_1(k,4) = avg_1_4;
%     Avgsaved_1(k,:)=AvgFilter(std_1(k,:));
% 
% end
% 
% for k = 1:Nsamples_3
%     
% %     avg_3_1 = AvgFilter(std_3(k,1));
% %     avg_3_2 = AvgFilter(std_3(k,2));
% %     avg_3_3 = AvgFilter(std_3(k,3));
% %     avg_3_4 = AvgFilter(std_3(k,4));
% 
% %     Avgsaved_3(k,1) = avg_3_1;
% %     Avgsaved_3(k,2) = avg_3_2;
% %     Avgsaved_3(k,3) = avg_3_3;
% %     Avgsaved_3(k,4) = avg_3_4;
%     Avgsaved_3(k,:)=AvgFilter(std_3(k,:));
% 
% end
% 
% figure('Name','std after average filter')
% subplot(2,2,1)
% plot(Avgsaved_1(:,1),'r-')
% hold on
% plot(Avgsaved_3(:,1),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('std of Lateral position error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,2)
% plot(Avgsaved_1(:,2),'r-')
% hold on
% plot(Avgsaved_3(:,2),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('std of Lateral velocity error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,3)
% plot(Avgsaved_1(:,3),'r-')
% hold on
% plot(Avgsaved_3(:,3),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('std of Yaw error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,4)
% plot(Avgsaved_1(:,4),'r-')
% hold on
% plot(Avgsaved_3(:,4),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('std of Yaw rate error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% sum(Avgsaved_1(:, 1) < Avgsaved_3(:, 1))
% size(Avgsaved_1(:, 1),1)
% 
% sum(Avgsaved_1(:, 2) < Avgsaved_3(:, 2))
% size(Avgsaved_1(:, 2),1)
% 
% sum(Avgsaved_1(:, 3) < Avgsaved_3(:, 3))
% size(Avgsaved_1(:, 3),1)
% 
% sum(Avgsaved_1(:, 4) < Avgsaved_3(:, 4))
% size(Avgsaved_1(:, 4),1)

%%%%%%%%%%%%%%%%%Moving average filter%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','error의 평균의 확률밀도함수값 변화')
% subplot(2,2,1)
% plot(pdf_1(:,1),'r')
% hold on
% plot(pdf_3(:,1),'b')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Lateral position error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,2)
% plot(pdf_1(:,2),'r')
% hold on
% plot(pdf_3(:,2),'b')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Lateral velocity error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,3)
% plot(pdf_1(:,3),'r')
% hold on
% plot(pdf_3(:,3),'b')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Yaw error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% 
% subplot(2,2,4)
% plot(pdf_1(:,4),'r')
% hold on
% plot(pdf_3(:,4),'b')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Yaw rate error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);

%%%%%%%%%%%%%%%%%%%error의 pdf의 Average Filter%%%%%%%%%%%%%%%%%%%%%
% Nsamples_1 = length(pdf_1);
% Avgsaved_1 = zeros(Nsamples_1, 4);
% 
% Nsamples_3 = length(pdf_3);
% Avgsaved_3 = zeros(Nsamples_3, 4);
% 
% for k = 1:Nsamples_1
%     
%     avg_1_1 = AvgFilter(pdf_1(k,1));
%     avg_1_2 = AvgFilter(pdf_1(k,2));
%     avg_1_3 = AvgFilter(pdf_1(k,3));
%     avg_1_4 = AvgFilter(pdf_1(k,4));
% 
%     Avgsaved_1(k,1) = avg_1_1;
%     Avgsaved_1(k,2) = avg_1_2;
%     Avgsaved_1(k,3) = avg_1_3;
%     Avgsaved_1(k,4) = avg_1_4;
% 
% end
% 
% for k = 1:Nsamples_3
%     
%     avg_3_1 = AvgFilter(pdf_3(k,1));
%     avg_3_2 = AvgFilter(pdf_3(k,2));
%     avg_3_3 = AvgFilter(pdf_3(k,3));
%     avg_3_4 = AvgFilter(pdf_3(k,4));
% 
%     Avgsaved_3(k,1) = avg_3_1;
%     Avgsaved_3(k,2) = avg_3_2;
%     Avgsaved_3(k,3) = avg_3_3;
%     Avgsaved_3(k,4) = avg_3_4;
% 
% end
% 
% figure('Name','error의 평균의 확률밀도함수값 변화 after average filter')
% subplot(2,2,1)
% plot(Avgsaved_1(:,1),'r-')
% hold on
% plot(Avgsaved_3(:,1),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Lateral position error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('Standard Deviation','FontSize', 15);
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('pdf','FontSize', 15);
% 
% subplot(2,2,2)
% plot(Avgsaved_1(:,2),'r-')
% hold on
% plot(Avgsaved_3(:,2),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of of Lateral velocity error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('pdf','FontSize', 15);
% 
% subplot(2,2,3)
% plot(Avgsaved_1(:,3),'r-')
% hold on
% plot(Avgsaved_3(:,3),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of Yaw error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('pdf','FontSize', 15);
% 
% subplot(2,2,4)
% plot(Avgsaved_1(:,4),'r-')
% hold on
% plot(Avgsaved_3(:,4),'b--')
% legend('고속 주회로 모델','K-city 모델')
% title('pdf of mean of of Yaw rate error','FontSize', 15)
% xlabel('$K_{th}$ state [x100]', 'Interpreter', 'latex', 'FontSize', 15);
% ylabel('pdf','FontSize', 15);
% 
% sum(Avgsaved_1(:, 1) > Avgsaved_3(:, 1))
% size(Avgsaved_1(:, 1),1)
% 
% sum(Avgsaved_1(:, 2) > Avgsaved_3(:, 2))
% size(Avgsaved_1(:, 2),1)
% 
% sum(Avgsaved_1(:, 3) > Avgsaved_3(:, 3))
% size(Avgsaved_1(:, 3),1)
% 
% sum(Avgsaved_1(:, 4) > Avgsaved_3(:, 4))
% size(Avgsaved_1(:, 4),1)

% %%%%%%%%%%%%%%%%%%%%%%gt1 정규분포 및 RMSE %%%%%%%%%%%%%%%%%%%%%%%%%
x1_k = x1_k.';
x3_k = x3_k.';

ylin1_1 = x_dot1(:,1);
yn1_1 = x1_k(:,1);
yn3_1 = x3_k(:,1);
err_1 = yn1_1-yn3_1;

ylin1_2 = x_dot1(:,2);
yn1_2 = x1_k(:,2);
yn3_2 = x3_k(:,2);
err_2 = yn1_2-yn3_2;

ylin1_3 = x_dot1(:,3);
yn1_3 = x1_k(:,3);
yn3_3 = x3_k(:,3);
err_3 = yn1_3-yn3_3;

ylin1_4 = x_dot1(:,4);
yn1_4 = x1_k(:,4);
yn3_4 = x3_k(:,4);
err_4 = yn1_4-yn3_4;

figure('Name','gt1 Model VS gt3 Model');
subplot(2,2,1);
plot(time3_2,ylin1_1,'r',time3_2,yn1_1,'b--',time3_2,yn3_1,'g-')
hold on
xline(248.741,'k')
text(248.741, 2, [' x = ' num2str(248.741)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
legend("Original","gt1 model","gt3 model",'FontSize',15);
grid on
title('Lateral position','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time3_2,ylin1_2,'r',time3_2,yn1_2,'b--',time3_2,yn3_2,'g-')
hold on
xline(211.641,'k')
text(211.641, 0.23, [' x = ' num2str(211.641)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
legend("Original","gt1 model","gt3 model",'FontSize',15);
grid on
title('Lateral velocity','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time3_2,ylin1_3,'r',time3_2,yn1_3,'b--',time3_2,yn3_3,'g-')
hold on
xline(255.311,'k')
text(255.311, 0, [' x = ' num2str(255.311)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
legend("Original","gt1 model","gt3 model",'FontSize',15);
grid on
title('yaw','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time3_2,ylin1_4,'r',time3_2,yn1_4,'b--',time3_2,yn3_4,'g-')
hold on
xline(256.491,'k')
text(256.491, 0.1, [' x = ' num2str(256.491)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'k','FontSize',15);
legend("Original","gt1 model","gt3 model",'FontSize',15);
grid on
title('yaw angle rate','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

figure('Name','gt1 Model VS gt3 Model error');
subplot(2,2,1);
plot(time3_2,err_1)
hold on
xline(248.741,'r')
yline(0,'k')
text(248.741, -0.12, [' x = ' num2str(248.741)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('Lateral position error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('y [m]','FontSize',15)

subplot(2,2,2);
plot(time3_2,err_2)
hold on
xline(211.641,'r')
yline(0,'k')
text(211.641, -0.22, [' x = ' num2str(211.641)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('Lateral velocity error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dy/dt [m/s]','FontSize',15)

subplot(2,2,3);
plot(time3_2,err_3)
hold on
xline(255.311,'r')
yline(0,'k')
text(255.311, -0.15, [' x = ' num2str(255.311)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('yaw error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('ψ [rad]','FontSize',15)

subplot(2,2,4);
plot(time3_2,err_4)
hold on
xline(256.491,'r')
yline(0,'k')
text(256.491, -0.1, [' x = ' num2str(256.491)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r','FontSize',15);
grid on
title('yaw angle rate error','FontSize',15)
xlabel('Time[s]','FontSize',15)
ylabel('dψ/dt [rad/s]','FontSize',15)

% analFullmotion: 전체 시간 영영에서 종방향 속도와 steering 분석
% analLocalmotion: 탐색하고자 하는 영역 내에서 종방향 속도와 steering 분석
gt1 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
gt2 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-51-54_gv80_v4_2.bag';   %GT/고속 주회로
gt3 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City/
gt4 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-10-49_gv80_v4_2.bag';   %GT/K-City

analFullmotion(gt1,input1_data,'gt1')
analFullmotion(gt2,input2_data,'gt2')
analFullmotion(gt3,input3_data,'gt3')
analFullmotion(gt4,input4_data,'gt4')

comparison_index = 5883;
[minmax_input,minmax_vel_x]=analLocalmotion(gt3,input3_data,comparison_index)
analFullmotion2(gt1,input1_data,minmax_input,'gt1')
analFullmotion2(gt2,input2_data,minmax_input,'gt2')
% %탐색하고자 하는 영역 내에서 error의 정규분포 시각화
% search_index = 26372;                                                             %std1이 std2보다 크거나 같은 구간 탐색 index
% [error1,error2,rmse1,rmse2]=outputcomparisonfunction(time1_2,x_dot1,x1_k,x3_k,search_index); %탐색하고자 하는 영역 내에서 ground truth data,고속주회로 Model의 output, k-city Model의 output 시각화
% [std1,std2]=normaldistributionfunction(error1, error2)                           %std1은 x_dot1과 x1_k의 error에 대한 std
%                                                                                  %std2은 x_dot1과 x3_k의 error에 대한 std
% rmse1
% rmse2
% 
% if all(std1 > std2)
%     disp('1');
% else
%     disp('0');
% end


% %%%%%%%%%%%%%%%%%%%%%%%%%%gt2%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz2=size(time2_2);
% 
% x2_k = zeros(4,sz2(1)); 
% 
% x2 = x2.';
% input_2 = input_2.';
% 
% error_1 = zeros(100,4); 
% 
% figure('Name','gt2 realtime normaldistribution')
% subplot(2,2,1)
% M_1_1=0;
% std_1_1=0;
% normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
% x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
% y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
% p_1_1=plot(x_1_1, y_1_1,'r', 'LineWidth', 2);
% xline_1_1=xline(M_1_1, 'k--');
% title_1_1=title({'initial Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2)
% M_1_2=0;
% std_1_2=0;
% normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
% x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
% y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
% p_1_2=plot(x_1_2, y_1_2,'r', 'LineWidth', 2);
% xline_1_2=xline(M_1_2, 'k--');
% title_1_2=title({'initial Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3)
% M_1_3=0;
% std_1_3=0;
% normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
% x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
% y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
% p_1_3=plot(x_1_3, y_1_3,'r', 'LineWidth', 2);
% xline_1_3=xline(M_1_3, 'k--');
% title_1_3=title({'initial Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4)
% M_1_4=0;
% std_1_4=0;
% normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
% x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
% y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
% p_1_4=plot(x_1_4, y_1_4,'r', 'LineWidth', 2);
% xline_1_4=xline(M_1_4, 'k--');
% title_1_4=title({'initial Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% sizestd_1 = floor(length(time2_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% std_1 = zeros(sizestd_1,4);
% 
% 
% for k = 1:length(time2_1)
% 
%     disp(k)
% 
%     x2_k(:,k) = evaluate(nss,x2(:,k),input_2(:,k)); 
%     
%     if mod(k,100) == 0     %k를 100으로 나눈 나머지가 0인지 확인
% 
%         error_1(:,1) = x_dot2(k-99:1:k,1) - x2_k(1,k-99:1:k).';
%         error_1(:,2) = x_dot2(k-99:1:k,2) - x2_k(2,k-99:1:k).';
%         error_1(:,3) = x_dot2(k-99:1:k,3) - x2_k(3,k-99:1:k).';
%         error_1(:,4) = x_dot2(k-99:1:k,4) - x2_k(4,k-99:1:k).';
%     
%         [V_1_1,M_1_1] = var(error_1(:,1));
%         [V_1_2,M_1_2] = var(error_1(:,2));
%         [V_1_3,M_1_3] = var(error_1(:,3));
%         [V_1_4,M_1_4] = var(error_1(:,4));
%     
%         std_1_1 = sqrt(V_1_1);
%         std_1_2 = sqrt(V_1_2);
%         std_1_3 = sqrt(V_1_3);
%         std_1_4 = sqrt(V_1_4);
% 
%         std_1(k/100,1) = std_1_1;
%         std_1(k/100,2) = std_1_2;
%         std_1(k/100,3) = std_1_3;
%         std_1(k/100,4) = std_1_4;
% 
%         normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
%         normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
%         normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
%         normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
%     
%         x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
%         y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
%         set(p_1_1,"XData",x_1_1,"YData",y_1_1);
%         set(xline_1_1,'Value',M_1_1,'Color','k','LineStyle','--')
%         set(title_1_1,'String',{'Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15)
% 
%         x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
%         y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
%         set(p_1_2,"XData",x_1_2,"YData",y_1_2);
%         set(xline_1_2,'Value',M_1_2,'Color','k','LineStyle','--')
%         set(title_1_2,'String',{'Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15)
% 
%         x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
%         y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
%         set(p_1_3,"XData",x_1_3,"YData",y_1_3);
%         set(xline_1_3,'Value',M_1_3,'Color','k','LineStyle','--')
%         set(title_1_3,'String',{'Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15)
% 
%         x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
%         y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
%         set(p_1_4,"XData",x_1_4,"YData",y_1_4);
%         set(xline_1_4,'Value',M_1_4)
%         set(title_1_4,'String',{'Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15)
%         
%         pause(1)
%         
%     end
% 
%     
% end
% 
% min_std_1_1 = min(std_1(:,1));
% max_std_1_1 = max(std_1(:,1));
% 
% min_std_1_2 = min(std_1(:,2));
% max_std_1_2 = max(std_1(:,2));
% 
% min_std_1_3 = min(std_1(:,3));
% max_std_1_3 = max(std_1(:,3));
% 
% min_std_1_4 = min(std_1(:,4));
% max_std_1_4 = max(std_1(:,4));
% 
% 
% x2_k = x2_k.';
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% ylin2_1 = x_dot2(:,1);
% yn2_1 = x2_k(:,1);
% 
% ylin2_2 = x_dot2(:,2);
% yn2_2 = x2_k(:,2);
% 
% ylin2_3 = x_dot2(:,3);
% yn2_3 = x2_k(:,3);
% 
% ylin2_4 = x_dot2(:,4);
% yn2_4 = x2_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%gt2 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt2');
% subplot(2,2,1);
% plot(time2_2,ylin2_1,'r',time2_2,yn2_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin2_1-yn2_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time2_2,ylin2_2,'r',time2_2,yn2_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin2_2-yn2_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time2_2,ylin2_3,'r',time2_2,yn2_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin2_3-yn2_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time2_2,ylin2_4,'r',time2_2,yn2_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin2_4-yn2_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% %%%%%%%%%%%%%%%%%%%%%gt2 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure('Name','Normal distribution2');
% % subplot(2,2,1);
% % error2_1 = ylin2_1-yn2_1;
% % [V2_1,M2_1] = var(error2_1);
% % std2_1 = sqrt(V2_1);
% % normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
% % x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
% % y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산
% % plot(x2_1, y2_1,'r', 'LineWidth', 2);
% % xline(M2_1, 'k--');
% % title({'Normal distribution of Lateral position error';['mean= ' num2str(M2_1) ', std= ' num2str(std2_1)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,2);
% % error2_2 = ylin2_2-yn2_2;
% % [V2_2,M2_2] = var(error2_2);
% % std2_2 = sqrt(V2_2);
% % normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
% % x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
% % y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산
% % plot(x2_2, y2_2,'g', 'LineWidth', 2);
% % xline(M2_2, 'k--');
% % title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M2_2) ', std= ' num2str(std2_2)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,3);
% % error2_3 = ylin2_3-yn2_3;
% % [V2_3,M2_3] = var(error2_3);
% % std2_3 = sqrt(V2_3);
% % normal_distribution2_3 = makedist('Normal', 'mu', M2_3, 'sigma', std2_3);
% % x2_3 = linspace(M2_3 - 3 * std2_3, M2_3 + 3 * std2_3, 1000); % 플롯을 위한 x 값 범위
% % y2_3 = pdf(normal_distribution2_3, x2_3); % 확률 밀도 함수 계산
% % plot(x2_3, y2_3,'b', 'LineWidth', 2);
% % xline(M2_3, 'k--');
% % title({'Normal distribution of yaw error';['mean= ' num2str(M2_3) ', std= ' num2str(std2_3)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,4);
% % error2_4 = ylin2_4-yn2_4;
% % [V2_4,M2_4] = var(error2_4);
% % std2_4 = sqrt(V2_4);
% % normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
% % x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
% % y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산
% % plot(x2_4, y2_4,'c', 'LineWidth', 2);
% % xline(M2_4, 'k--');
% % title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M2_4) ', std= ' num2str(std2_4)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz3=size(time3_1);
% 
% x3_k = zeros(4,sz3(1)); 
% x3 = x3.';
% input_3 = input_3.';
% 
% error_1 = zeros(100,4); 
% 
% figure('Name','gt3 realtime normaldistribution')
% subplot(2,2,1)
% M_1_1=0;
% std_1_1=0;
% normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
% x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
% y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
% p_1_1=plot(x_1_1, y_1_1,'r', 'LineWidth', 2);
% xline_1_1=xline(M_1_1, 'k--');
% title_1_1=title({'initial Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2)
% M_1_2=0;
% std_1_2=0;
% normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
% x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
% y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
% p_1_2=plot(x_1_2, y_1_2,'r', 'LineWidth', 2);
% xline_1_2=xline(M_1_2, 'k--');
% title_1_2=title({'initial Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3)
% M_1_3=0;
% std_1_3=0;
% normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
% x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
% y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
% p_1_3=plot(x_1_3, y_1_3,'r', 'LineWidth', 2);
% xline_1_3=xline(M_1_3, 'k--');
% title_1_3=title({'initial Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4)
% M_1_4=0;
% std_1_4=0;
% normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
% x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
% y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
% p_1_4=plot(x_1_4, y_1_4,'r', 'LineWidth', 2);
% xline_1_4=xline(M_1_4, 'k--');
% title_1_4=title({'initial Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% sizestd_1 = floor(length(time3_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% std_1 = zeros(sizestd_1,4);
% 
% for k = 1:length(time3_1)
% 
%     disp(k)
% 
%     x3_k(:,k) = evaluate(nss,x3(:,k),input_3(:,k)); 
% 
%     if mod(k,100) == 0     %k를 100으로 나눈 나머지가 0인지 확인
% 
%         error_1(:,1) = x_dot3(k-99:1:k,1) - x3_k(1,k-99:1:k).';
%         error_1(:,2) = x_dot3(k-99:1:k,2) - x3_k(2,k-99:1:k).';
%         error_1(:,3) = x_dot3(k-99:1:k,3) - x3_k(3,k-99:1:k).';
%         error_1(:,4) = x_dot3(k-99:1:k,4) - x3_k(4,k-99:1:k).';
%     
%         [V_1_1,M_1_1] = var(error_1(:,1));
%         [V_1_2,M_1_2] = var(error_1(:,2));
%         [V_1_3,M_1_3] = var(error_1(:,3));
%         [V_1_4,M_1_4] = var(error_1(:,4));
%     
%         std_1_1 = sqrt(V_1_1);
%         std_1_2 = sqrt(V_1_2);
%         std_1_3 = sqrt(V_1_3);
%         std_1_4 = sqrt(V_1_4);
% 
%         std_1(k/100,1) = std_1_1;
%         std_1(k/100,2) = std_1_2;
%         std_1(k/100,3) = std_1_3;
%         std_1(k/100,4) = std_1_4;
% 
%         normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
%         normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
%         normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
%         normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
%     
%         x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
%         y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
%         set(p_1_1,"XData",x_1_1,"YData",y_1_1);
%         set(xline_1_1,'Value',M_1_1,'Color','k','LineStyle','--')
%         set(title_1_1,'String',{'Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15)
% 
%         x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
%         y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
%         set(p_1_2,"XData",x_1_2,"YData",y_1_2);
%         set(xline_1_2,'Value',M_1_2,'Color','k','LineStyle','--')
%         set(title_1_2,'String',{'Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15)
% 
%         x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
%         y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
%         set(p_1_3,"XData",x_1_3,"YData",y_1_3);
%         set(xline_1_3,'Value',M_1_3,'Color','k','LineStyle','--')
%         set(title_1_3,'String',{'Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15)
% 
%         x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
%         y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
%         set(p_1_4,"XData",x_1_4,"YData",y_1_4);
%         set(xline_1_4,'Value',M_1_4)
%         set(title_1_4,'String',{'Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15)
%         
%         pause(1)
%         
%     end
% 
% end
% 
% min_std_1_1 = min(std_1(:,1));
% max_std_1_1 = max(std_1(:,1));
% 
% min_std_1_2 = min(std_1(:,2));
% max_std_1_2 = max(std_1(:,2));
% 
% min_std_1_3 = min(std_1(:,3));
% max_std_1_3 = max(std_1(:,3));
% 
% min_std_1_4 = min(std_1(:,4));
% max_std_1_4 = max(std_1(:,4));
% 
% 
% x3_k = x3_k.';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% ylin3_1 = x_dot3(:,1);
% yn3_1 = x3_k(:,1);
% 
% ylin3_2 = x_dot3(:,2);
% yn3_2 = x3_k(:,2);
% 
% ylin3_3 = x_dot3(:,3);
% yn3_3 = x3_k(:,3);
% 
% ylin3_4 = x_dot3(:,4);
% yn3_4 = x3_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt3 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt3');
% subplot(2,2,1);
% plot(time3_2,ylin3_1,'r',time3_2,yn3_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin3_1-yn3_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time3_2,ylin3_2,'r',time3_2,yn3_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin3_2-yn3_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time3_2,ylin3_3,'r',time3_2,yn3_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin3_3-yn3_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time3_2,ylin3_4,'r',time3_2,yn3_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin3_4-yn3_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% % %%%%%%%%%%%%%%%%%%%%%gt3 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure('Name','Normal distribution3');
% % subplot(2,2,1);
% % error3_1 = ylin3_1-yn3_1;
% % [V3_1,M3_1] = var(error3_1);
% % std3_1 = sqrt(V3_1);
% % normal_distribution3_1 = makedist('Normal', 'mu', M3_1, 'sigma', std3_1);
% % x3_1 = linspace(M3_1 - 3 * std3_1, M3_1 + 3 * std3_1, 1000); % 플롯을 위한 x 값 범위
% % y3_1 = pdf(normal_distribution3_1, x3_1); % 확률 밀도 함수 계산
% % plot(x3_1, y3_1,'r', 'LineWidth', 2);
% % xline(M3_1, 'k--');
% % title({'Normal distribution of Lateral position error';['mean= ' num2str(M3_1) ', std= ' num2str(std3_1)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,2);
% % error3_2 = ylin3_2-yn3_2;
% % [V3_2,M3_2] = var(error3_2);
% % std3_2 = sqrt(V3_2);
% % normal_distribution3_2 = makedist('Normal', 'mu', M3_2, 'sigma', std3_2);
% % x3_2 = linspace(M3_2 - 3 * std3_2, M3_2 + 3 * std3_2, 1000); % 플롯을 위한 x 값 범위
% % y3_2 = pdf(normal_distribution3_2, x3_2); % 확률 밀도 함수 계산
% % plot(x3_2, y3_2,'g', 'LineWidth', 2);
% % xline(M3_2, 'k--');
% % title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M3_2) ', std= ' num2str(std3_2)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,3);
% % error3_3 = ylin3_3-yn3_3;
% % [V3_3,M3_3] = var(error3_3);
% % std3_3 = sqrt(V3_3);
% % normal_distribution3_3 = makedist('Normal', 'mu', M3_3, 'sigma', std3_3);
% % x3_3 = linspace(M3_3 - 3 * std3_3, M3_3 + 3 * std3_3, 1000); % 플롯을 위한 x 값 범위
% % y3_3 = pdf(normal_distribution3_3, x3_3); % 확률 밀도 함수 계산
% % plot(x3_3, y3_3,'b', 'LineWidth', 2);
% % xline(M3_3, 'k--');
% % title({'Normal distribution of yaw error';['mean= ' num2str(M3_3) ', std= ' num2str(std3_3)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,4);
% % error3_4 = ylin3_4-yn3_4;
% % [V3_4,M3_4] = var(error3_4);
% % std3_4 = sqrt(V3_4);
% % normal_distribution3_4 = makedist('Normal', 'mu', M3_4, 'sigma', std3_4);
% % x3_4 = linspace(M3_4 - 3 * std3_4, M3_4 + 3 * std3_4, 1000); % 플롯을 위한 x 값 범위
% % y3_4 = pdf(normal_distribution3_4, x3_4); % 확률 밀도 함수 계산
% % plot(x3_4, y3_4,'c', 'LineWidth', 2);
% % xline(M3_4, 'k--');
% % title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M3_4) ', std= ' num2str(std3_4)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sz4=size(time4_1);
% 
% x4_k = zeros(4,sz4(1)); 
% x4 = x4.';
% input_4 = input_4.';
% 
% error_1 = zeros(100,4); 
% 
% figure('Name','gt4 realtime normaldistribution')
% subplot(2,2,1)
% M_1_1=0;
% std_1_1=0;
% normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
% x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
% y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
% p_1_1=plot(x_1_1, y_1_1,'r', 'LineWidth', 2);
% xline_1_1=xline(M_1_1, 'k--');
% title_1_1=title({'initial Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,2)
% M_1_2=0;
% std_1_2=0;
% normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
% x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
% y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
% p_1_2=plot(x_1_2, y_1_2,'r', 'LineWidth', 2);
% xline_1_2=xline(M_1_2, 'k--');
% title_1_2=title({'initial Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,3)
% M_1_3=0;
% std_1_3=0;
% normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
% x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
% y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
% p_1_3=plot(x_1_3, y_1_3,'r', 'LineWidth', 2);
% xline_1_3=xline(M_1_3, 'k--');
% title_1_3=title({'initial Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% subplot(2,2,4)
% M_1_4=0;
% std_1_4=0;
% normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
% x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
% y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
% p_1_4=plot(x_1_4, y_1_4,'r', 'LineWidth', 2);
% xline_1_4=xline(M_1_4, 'k--');
% title_1_4=title({'initial Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15);
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% sizestd_1 = floor(length(time4_1)/100);    %floor(x)는 X의 각 요소를 해당 요소보다 작거나 같은 가장 가까운 정수로 내림
% std_1 = zeros(sizestd_1,4);
% 
% for k = 1:length(time4_1)
%     
%     disp(k)
%     
%     x4_k(:,k) = evaluate(nss,x4(:,k),input_4(:,k)); 
%     
%     if mod(k,100) == 0     %k를 100으로 나눈 나머지가 0인지 확인
% 
%         error_1(:,1) = x_dot4(k-99:1:k,1) - x4_k(1,k-99:1:k).';
%         error_1(:,2) = x_dot4(k-99:1:k,2) - x4_k(2,k-99:1:k).';
%         error_1(:,3) = x_dot4(k-99:1:k,3) - x4_k(3,k-99:1:k).';
%         error_1(:,4) = x_dot4(k-99:1:k,4) - x4_k(4,k-99:1:k).';
%     
%         [V_1_1,M_1_1] = var(error_1(:,1));
%         [V_1_2,M_1_2] = var(error_1(:,2));
%         [V_1_3,M_1_3] = var(error_1(:,3));
%         [V_1_4,M_1_4] = var(error_1(:,4));
%     
%         std_1_1 = sqrt(V_1_1);
%         std_1_2 = sqrt(V_1_2);
%         std_1_3 = sqrt(V_1_3);
%         std_1_4 = sqrt(V_1_4);
% 
%         std_1(k/100,1) = std_1_1;
%         std_1(k/100,2) = std_1_2;
%         std_1(k/100,3) = std_1_3;
%         std_1(k/100,4) = std_1_4;
% 
%         normal_distribution_1_1 = makedist('Normal', 'mu', M_1_1, 'sigma', std_1_1);
%         normal_distribution_1_2 = makedist('Normal', 'mu', M_1_2, 'sigma', std_1_2);
%         normal_distribution_1_3 = makedist('Normal', 'mu', M_1_3, 'sigma', std_1_3);
%         normal_distribution_1_4 = makedist('Normal', 'mu', M_1_4, 'sigma', std_1_4);
%     
%         x_1_1 = linspace(M_1_1 - 3 * std_1_1, M_1_1 + 3 * std_1_1, 1000); % 플롯을 위한 x 값 범위
%         y_1_1 = pdf(normal_distribution_1_1, x_1_1); % 확률 밀도 함수 계산
%         set(p_1_1,"XData",x_1_1,"YData",y_1_1);
%         set(xline_1_1,'Value',M_1_1,'Color','k','LineStyle','--')
%         set(title_1_1,'String',{'Normal distribution of Lateral position error';['mean= ' num2str(M_1_1) ', std= ' num2str(std_1_1)]},'FontSize',15)
% 
%         x_1_2 = linspace(M_1_2 - 3 * std_1_2, M_1_2 + 3 * std_1_2, 1000); % 플롯을 위한 x 값 범위
%         y_1_2 = pdf(normal_distribution_1_2, x_1_2); % 확률 밀도 함수 계산
%         set(p_1_2,"XData",x_1_2,"YData",y_1_2);
%         set(xline_1_2,'Value',M_1_2,'Color','k','LineStyle','--')
%         set(title_1_2,'String',{'Normal distribution of Lateral velocity error';['mean= ' num2str(M_1_2) ', std= ' num2str(std_1_2)]},'FontSize',15)
% 
%         x_1_3 = linspace(M_1_3 - 3 * std_1_3, M_1_3 + 3 * std_1_3, 1000); % 플롯을 위한 x 값 범위
%         y_1_3 = pdf(normal_distribution_1_3, x_1_3); % 확률 밀도 함수 계산
%         set(p_1_3,"XData",x_1_3,"YData",y_1_3);
%         set(xline_1_3,'Value',M_1_3,'Color','k','LineStyle','--')
%         set(title_1_3,'String',{'Normal distribution of Yaw error';['mean= ' num2str(M_1_3) ', std= ' num2str(std_1_3)]},'FontSize',15)
% 
%         x_1_4 = linspace(M_1_4 - 3 * std_1_4, M_1_4 + 3 * std_1_4, 1000); % 플롯을 위한 x 값 범위
%         y_1_4 = pdf(normal_distribution_1_4, x_1_4); % 확률 밀도 함수 계산
%         set(p_1_4,"XData",x_1_4,"YData",y_1_4);
%         set(xline_1_4,'Value',M_1_4)
%         set(title_1_4,'String',{'Normal distribution of Yaw rate error';['mean= ' num2str(M_1_4) ', std= ' num2str(std_1_4)]},'FontSize',15)
%         
%         pause(1)
%         
%     end
%     
% end
% 
% min_std_1_1 = min(std_1(:,1));
% max_std_1_1 = max(std_1(:,1));
% 
% min_std_1_2 = min(std_1(:,2));
% max_std_1_2 = max(std_1(:,2));
% 
% min_std_1_3 = min(std_1(:,3));
% max_std_1_3 = max(std_1(:,3));
% 
% min_std_1_4 = min(std_1(:,4));
% max_std_1_4 = max(std_1(:,4));
% 
% x4_k = x4_k.';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%
% ylin4_1 = x_dot4(:,1);
% yn4_1 = x4_k(:,1);
% 
% ylin4_2 = x_dot4(:,2);
% yn4_2 = x4_k(:,2);
% 
% ylin4_3 = x_dot4(:,3);
% yn4_3 = x4_k(:,3);
% 
% ylin4_4 = x_dot4(:,4);
% yn4_4 = x4_k(:,4);
% 
% %%%%%%%%%%%%%%%%%%%%%%gt4 subplot%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','validation gt4');
% subplot(2,2,1);
% plot(time4_2,ylin4_1,'r',time4_2,yn4_1,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral position';['RMSE = ' num2str(sqrt(mean((ylin4_1-yn4_1).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('y [m]','FontSize',15)
% 
% subplot(2,2,2);
% plot(time4_2,ylin4_2,'r',time4_2,yn4_2,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'Lateral velocity';['RMSE = ' num2str(sqrt(mean((ylin4_2-yn4_2).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dy/dt [m/s]','FontSize',15)
% 
% subplot(2,2,3);
% plot(time4_2,ylin4_3,'r',time4_2,yn4_3,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw';['RMSE = ' num2str(sqrt(mean((ylin4_3-yn4_3).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('ψ [rad]','FontSize',15)
% 
% subplot(2,2,4);
% plot(time4_2,ylin4_4,'r',time4_2,yn4_4,'b--')
% legend("Original","Estimated",'FontSize',15);
% grid on
% title({'yaw angle rate';['RMSE = ' num2str(sqrt(mean((ylin4_4-yn4_4).^2)))]},'FontSize',15)
% xlabel('Time[s]','FontSize',15)
% ylabel('dψ/dt [rad/s]','FontSize',15)
% 
% %%%%%%%%%%%%%%%%%%%%%gt4 정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure('Name','Normal distribution4');
% % subplot(2,2,1);
% % error4_1 = ylin4_1-yn4_1;
% % [V4_1,M4_1] = var(error4_1);
% % std4_1 = sqrt(V4_1);
% % normal_distribution4_1 = makedist('Normal', 'mu', M4_1, 'sigma', std4_1);
% % x4_1 = linspace(M4_1 - 3 * std4_1, M4_1 + 3 * std4_1, 1000); % 플롯을 위한 x 값 범위
% % y4_1 = pdf(normal_distribution4_1, x4_1); % 확률 밀도 함수 계산
% % plot(x4_1, y4_1,'r', 'LineWidth', 2);
% % xline(M4_1, 'k--');
% % title({'Normal distribution of Lateral position error';['mean= ' num2str(M4_1) ', std= ' num2str(std4_1)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,2);
% % error4_2 = ylin4_2-yn4_2;
% % [V4_2,M4_2] = var(error4_2);
% % std4_2 = sqrt(V4_2);
% % normal_distribution4_2 = makedist('Normal', 'mu', M4_2, 'sigma', std4_2);
% % x4_2 = linspace(M4_2 - 3 * std4_2, M4_2 + 3 * std4_2, 1000); % 플롯을 위한 x 값 범위
% % y4_2 = pdf(normal_distribution4_2, x4_2); % 확률 밀도 함수 계산
% % plot(x4_2, y4_2,'g', 'LineWidth', 2);
% % xline(M4_2, 'k--');
% % title({'Normal distribution of Lateral velocity error';['mean= ' num2str(M4_2) ', std= ' num2str(std4_2)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,3);
% % error4_3 = ylin4_3-yn4_3;
% % [V4_3,M4_3] = var(error4_3);
% % std4_3 = sqrt(V4_3);
% % normal_distribution4_3 = makedist('Normal', 'mu', M4_3, 'sigma', std4_3);
% % x4_3 = linspace(M4_3 - 3 * std4_3, M4_3 + 3 * std4_3, 1000); % 플롯을 위한 x 값 범위
% % y4_3 = pdf(normal_distribution4_3, x4_3); % 확률 밀도 함수 계산
% % plot(x4_3, y4_3,'b', 'LineWidth', 2);
% % xline(M4_3, 'k--');
% % title({'Normal distribution of yaw error';['mean= ' num2str(M4_3) ', std= ' num2str(std4_3)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% % 
% % subplot(2,2,4);
% % error4_4 = ylin4_4-yn4_4;
% % [V4_4,M4_4] = var(error4_4);
% % std4_4 = sqrt(V4_4);
% % normal_distribution4_4 = makedist('Normal', 'mu', M4_4, 'sigma', std4_4);
% % x4_4 = linspace(M4_4 - 3 * std4_4, M4_4 + 3 * std4_4, 1000); % 플롯을 위한 x 값 범위
% % y4_4 = pdf(normal_distribution4_4, x4_4); % 확률 밀도 함수 계산
% % plot(x4_4, y4_4,'c', 'LineWidth', 2);
% % xline(M4_4, 'k--');
% % title({'Normal distribution of yaw angle rate error';['mean= ' num2str(M4_4) ', std= ' num2str(std4_4)]},'FontSize',15)
% % xlabel('error');
% % ylabel('probability density');
% % grid on;
% 
% %%%%%%%%%%%%%%%%%%%%%gt1 & gt2 & gt3 & gt4정규분포%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('Name','Normal distribution gt1 & gt2 & gt3 & gt4');
% subplot(2,2,1);
% error1_1 = ylin1_1-yn1_1;
% [V1_1,M1_1] = var(error1_1);
% std1_1 = sqrt(V1_1);
% normal_distribution1_1 = makedist('Normal', 'mu', M1_1, 'sigma', std1_1);
% x1_1 = linspace(M1_1 - 3 * std1_1, M1_1 + 3 * std1_1, 1000); % 플롯을 위한 x 값 범위
% y1_1 = pdf(normal_distribution1_1, x1_1); % 확률 밀도 함수 계산
% 
% error2_1 = ylin2_1-yn2_1;
% [V2_1,M2_1] = var(error2_1);
% std2_1 = sqrt(V2_1);
% normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
% x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
% y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산
% 
% error3_1 = ylin3_1-yn3_1;
% [V3_1,M3_1] = var(error3_1);
% std3_1 = sqrt(V3_1);
% normal_distribution3_1 = makedist('Normal', 'mu', M3_1, 'sigma', std3_1);
% x3_1 = linspace(M3_1 - 3 * std3_1, M3_1 + 3 * std3_1, 1000); % 플롯을 위한 x 값 범위
% y3_1 = pdf(normal_distribution3_1, x3_1); % 확률 밀도 함수 계산
% 
% error4_1 = ylin4_1-yn4_1;
% [V4_1,M4_1] = var(error4_1);
% std4_1 = sqrt(V4_1);
% normal_distribution4_1 = makedist('Normal', 'mu', M4_1, 'sigma', std4_1);
% x4_1 = linspace(M4_1 - 3 * std4_1, M4_1 + 3 * std4_1, 1000); % 플롯을 위한 x 값 범위
% y4_1 = pdf(normal_distribution4_1, x4_1); % 확률 밀도 함수 계산
% 
% plot(x1_1, y1_1,'r--', 'LineWidth', 1);
% hold on;
% plot(x2_1, y2_1,'g--', 'LineWidth', 1);
% hold on;
% plot(x3_1, y3_1,'b--', 'LineWidth', 1);
% hold on;
% plot(x4_1, y4_1,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
% title('Normal distribution of Lateral position error','FontSize',15)
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
% 
% error2_2 = ylin2_2-yn2_2;
% [V2_2,M2_2] = var(error2_2);
% std2_2 = sqrt(V2_2);
% normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
% x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
% y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산
% 
% error3_2 = ylin3_2-yn3_2;
% [V3_2,M3_2] = var(error3_2);
% std3_2 = sqrt(V3_2);
% normal_distribution3_2 = makedist('Normal', 'mu', M3_2, 'sigma', std3_2);
% x3_2 = linspace(M3_2 - 3 * std3_2, M3_2 + 3 * std3_2, 1000); % 플롯을 위한 x 값 범위
% y3_2 = pdf(normal_distribution3_2, x3_2); % 확률 밀도 함수 계산
% 
% error4_2 = ylin4_2-yn4_2;
% [V4_2,M4_2] = var(error4_2);
% std4_2 = sqrt(V4_2);
% normal_distribution4_2 = makedist('Normal', 'mu', M4_2, 'sigma', std4_2);
% x4_2 = linspace(M4_2 - 3 * std4_2, M4_2 + 3 * std4_2, 1000); % 플롯을 위한 x 값 범위
% y4_2 = pdf(normal_distribution4_2, x4_2); % 확률 밀도 함수 계산
% 
% plot(x1_2, y1_2,'r--', 'LineWidth', 1);
% hold on;
% plot(x2_2, y2_2,'g--', 'LineWidth', 1);
% hold on;
% plot(x3_2, y3_2,'b--', 'LineWidth', 1);
% hold on;
% plot(x4_2, y4_2,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
% title('Normal distribution of Lateral velocity error','FontSize',15)
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
% 
% error2_3 = ylin2_3-yn2_3;
% [V2_3,M2_3] = var(error2_3);
% std2_3 = sqrt(V2_3);
% normal_distribution2_3 = makedist('Normal', 'mu', M2_3, 'sigma', std2_3);
% x2_3 = linspace(M2_3 - 3 * std2_3, M2_3 + 3 * std2_3, 1000); % 플롯을 위한 x 값 범위
% y2_3 = pdf(normal_distribution2_3, x2_3); % 확률 밀도 함수 계산
% 
% error3_3 = ylin3_3-yn3_3;
% [V3_3,M3_3] = var(error3_3);
% std3_3 = sqrt(V3_3);
% normal_distribution3_3 = makedist('Normal', 'mu', M3_3, 'sigma', std3_3);
% x3_3 = linspace(M3_3 - 3 * std3_3, M3_3 + 3 * std3_3, 1000); % 플롯을 위한 x 값 범위
% y3_3 = pdf(normal_distribution3_3, x3_3); % 확률 밀도 함수 계산
% 
% error4_3 = ylin4_3-yn4_3;
% [V4_3,M4_3] = var(error4_3);
% std4_3 = sqrt(V4_3);
% normal_distribution4_3 = makedist('Normal', 'mu', M4_3, 'sigma', std4_3);
% x4_3 = linspace(M4_3 - 3 * std4_3, M4_3 + 3 * std4_3, 1000); % 플롯을 위한 x 값 범위
% y4_3 = pdf(normal_distribution4_3, x4_3); % 확률 밀도 함수 계산
% 
% plot(x1_3, y1_3,'r--', 'LineWidth', 1);
% hold on;
% plot(x2_3, y2_3,'g--', 'LineWidth', 1);
% hold on;
% plot(x3_3, y3_3,'b--', 'LineWidth', 1);
% hold on;
% plot(x4_3, y4_3,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
% title('Normal distribution of yaw error','FontSize',15)
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
% 
% error2_4 = ylin2_4-yn2_4;
% [V2_4,M2_4] = var(error2_4);
% std2_4 = sqrt(V2_4);
% normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
% x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
% y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산
% 
% error3_4 = ylin3_4-yn3_4;
% [V3_4,M3_4] = var(error3_4);
% std3_4 = sqrt(V3_4);
% normal_distribution3_4 = makedist('Normal', 'mu', M3_4, 'sigma', std3_4);
% x3_4 = linspace(M3_4 - 3 * std3_4, M3_4 + 3 * std3_4, 1000); % 플롯을 위한 x 값 범위
% y3_4 = pdf(normal_distribution3_4, x3_4); % 확률 밀도 함수 계산
% 
% error4_4 = ylin4_4-yn4_4;
% [V4_4,M4_4] = var(error4_4);
% std4_4 = sqrt(V4_4);
% normal_distribution4_4 = makedist('Normal', 'mu', M4_4, 'sigma', std4_4);
% x4_4 = linspace(M4_4 - 3 * std4_4, M4_4 + 3 * std4_4, 1000); % 플롯을 위한 x 값 범위
% y4_4 = pdf(normal_distribution4_4, x4_4); % 확률 밀도 함수 계산
% 
% plot(x1_4, y1_4,'r--', 'LineWidth', 1);
% hold on;
% plot(x2_4, y2_4,'g--', 'LineWidth', 1);
% hold on;
% plot(x3_4, y3_4,'b--', 'LineWidth', 1);
% hold on;
% plot(x4_4, y4_4,'c--', 'LineWidth', 1);
% legend("gt1","gt2","gt3(training)","gt4(training)");
% title('Normal distribution of yaw angle rate error','FontSize',15)
% xlabel('error');
% ylabel('probability density');
% grid on;
% 
% rmse1_1=sqrt(mean((error1_1).^2));
% rmse2_1=sqrt(mean((error2_1).^2));
% mean_rmse12_1=mean([rmse1_1 rmse2_1]);
% 
% rmse3_1=sqrt(mean((error3_1).^2));
% rmse4_1=sqrt(mean((error4_1).^2));
% mean_rmse34_1=mean([rmse3_1 rmse4_1]);
% 
% rmse1_2=sqrt(mean((error1_2).^2));
% rmse2_2=sqrt(mean((error2_2).^2));
% mean_rmse12_2=mean([rmse1_2 rmse2_2]);
% 
% rmse3_2=sqrt(mean((error3_2).^2));
% rmse4_2=sqrt(mean((error4_2).^2));
% mean_rmse34_2=mean([rmse3_2 rmse4_2]);
% 
% rmse1_3=sqrt(mean((error1_3).^2));
% rmse2_3=sqrt(mean((error2_3).^2));
% mean_rmse12_3=mean([rmse1_3 rmse2_3]);
% 
% rmse3_3=sqrt(mean((error3_3).^2));
% rmse4_3=sqrt(mean((error4_3).^2));
% mean_rmse34_3=mean([rmse3_3 rmse4_3]);
% 
% rmse1_4=sqrt(mean((error1_4).^2));
% rmse2_4=sqrt(mean((error2_4).^2));
% mean_rmse12_4=mean([rmse1_4 rmse2_4]);
% 
% rmse3_4=sqrt(mean((error3_4).^2));
% rmse4_4=sqrt(mean((error4_4).^2));
% mean_rmse34_4=mean([rmse3_4 rmse4_4]);
% 
% mean_std12_1=mean([std1_1 std2_1]);
% mean_std34_1=mean([std3_1 std4_1]);
% mean_std12_2=mean([std1_2 std2_2]);
% mean_std34_2=mean([std3_2 std4_2]);
% mean_std12_3=mean([std1_3 std2_3]);
% mean_std34_3=mean([std3_3 std4_3]);
% mean_std12_4=mean([std1_4 std2_4]);
% mean_std34_4=mean([std3_4 std4_4]);