close all; clear; clc;

%% Data
%같은 시점의 data끼리 하나의 condition으로 묶은 것이다. 
%Logging data는 100Hz 로 샘플링 되었다. 
% 주파수 100Hz = 주기 0.01초

gt1 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
can1 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-44-04_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt2 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-51-54_gv80_v4_2.bag';   %GT/고속 주회로
can2 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-51-54_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt3 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City/
can3 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-04-45_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt4 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-10-49_gv80_v4_2.bag';   %GT/K-City
can4 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-10-49_gv80_v4_2_output_groot_0.bag_extracted.bag';
% 
% gt5 = "C:\20231012_INU_seconddata\general_Airport-JungmunComplex(Solati V5-4)\resim_2023-09-15-12-06-49_solati_v5_4.bag";    %일반날씨/제주공항-중문단지/(장거리, 고속화도로)구간(Solati V5-4)
% gt6 = "C:\20231012_INU_seconddata\general_Airport-JungmunComplex(Solati V5-4)\resim_2023-09-22-12-13-21_solati_v5_4.bag";   %일반날씨/제주공항-중문단지/(장거리, 고속화도로)구간(Solati V5-4)
% gt7 = "C:\20231012_INU_seconddata\general_Airport-JungmunComplex(Solati V5-4)\resim_2023-09-26-15-02-50_solati_v5_4.bag";    %일반날씨/제주공항-중문단지/(장거리, 고속화도로)구간(Solati V5-4)
% 
% gt8 = "C:\20231012_INU_seconddata\rainy_Airport-JungmunComplex(Solati V5-4)\resim_2023-09-14-11-21-24_solati_v5_4.bag";      %비/제주공항-중문단지/(장거리, 고속화도로)구간(Solati V5-4)
% gt9 = "C:\20231012_INU_seconddata\rainy_Airport-JungmunComplex(Solati V5-4)\resim_2023-09-14-12-16-38_solati_v5_4.bag";     %비/제주공항-중문단지/(장거리, 고속화도로)구간(Solati V5-4)
% 
% gt10 = "C:\20231012_INU_seconddata\nearAirport(Solati V5-5)\resim_2023-09-14-11-22-04_solati_v5_5.bag";      %일반 날씨/제주공항 주변 해안도로/(단거리, 도심지)구간(Solati V5-5)
% gt11 = "C:\20231012_INU_seconddata\nearAirport(Solati V5-5)\resim_2023-09-19-11-20-38_solati_v5_5.bag";      %일반 날씨/제주공항 주변 해안도로/(단거리, 도심지)구간(Solati V5-5)
% gt12 = "C:\20231012_INU_seconddata\nearAirport(Solati V5-5)\resim_2023-09-22-16-09-05_solati_v5_5.bag";      %일반 날씨/제주공항 주변 해안도로/(단거리, 도심지)구간(Solati V5-5)
% 
% gt13 = "C:\20231221_INU_thirddata\nearAirport(Solati V5-5)\resim_2023-12-21-15-39-43_solati_v5_5.bag";      %눈 날씨/ 제주공항 외곽 도로 / 경사로 + 급격한 우회전 구간 (Solati V5-5 )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 기존의 stategenerationfunction과 inputgenerationfunction
% [state1_data,time1]=stategenerationfunction(gt1,'gt1');
% input1_data=inputgenerationfunction(can1);
% input1_data(44701:end)=[];

% 종방향 속도와 WSA 정보를 포함한 각 topic data를 개별적으로 추출하는 datagenerationfunction_forPython
[Vel_Y,Vel_X1,Pos_Y,Yaw_Rate,Yaw,Acc_Y,WSA_rad,time]=datagenerationfunction_forPython(gt1,'gt1',can1);
WSA_rad(44701:end)=[];
WSAplotfunction(WSA_rad,time,'can1');
% % save('gt1_forPython.mat','Pos_Y','time','Vel_X','Vel_Y','WSA_rad','Yaw','Yaw_Rate','Acc_Y')

% [min_Vx_gt1, max_Vx_gt1, min_Ay_gt1, max_Ay_gt1, min_WSA_gt1, max_WSA_gt1] = moreinfo_generation_function(gt1,can1,'gt1&can1');
% analFullmotion(gt1,input1_data,'gt1')

% input1_data_inverse(44701:end)=[];
% inputplotfunction('can1',input1_data,time1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 기존의 stategenerationfunction과 inputgenerationfunction
% [state2_data, time2]=stategenerationfunction(gt2,'gt2');
% input2_data=inputgenerationfunction(can2);
% input2_data(45825:end)=[];

% 종방향 속도와 WSA 정보를 포함한 각 topic data를 개별적으로 추출하는 datagenerationfunction_forPython
% [Vel_Y,Vel_X2,Pos_Y,Yaw_Rate,Yaw,Acc_Y,WSA_rad,time]=datagenerationfunction_forPython(gt2,'gt2',can2);
% WSA_rad(45825:end)=[];
% WSAplotfunction(WSA_rad,time,'can2');
% % save('gt2_forPython.mat','Pos_Y','time','Vel_X','Vel_Y','WSA_rad','Yaw','Yaw_Rate','Acc_Y')

% [min_Vx_gt2, max_Vx_gt2, min_Ay_gt2, max_Ay_gt2, min_WSA_gt2, max_WSA_gt2] = moreinfo_generation_function(gt2,can2,'gt2 & can2');
% analFullmotion(gt2,input2_data,'gt2')

% input2_data_inverse(45825:end)=[];
% inputplotfunction('can2',input2_data,time2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 기존의 stategenerationfunction과 inputgenerationfunction
% [state3_data, time3]=stategenerationfunction(gt3,'gt3');
% input3_data =inputgenerationfunction(can3);
% input3_data(35260:end)=[];

% 종방향 속도와 WSA 정보를 포함한 각 topic data를 개별적으로 추출하는 datagenerationfunction_forPython
% [Vel_Y,Vel_X3,Pos_Y,Yaw_Rate,Yaw,Acc_Y,WSA_rad,time]=datagenerationfunction_forPython(gt3,'gt3',can3);
% WSA_rad(35260:end)=[];
% WSAplotfunction(WSA_rad,time,'can3');
% % save('gt3_forPython.mat','Pos_Y','time','Vel_X','Vel_Y','WSA_rad','Yaw','Yaw_Rate','Acc_Y')

% [min_Vx_gt3, max_Vx_gt3, min_Ay_gt3, max_Ay_gt3, min_WSA_gt3, max_WSA_gt3] = moreinfo_generation_function(gt3,can3,'gt3 & can3');
% [high_min_Long_vel, high_max_Long_vel, Kcity_min_Long_vel, Kcity_max_Long_vel, high_min_WSA, high_max_WSA, Kcity_min_WSA, Kcity_max_WSA] = anal_Vx_WSA_index_function(gt1,gt3,can1,can3,'gt1&gt3 Vx, δ')
% [high_min_Long_vel2, high_max_Long_vel2, Kcity_min_Long_vel2, Kcity_max_Long_vel2, high_min_WSA2, high_max_WSA2, Kcity_min_WSA2, Kcity_max_WSA2] = anal_Vx_WSA_index_function(gt2,gt4,can2,can4,'gt2&gt4 Vx, δ')
% analFullmotion(gt3,input3_data,'gt3')

% % input3_data_inverse(35260:end)=[];
% % inputplotfunction('can3',input3_data,time3);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gt4%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 기존의 stategenerationfunction과 inputgenerationfunction
% [state4_data, time4]=stategenerationfunction(gt4,'gt4');
% input4_data=inputgenerationfunction(can4);
% input4_data(29544:end)=[];

% 종방향 속도와 WSA 정보를 포함한 각 topic data를 개별적으로 추출하는 datagenerationfunction_forPython
% [Vel_Y,Vel_X4,Pos_Y,Yaw_Rate,Yaw,Acc_Y,WSA_rad,time]=datagenerationfunction_forPython(gt4,'gt4',can4);
% WSA_rad(29544:end)=[];
% WSAplotfunction(WSA_rad,time,'can4');
% % save('gt4_forPython.mat','Pos_Y','time','Vel_X','Vel_Y','WSA_rad','Yaw','Yaw_Rate','Acc_Y')

% [min_Vx_gt4, max_Vx_gt4, min_Ay_gt4, max_Ay_gt4, min_WSA_gt4, max_WSA_gt4] = moreinfo_generation_function(gt4,can4,'gt4 & can4');
% analFullmotion(gt4,input4_data,'gt4')

% % input4_data_inverse(29544:end)=[];
% % inputplotfunction('can4',input4_data,time4);
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%ICROS 논문 분석용 %%%%%%%%%%%%%%%%%%%%%
% anal_Vx_WSA_ICROS_function(gt1,gt4,can1,can4)