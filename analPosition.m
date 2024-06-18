close all; clc; clear all;

gt1 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
can1 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-44-04_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt2 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-51-54_gv80_v4_2.bag';   %GT/고속 주회로
can2 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-51-54_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt3 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City/
can3 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-04-45_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt4 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-10-49_gv80_v4_2.bag';   %GT/K-City
can4 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-10-49_gv80_v4_2_output_groot_0.bag_extracted.bag';

input1_data=inputgenerationfunction(can1);
input1_data(44701:end)=[];

input3_data =inputgenerationfunction(can3);
input3_data(35260:end)=[];

input4_data=inputgenerationfunction(can4);
input4_data(29544:end)=[];

% %gt1을 input으로 했을 때
% load("index_all.mat");  %모든 state에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우 
% load("index1.mat");     %Lateral Position에서만 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index2.mat");     %Lateral Velocity에서만 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index3.mat");     %Yaw에서만 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index4.mat");     %Yaw Rate에서만 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index12.mat");    %Lateral Position, Lateral veloctiy 에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index13.mat");    %Lateral Position, Yaw에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index14.mat");    %Lateral Position, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index23.mat");    %Lateral Velocity, Yaw 에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index24.mat");    %Lateral Velocity, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index34.mat");    %Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
load("index123.mat");   %Lateral Position, Lateral veloctiy, Yaw에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index124.mat");   %Lateral Position, Lateral veloctiy, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index134.mat");   %Lateral Position, Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우
% load("index234.mat");   %Lateral Veloctiy, Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 클 경우

%gt3을 input으로 했을 때
% load("inputgt3_index_all.mat");  %모든 state에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우 
% load("inputgt3_index1.mat");     %Lateral Position에서만 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index2.mat");     %Lateral Velocity에서만 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index3.mat");     %Yaw에서만 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index4.mat");     %Yaw Rate에서만 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index12.mat");    %Lateral Position, Lateral veloctiy 에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index13.mat");    %Lateral Position, Yaw에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index14.mat");    %Lateral Position, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index23.mat");    %Lateral Velocity, Yaw 에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index24.mat");    %Lateral Velocity, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index34.mat");    %Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index123.mat");   %Lateral Position, Lateral veloctiy, Yaw에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index124.mat");   %Lateral Position, Lateral veloctiy, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index134.mat");   %Lateral Position, Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우
% load("inputgt3_index234.mat");   %Lateral Veloctiy, Yaw, Yaw Rate에서 표준편차가 고속주회로 모델이 k-city모델보다 작을 경우

% % 비워있는 인덱스에 값 추가
% index_all(1438:1437+99) = 1;
% index_all(1913:1912+99) = 1;
% 
% index1(2:1+99) = 1;
% index1(613:612+99) = 1;
% index1(678:677+99) = 1;
% index1(809:808+99) = 1;
% index1(903:902+99) = 1;
% index1(1030:1034) = 1;
% index1(1037:1039) = 1;
% index1(1042:1098) = 1;
% index1(1140:1191) = 1;
% index1(1250:1278) = 1;
% index1(1364:1370) = 1;
% index1(1409:1408+99) = 1;
% index1(1533:1532+103) = 1;
% index1(1644:1643+99) = 1;
% index1(1650:1649+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% index1(:+99) = 1;
% 
% k = find(index_all);

% gt1_trajectory_all = PloatRoad_index(gt1,index_all,'index all');
% gt1_trajectory_1 = PloatRoad_index(gt1,index1,'index1');
% gt1_trajectory_2 = PloatRoad_index(gt1,index2,'index2');
% gt1_trajectory_3 = PloatRoad_index(gt1,index3,'index3');
% gt1_trajectory_4 = PloatRoad_index(gt1,index4,'index4');
% gt1_trajectory_12 = PloatRoad_index(gt1,index12,'index12');
% gt1_trajectory_13 = PloatRoad_index(gt1,index13,'index13');
% gt1_trajectory_14 = PloatRoad_index(gt1,index14,'index14');
% gt1_trajectory_23 = PloatRoad_index(gt1,index23,'index23');
% gt1_trajectory_24 = PloatRoad_index(gt1,index24,'index24');
% gt1_trajectory_34 = PloatRoad_index(gt1,index34,'index34');
gt1_trajectory_123 = PloatRoad_index(gt1,index123,'index123');
% gt1_trajectory_124 = PloatRoad_index(gt1,index124,'index124');
% gt1_trajectory_134 = PloatRoad_index(gt1,index134,'index134');
% gt1_trajectory_234 = PloatRoad_index(gt1,index234,'index234');

% gt3_trajectory_all = PloatRoad_index(gt3,index_all,'input gt3 index all');
% gt3_trajectory_1 = PloatRoad_index(gt3,index1,'input gt3 index1');
% gt3_trajectory_2 = PloatRoad_index(gt3,index2,'input gt3 index2');
% gt3_trajectory_3 = PloatRoad_index(gt3,index3,'input gt3 index3');
% gt3_trajectory_4 = PloatRoad_index(gt3,index4,'input gt3 index4');
% gt3_trajectory_12 = PloatRoad_index(gt3,index12,'input gt3 index12');
% gt3_trajectory_13 = PloatRoad_index(gt3,index13,'input gt3 index13');
% gt3_trajectory_14 = PloatRoad_index(gt3,index14,'input gt3 index14');
% gt3_trajectory_23 = PloatRoad_index(gt3,index23,'input gt3 index23');
% gt3_trajectory_24 = PloatRoad_index(gt3,index24,'input gt3 index24');
% gt3_trajectory_34 = PloatRoad_index(gt3,index34,'input gt3 index34');
% gt3_trajectory_123 = PloatRoad_index(gt3,index123,'input gt3 index123');
% gt3_trajectory_124 = PloatRoad_index(gt3,index124,'input gt3 index124');
% gt3_trajectory_134 = PloatRoad_index(gt3,index134,'input gt3 index134');
% gt3_trajectory_234 = PloatRoad_index(gt3,index234,'input gt3 index234');

gt_vel=select(rosbag(gt1),"Time",[rosbag(gt1).StartTime rosbag(gt1).EndTime],"Topic","/gt/vel");
ts=timeseries(gt_vel,'Vector.Y');
gt1_time=ts.Time(:)-ts.Time(1);
time_index = gt1_time(k);   %index_all에서 1이 아닌 값을 가진 인덱스에 해당하는 시간 계산

[minmax_input1, minmax_vel_x1]= analLocalmotion(gt1,input1_data,k(1:101));
[minmax_input2, minmax_vel_x2]= analLocalmotion(gt1,input1_data,k(102:end));
analFullmotion2(gt3,input3_data,minmax_input1,'gt3')
analFullmotion2(gt3,input3_data,minmax_input2,'gt3')

gt1_trajectory1 = PloatRoad_index(gt1,index1);
gt1_trajectory2 = PloatRoad_index(gt1,index2);
gt1_trajectory3 = PloatRoad_index(gt1,index3);
gt1_trajectory4 = PloatRoad_index(gt1,index4);

gt1_trajectory = PloatRoad(gt1);
gt2_trajectory = PloatRoad(gt2);
gt3_trajectory = PloatRoad(gt3);
gt4_trajectory = PloatRoad(gt4);