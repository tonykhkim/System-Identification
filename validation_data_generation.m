%close all; clear; clc;

%% Data
%같은 시점의 data끼리 하나의 condition으로 묶은 것이다. 
condition = 1;
if condition == 1
    gt = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
    can = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-44-04_gv80_v4_2_output_groot_0.bag_extracted.bag';
elseif condition == 2
    gt = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-51-54_gv80_v4_2.bag';   %GT/고속 주회로
    can = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-51-54_gv80_v4_2_output_groot_0.bag_extracted.bag';
elseif condition == 3
    gt = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City
    can = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-04-45_gv80_v4_2_output_groot_0.bag_extracted.bag';
elseif condition == 4
    gt = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-10-49_gv80_v4_2.bag';   %GT/K-City
    can = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-10-49_gv80_v4_2_output_groot_0.bag_extracted.bag';
end
%%학습은 고속 주회로 테스트 데이터로 하고, validation은 k-city 데이터로 하기

%% Read
gt_baginfo = rosbag('info', gt);
can_baginfo = rosbag('info', can);

gt_bag = rosbag(gt);
can_bag = rosbag(can);

gt_mat_bag = readMessages(gt_bag,'DataFormat','struct');
can_mat_bag = readMessages(can_bag,'DataFormat','struct');

gt_time_offset = gt_bag.StartTime;
gt_combined_data = gt_bag.MessageList;
gt_combined_data(:,1) = array2table(gt_bag.MessageList{:,1}-gt_time_offset);
gt_combined_data(:,5) = gt_mat_bag;
gt_combined_data.Properties.VariableNames{5} = 'Data';
gt_topic_list = gt_baginfo.Topics;

can_time_offset = can_bag.StartTime;
can_combined_data = can_bag.MessageList;
can_combined_data(:,1) = array2table(can_bag.MessageList{:,1}-can_time_offset);
can_combined_data(:,5) = can_mat_bag;
can_combined_data.Properties.VariableNames{5} = 'Data';
can_topic_list = can_baginfo.Topics;

%% generate state data

gt_vel=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/vel");
gt_att = select(gt_bag,'Topic','/gt/att');
gt_angvel = select(gt_bag,'Topic','/gt/ang_vel')

msgStructs1=readMessages(gt_vel,"DataFormat","struct");
msgStructs2 = readMessages(gt_att,'DataFormat','struct');
msgStructs3 = readMessages(gt_angvel,'DataFormat','struct');

%msgStructs1{1}
%msgStructs2{1}
%msgStructs3{1}

yaw = cellfun(@(m) double(m.Vector.Z),msgStructs2);
yaw_dot = cellfun(@(m) double(m.Vector.Z),msgStructs3);

ts=timeseries(gt_vel,'Vector.Y')
y_dot=ts.Data(:);

time=ts.Time(:)-ts.Time(1);
time_diff=diff(time);
time_diff(end+1)=time_diff(end);
temp = y_dot .* time_diff;
y = cumsum(temp);

%%%%%%%%%%%%%%%%%%% new y prediction %%%%%%%%%%%%%%%%%%
bagselect_yaw_dot=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/ang_vel");
msgs_yaw_dot=readMessages(bagselect_yaw_dot,"DataFormat","struct");
%size(msgs_yaw_dot)

%msgs{2}

ts_yaw_dot=timeseries(bagselect_yaw_dot,'Vector.Z')
time_yaw_dot=ts_yaw_dot.Time(:)-ts_yaw_dot.Time(1);
new_yaw_dot=ts_yaw_dot.Data(:);
time_diff_yaw_dot=diff(time_yaw_dot);
time_diff_yaw_dot(end+1)=time_diff_yaw_dot(end);
temp_yaw = new_yaw_dot .* time_diff_yaw_dot;
new_yaw = cumsum(temp_yaw);

state_data = cat(2,y,y_dot,new_yaw,yaw_dot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(5,1,1);
plot(time,y)
title('Lateral position')
xlabel('Time[s]')
ylabel('y [m]')

subplot(5,1,2);
plot(time,y_dot)
title('Lateral velocity')
xlabel('Time[s]')
ylabel('dy/dt [m/s]')

subplot(5,1,3);
plot(time,yaw)
title('Yaw')
xlabel('Time[s]')
ylabel('ψ [rad]')

subplot(5,1,4);
plot(time,yaw_dot)
title('Yaw rate')
xlabel('Time[s]')
ylabel('dψ/dt [rad/s]')

subplot(5,1,5);
plot(time_yaw_dot,new_yaw)
title('New Yaw')
xlabel('Time[s]')
ylabel('ψ [rad]')

%% generate input data 
steering = select(can_bag,'Topic','/groot/chassis_can/platform_standard');
msgStructs4 = readMessages(steering,'DataFormat','struct');
msgStructs4{1}
input_data = cellfun(@(m) double(m.WsaDeg),msgStructs4);
