%close all; clear; clc;

%% Data
%같은 시점의 data끼리 하나의 condition으로 묶은 것이다. 

gt1 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-44-04_gv80_v4_2.bag';   %GT/고속 주회로
can1 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-44-04_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt2 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-16-51-54_gv80_v4_2.bag';   %GT/고속 주회로
can2 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-16-51-54_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt3 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-04-45_gv80_v4_2.bag';   %GT/K-City
can3 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-04-45_gv80_v4_2_output_groot_0.bag_extracted.bag';

gt4 = 'C:\230619_KATRI_Resend\GT\result_23-06-19-17-10-49_gv80_v4_2.bag';   %GT/K-City
can4 = 'C:\230619_KATRI_Resend\SHARE_BAG\2023-06-19-17-10-49_gv80_v4_2_output_groot_0.bag_extracted.bag';

%%학습은 고속 주회로 테스트 데이터로 하고, validation은 k-city 데이터로 하기

%% Read GT1 data & CAN1 data
% gt1_baginfo = rosbag('info', gt);
% can1_baginfo = rosbag('info', can);
% 
% gt_bag = rosbag(gt);
% can_bag = rosbag(can);
% 
% gt_mat_bag = readMessages(gt_bag,'DataFormat','struct');
% can_mat_bag = readMessages(can_bag,'DataFormat','struct');
% 
% gt_time_offset = gt_bag.StartTime;
% gt_combined_data = gt_bag.MessageList;
% gt_combined_data(:,1) = array2table(gt_bag.MessageList{:,1}-gt_time_offset);
% gt_combined_data(:,5) = gt_mat_bag;
% gt_combined_data.Properties.VariableNames{5} = 'Data';
% gt_topic_list = gt_baginfo.Topics;
% 
% can_time_offset = can_bag.StartTime;
% can_combined_data = can_bag.MessageList;
% can_combined_data(:,1) = array2table(can_bag.MessageList{:,1}-can_time_offset);
% can_combined_data(:,5) = can_mat_bag;
% can_combined_data.Properties.VariableNames{5} = 'Data';
% can_topic_list = can_baginfo.Topics;

%%%%%%%%%%%%%%%%%%%%%%%Generate Training Data%%%%%%%%%%%%%%%%%%%%%%

%% generate state1 data
gt1_bag = rosbag(gt1);
can1_bag = rosbag(can1);

gt1_vel=select(gt1_bag,"Time",[gt1_bag.StartTime gt1_bag.EndTime],"Topic","/gt/vel");
gt1_att = select(gt1_bag,'Topic','/gt/att');
gt1_angvel = select(gt1_bag,'Topic','/gt/ang_vel')

msg1Structs1=readMessages(gt1_vel,"DataFormat","struct");
msg1Structs2 = readMessages(gt1_att,'DataFormat','struct');
msg1Structs3 = readMessages(gt1_angvel,'DataFormat','struct');

%msgStructs1{1}
%msgStructs2{1}
%msgStructs3{1}

yaw1 = cellfun(@(m) double(m.Vector.Z),msg1Structs2);
yaw1_dot = cellfun(@(m) double(m.Vector.Z),msg1Structs3);

ts1=timeseries(gt1_vel,'Vector.Y')
y1_dot=ts1.Data(:);

time1=ts1.Time(:)-ts1.Time(1);
time1_diff=diff(time1);
time1_diff(end+1)=time1_diff(end);
temp1 = y1_dot .* time1_diff;
y1 = cumsum(temp1);

%%%%%%%%%%%%%%%%%%% new y1 prediction %%%%%%%%%%%%%%%%%%
bagselect1_yaw_dot=select(gt1_bag,"Time",[gt1_bag.StartTime gt1_bag.EndTime],"Topic","/gt/ang_vel");
msgs1_yaw_dot=readMessages(bagselect1_yaw_dot,"DataFormat","struct");
%size(msgs_yaw_dot)

%msgs{2}

ts1_yaw_dot=timeseries(bagselect1_yaw_dot,'Vector.Z')
time1_yaw_dot=ts1_yaw_dot.Time(:)-ts1_yaw_dot.Time(1);
new1_yaw_dot=ts1_yaw_dot.Data(:);
time1_diff_yaw_dot=diff(time1_yaw_dot);
time1_diff_yaw_dot(end+1)=time1_diff_yaw_dot(end);
temp1_yaw = new1_yaw_dot .* time1_diff_yaw_dot;
new1_yaw = cumsum(temp1_yaw);

state1_data = cat(2,y1,y1_dot,new1_yaw,yaw1_dot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subplot(5,1,1);
% plot(time1,y1)
% title('Lateral position')
% xlabel('Time[s]')
% ylabel('y [m]')
% 
% subplot(5,1,2);
% plot(time1,y1_dot)
% title('Lateral velocity')
% xlabel('Time[s]')
% ylabel('dy/dt [m/s]')
% 
% subplot(5,1,3);
% plot(time1,yaw1)
% title('Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')
% 
% subplot(5,1,4);
% plot(time1,yaw1_dot)
% title('Yaw rate')
% xlabel('Time[s]')
% ylabel('dψ/dt [rad/s]')
% 
% subplot(5,1,5);
% plot(time1_yaw_dot,new1_yaw)
% title('New Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')

%% generate input1 data 
steering1 = select(can1_bag,'Topic','/groot/chassis_can/platform_standard');
msg1Structs4 = readMessages(steering1,'DataFormat','struct');
msg1Structs4{1}
input1_data = cellfun(@(m) double(m.WsaDeg),msg1Structs4);
input1_data(44701:end)=[];

%% generate state2 data
gt2_bag = rosbag(gt2);
can2_bag = rosbag(can2);

gt2_vel=select(gt2_bag,"Time",[gt2_bag.StartTime gt2_bag.EndTime],"Topic","/gt/vel");
gt2_att = select(gt2_bag,'Topic','/gt/att');
gt2_angvel = select(gt2_bag,'Topic','/gt/ang_vel')

msg2Structs1=readMessages(gt2_vel,"DataFormat","struct");
msg2Structs2 = readMessages(gt2_att,'DataFormat','struct');
msg2Structs3 = readMessages(gt2_angvel,'DataFormat','struct');

%msgStructs1{1}
%msgStructs2{1}
%msgStructs3{1}

yaw2 = cellfun(@(m) double(m.Vector.Z),msg2Structs2);
yaw2_dot = cellfun(@(m) double(m.Vector.Z),msg2Structs3);

ts2=timeseries(gt2_vel,'Vector.Y')
y2_dot=ts2.Data(:);

time2=ts2.Time(:)-ts2.Time(1);
time2_diff=diff(time2);
time2_diff(end+1)=time2_diff(end);
temp2 = y2_dot .* time2_diff;
y2 = cumsum(temp2);

%%%%%%%%%%%%%%%%%%% new y1 prediction %%%%%%%%%%%%%%%%%%
bagselect2_yaw_dot=select(gt2_bag,"Time",[gt2_bag.StartTime gt2_bag.EndTime],"Topic","/gt/ang_vel");
msgs2_yaw_dot=readMessages(bagselect2_yaw_dot,"DataFormat","struct");
%size(msgs_yaw_dot)

%msgs{2}

ts2_yaw_dot=timeseries(bagselect2_yaw_dot,'Vector.Z')
time2_yaw_dot=ts2_yaw_dot.Time(:)-ts2_yaw_dot.Time(1);
new2_yaw_dot=ts2_yaw_dot.Data(:);
time2_diff_yaw_dot=diff(time2_yaw_dot);
time2_diff_yaw_dot(end+1)=time2_diff_yaw_dot(end);
temp2_yaw = new2_yaw_dot .* time2_diff_yaw_dot;
new2_yaw = cumsum(temp2_yaw);

state2_data = cat(2,y2,y2_dot,new2_yaw,yaw2_dot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subplot(5,1,1);
% plot(time,y)
% title('Lateral position')
% xlabel('Time[s]')
% ylabel('y [m]')
% 
% subplot(5,1,2);
% plot(time,y_dot)
% title('Lateral velocity')
% xlabel('Time[s]')
% ylabel('dy/dt [m/s]')
% 
% subplot(5,1,3);
% plot(time,yaw)
% title('Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')
% 
% subplot(5,1,4);
% plot(time,yaw_dot)
% title('Yaw rate')
% xlabel('Time[s]')
% ylabel('dψ/dt [rad/s]')
% 
% subplot(5,1,5);
% plot(time_yaw_dot,new_yaw)
% title('New Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')

%% generate input2 data 
steering2 = select(can2_bag,'Topic','/groot/chassis_can/platform_standard');
msg2Structs4 = readMessages(steering2,'DataFormat','struct');
msg2Structs4{1}
input2_data = cellfun(@(m) double(m.WsaDeg),msg2Structs4);
input2_data(45825:end)=[];

%%%%%%%%%%%%%%%%%%%%%%%%Generate Validation Data%%%%%%%%%%%%%%%%%%%%

%% generate state3 data
gt3_bag = rosbag(gt3);
can3_bag = rosbag(can3);

gt3_vel=select(gt3_bag,"Time",[gt3_bag.StartTime gt3_bag.EndTime],"Topic","/gt/vel");
gt3_att = select(gt3_bag,'Topic','/gt/att');
gt3_angvel = select(gt3_bag,'Topic','/gt/ang_vel')

msg3Structs1=readMessages(gt3_vel,"DataFormat","struct");
msg3Structs2 = readMessages(gt3_att,'DataFormat','struct');
msg3Structs3 = readMessages(gt3_angvel,'DataFormat','struct');

%msgStructs1{1}
%msgStructs2{1}
%msgStructs3{1}

yaw3 = cellfun(@(m) double(m.Vector.Z),msg3Structs2);
yaw3_dot = cellfun(@(m) double(m.Vector.Z),msg3Structs3);

ts3=timeseries(gt3_vel,'Vector.Y')
y3_dot=ts3.Data(:);

time3=ts3.Time(:)-ts3.Time(1);
time3_diff=diff(time3);
time3_diff(end+1)=time3_diff(end);
temp3 = y3_dot .* time3_diff;
y3 = cumsum(temp3);

%%%%%%%%%%%%%%%%%%% new y3 prediction %%%%%%%%%%%%%%%%%%
bagselect3_yaw_dot=select(gt3_bag,"Time",[gt3_bag.StartTime gt3_bag.EndTime],"Topic","/gt/ang_vel");
msgs3_yaw_dot=readMessages(bagselect3_yaw_dot,"DataFormat","struct");
%size(msgs_yaw_dot)

%msgs{2}

ts3_yaw_dot=timeseries(bagselect3_yaw_dot,'Vector.Z')
time3_yaw_dot=ts3_yaw_dot.Time(:)-ts3_yaw_dot.Time(1);
new3_yaw_dot=ts3_yaw_dot.Data(:);
time3_diff_yaw_dot=diff(time3_yaw_dot);
time3_diff_yaw_dot(end+1)=time3_diff_yaw_dot(end);
temp3_yaw = new3_yaw_dot .* time3_diff_yaw_dot;
new3_yaw = cumsum(temp3_yaw);

state3_data = cat(2,y3,y3_dot,new3_yaw,yaw3_dot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subplot(5,1,1);
% plot(time,y)
% title('Lateral position')
% xlabel('Time[s]')
% ylabel('y [m]')
% 
% subplot(5,1,2);
% plot(time,y_dot)
% title('Lateral velocity')
% xlabel('Time[s]')
% ylabel('dy/dt [m/s]')
% 
% subplot(5,1,3);
% plot(time,yaw)
% title('Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')
% 
% subplot(5,1,4);
% plot(time,yaw_dot)
% title('Yaw rate')
% xlabel('Time[s]')
% ylabel('dψ/dt [rad/s]')
% 
% subplot(5,1,5);
% plot(time_yaw_dot,new_yaw)
% title('New Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')

%% generate input3 data 
steering3 = select(can3_bag,'Topic','/groot/chassis_can/platform_standard');
msg3Structs4 = readMessages(steering3,'DataFormat','struct');
msg3Structs4{1}
input3_data = cellfun(@(m) double(m.WsaDeg),msg3Structs4);
input3_data(35260:end)=[];

%% generate state4 data
gt4_bag = rosbag(gt4);
can4_bag = rosbag(can4);

gt4_vel=select(gt4_bag,"Time",[gt4_bag.StartTime gt4_bag.EndTime],"Topic","/gt/vel");
gt4_att = select(gt4_bag,'Topic','/gt/att');
gt4_angvel = select(gt4_bag,'Topic','/gt/ang_vel')

msg4Structs1=readMessages(gt4_vel,"DataFormat","struct");
msg4Structs2 = readMessages(gt4_att,'DataFormat','struct');
msg4Structs3 = readMessages(gt4_angvel,'DataFormat','struct');

%msgStructs1{1}
%msgStructs2{1}
%msgStructs3{1}

yaw4 = cellfun(@(m) double(m.Vector.Z),msg4Structs2);
yaw4_dot = cellfun(@(m) double(m.Vector.Z),msg4Structs3);

ts4=timeseries(gt4_vel,'Vector.Y')
y4_dot=ts4.Data(:);

time4=ts4.Time(:)-ts4.Time(1);
time4_diff=diff(time4);
time4_diff(end+1)=time4_diff(end);
temp4 = y4_dot .* time4_diff;
y4 = cumsum(temp4);

%%%%%%%%%%%%%%%%%%% new y4 prediction %%%%%%%%%%%%%%%%%%
bagselect4_yaw_dot=select(gt4_bag,"Time",[gt4_bag.StartTime gt4_bag.EndTime],"Topic","/gt/ang_vel");
msgs4_yaw_dot=readMessages(bagselect4_yaw_dot,"DataFormat","struct");
%size(msgs_yaw_dot)

%msgs{2}

ts4_yaw_dot=timeseries(bagselect4_yaw_dot,'Vector.Z')
time4_yaw_dot=ts4_yaw_dot.Time(:)-ts4_yaw_dot.Time(1);
new4_yaw_dot=ts4_yaw_dot.Data(:);
time4_diff_yaw_dot=diff(time4_yaw_dot);
time4_diff_yaw_dot(end+1)=time4_diff_yaw_dot(end);
temp4_yaw = new4_yaw_dot .* time4_diff_yaw_dot;
new4_yaw = cumsum(temp4_yaw);

state4_data = cat(2,y4,y4_dot,new4_yaw,yaw4_dot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subplot(5,1,1);
% plot(time,y)
% title('Lateral position')
% xlabel('Time[s]')
% ylabel('y [m]')
% 
% subplot(5,1,2);
% plot(time,y_dot)
% title('Lateral velocity')
% xlabel('Time[s]')
% ylabel('dy/dt [m/s]')
% 
% subplot(5,1,3);
% plot(time,yaw)
% title('Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')
% 
% subplot(5,1,4);
% plot(time,yaw_dot)
% title('Yaw rate')
% xlabel('Time[s]')
% ylabel('dψ/dt [rad/s]')
% 
% subplot(5,1,5);
% plot(time_yaw_dot,new_yaw)
% title('New Yaw')
% xlabel('Time[s]')
% ylabel('ψ [rad]')

%% generate input4 data 
steering4 = select(can4_bag,'Topic','/groot/chassis_can/platform_standard');
msg4Structs4 = readMessages(steering4,'DataFormat','struct');
msg4Structs4{1}
input4_data = cellfun(@(m) double(m.WsaDeg),msg4Structs4);
input4_data(29544:end)=[];
