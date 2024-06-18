function [high_min_Long_vel, high_max_Long_vel, Kcity_min_Long_vel, Kcity_max_Long_vel, high_min_WSA, high_max_WSA, Kcity_min_WSA, Kcity_max_WSA] = anal_Vx_WSA_index_function(high_speed_gt,Kcity_gt,high_speed_can,Kcity_can,name)
    high_gt_bag = rosbag(high_speed_gt);
    Kcity_gt_bag = rosbag(Kcity_gt);
    high_can = rosbag(high_speed_can);
    Kcity_can = rosbag(Kcity_can);
    
    % 분석하고 싶은 구간
    range = 18101:44699;
%     range = 10501:29542;

    %Longitudinal velocity(Vx) for high speed circuit gt data
    high_gt_vel=select(high_gt_bag,"Time",[high_gt_bag.StartTime high_gt_bag.EndTime],"Topic","/gt/vel");
    high_ts=timeseries(high_gt_vel,'Vector.X');
    high_Long_vel=high_ts.Data(:);
    high_Long_vel=high_Long_vel(range,:);
    high_min_Long_vel = min(high_Long_vel);
    high_max_Long_vel = max(high_Long_vel);
    
    %Longitudinal velocity(Vx) for low speed K-city gt data
    Kcity_gt_vel=select(Kcity_gt_bag,"Time",[Kcity_gt_bag.StartTime Kcity_gt_bag.EndTime],"Topic","/gt/vel");
    Kcity_ts=timeseries(Kcity_gt_vel,'Vector.X');
    Kcity_Long_vel=Kcity_ts.Data(:);
    Kcity_Long_vel=Kcity_Long_vel(range,:);
    Kcity_min_Long_vel = min(Kcity_Long_vel);
    Kcity_max_Long_vel = max(Kcity_Long_vel);

    % time
    time=high_ts.Time(:)-high_ts.Time(1);
    time=time(range,:);
    
    % Wheel Steering Angle(delta) for high speed circuit can data
    high_steering = select(high_can,'Topic','/groot/chassis_can/platform_standard');
    high_msgStructs = readMessages(high_steering,'DataFormat','struct');
    high_input_data_deg = cellfun(@(m) double(m.WsaDeg),high_msgStructs);
    high_input_data_rad=deg2rad(high_input_data_deg);
%     high_input_data=high_input_data_rad(1:length(time),:);
    high_input_data=high_input_data_rad(range,:);
    high_min_WSA = min(high_input_data);
    high_max_WSA = max(high_input_data);
    
    % Wheel Steering Angle(delta) for low speed K-city can data
    Kcity_steering = select(Kcity_can,'Topic','/groot/chassis_can/platform_standard');
    Kcity_msgStructs = readMessages(Kcity_steering,'DataFormat','struct');
    Kcity_input_data_deg = cellfun(@(m) double(m.WsaDeg),Kcity_msgStructs);
    Kcity_input_data_rad=deg2rad(Kcity_input_data_deg);
%     Kcity_input_data=Kcity_input_data_rad(1:length(time),:);
    Kcity_input_data=Kcity_input_data_rad(range,:);
    Kcity_min_WSA = min(Kcity_input_data);
    Kcity_max_WSA = max(Kcity_input_data);
    
    figure('Name',name);
    plot(time,high_Long_vel,'r','LineWidth', 2)    
    hold on
    grid on
    plot(time,Kcity_Long_vel,'b','LineWidth', 2)
    title('Longitudinal Velocity, V_{x}','FontSize',13)
    legend('고속 주회로','K-city')
    xlabel('Time [s]','FontSize',10,'FontName','Times New Roman')
    ylabel('Velocity [m/s]','FontSize',10,'FontName','Times New Roman')
    
    figure('Name',name);
    plot(time,high_input_data,'r','LineWidth', 2)
    hold on
    grid on
    plot(time,Kcity_input_data,'b','LineWidth', 2)
    title('Wheel Steering Angle, \delta','FontSize',13)
    legend('고속 주회로','K-city')
    xlabel('Time [s]','FontSize',10,'FontName','Times New Roman')
    ylabel('Angle [rad]','FontSize',10,'FontName','Times New Roman')

end

