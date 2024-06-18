function [min_Long_vel, max_Long_vel, min_Lat_acc_y, max_Lat_acc_y, min_WSA, max_WSA] = moreinfo_generation_function(gt_data,can_data,name)
    gt_bag = rosbag(gt_data);
    can_bag = rosbag(can_data);
    %%%%%%%%%%%%%%%%%%%%%%%%% gt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %gt Longitudinal velocity(Vx)
    gt_vel=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/vel");
    ts=timeseries(gt_vel,'Vector.X');
    Long_vel=ts.Data(:);
    min_Long_vel = min(Long_vel);
    max_Long_vel = max(Long_vel);

    % time
    time=ts.Time(:)-ts.Time(1);

    %gt Acceleration(a_y)
    gt_acc = select(gt_bag,'Topic','/gt/acc');
    msgStructs1 = readMessages(gt_acc,'DataFormat','struct');
    Lat_acc_y = cellfun(@(m) double(m.Vector.Y),msgStructs1);
    min_Lat_acc_y = min(Lat_acc_y);
    max_Lat_acc_y = max(Lat_acc_y);
    
    steering = select(can_bag,'Topic','/groot/chassis_can/platform_standard');
    msgStructs4 = readMessages(steering,'DataFormat','struct');
    msgStructs4{1};
    input_data_deg = cellfun(@(m) double(m.WsaDeg),msgStructs4);
%     input_data = cellfun(@(m) double(m.SwaDeg),msgStructs4);
    input_data_rad=deg2rad(input_data_deg);
    input_data=input_data_rad(1:length(time),:);
    min_WSA = min(input_data);
    max_WSA = max(input_data);
    size(Long_vel)
    size(Lat_acc_y)
    size(input_data_rad)
    size(input_data)
    size(time)
    length(Lat_acc_y)
    
%     more_info = cat(2,Long_vel,Lat_acc_y,input_data);

    
    figure('Name',name);
    subplot(3,1,1);
    plot(time,Long_vel)    
    title('Longitudinal Velocity gt','FontSize',10)
    xlabel('Time[s]')
    ylabel('Vx [m/s]')
    
    subplot(3,1,2);
    plot(time,Lat_acc_y)
    title('Lateral Acceleration gt','FontSize',10)
    xlabel('Time[s]')
    ylabel('ACC_{y} [m/s$^2$]')
    
    subplot(3,1,3);
    plot(time,input_data)
    hold on
    title('WSA_{rad}','FontSize',10)
    xlabel('Time[s]')
    ylabel('δ [rad]')

end

