function [minmax_input, minmax_vel_x]= analLocalmotion(gt_data,steeringinput,range)
    %전체 시간 범위가 아닌 일부의 선택된 범위의 시간에서 종방향 속도와 WSA deg 분석  

%     comparison_range = comparison_index-499:(comparison_index+499);

    gt_bag = rosbag(gt_data);
    gt_vel=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/vel");
    ts=timeseries(gt_vel,'Vector.X');      %종방향 속도
    x_dot=ts.Data(:);
    gt_time=ts.Time(:)-ts.Time(1);

    figure('Name','high-speed circuit data analysis')
    subplot(2,1,1)
%     plot(gt_time(comparison_range),x_dot(comparison_range),'b')
    plot(gt_time(range),x_dot(range),'b')
    grid on
    title('Longitudinal Velocity','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('dx/dt [m/s]','FontSize',10)
    
    subplot(2,1,2)
%     plot(gt_time(comparison_range),steeringinput(comparison_range),'b')
    plot(gt_time(range),steeringinput(range),'b')
    grid on
    title('Wheel Steering Angle','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('δ [rad]','FontSize',10)
    
    
%     min(time(comparison_range))
%     max(time(comparison_range))
    
%     minmax_input = [min(steeringinput(comparison_range)) max(steeringinput(comparison_range))];
%     minmax_vel_x = [min(x_dot(comparison_range)) max(x_dot(comparison_range))];

    minmax_input = [min(steeringinput(range)) max(steeringinput(range))];
    minmax_vel_x = [min(x_dot(range)) max(x_dot(range))];
    
end

    
