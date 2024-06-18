function analFullmotion(gt_data,input_data,dataname)

    %전체 시간 데이터 범위에서 종방향 속도와 WSA deg 분석 
    gt_bag = rosbag(gt_data);
    gt_vel_x=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/vel");
    ts_x=timeseries(gt_vel_x,'Vector.X');      %종방향 속도
    x_dot=ts_x.Data(:);
    time_x=ts_x.Time(:)-ts_x.Time(1);
    
    figure('Name',[dataname ' original full data'])
    subplot(2,1,1)
    plot(time_x,x_dot,'r')
    grid on
    title('Longitudinal Velocity','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('dx/dt [m/s]','FontSize',10)
    
    subplot(2,1,2)
    plot(time_x,input_data,'r')
    grid on
    title('Wheel Steering Angle','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('δ [rad]','FontSize',10)